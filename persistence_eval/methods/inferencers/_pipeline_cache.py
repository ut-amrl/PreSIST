"""
On-disk cache for the three LLM-pipeline phases (scene caption, object
descriptions, persistence-querier output). Used by ``LLMPipelineInferencer``.

Cache layout under ``cache_dir``:

    scene_caption/<captioner_name>/<image_id>.json
    object_descriptions/<describer_name>/<image_id>/<track_id>.json
    persistence/<querier_name>/<image_id>/<group_token>__<upstream_hash>.json

Each JSON blob carries an ``elapsed_s`` field with the wall time the original
computation took (split per-item for batched describer calls). On a cache hit
we replay that time into the global ``timer`` under the same name the live
call would have used, so timing reports remain meaningful even when most calls
are served from cache.

Cache key correctness:
- Scene caption keyed by (captioner_name, image_id) — captioner_name must be
  unique per captioner config.
- Object descriptions keyed by (describer_name, image_id, track_id) — same
  rule; describer_name must encode the describer's config.
- Persistence keyed by (querier_name, image_id, sorted track_ids,
  ``upstream_hash``) where ``upstream_hash`` is a short sha1 of the actual
  scene_caption + object_descriptions used. Changing the captioner or
  describer (and thus their outputs) naturally produces a new persistence
  entry without requiring callers to bump the querier name or wipe the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.timing import timer


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def _safe(name: str) -> str:
    """Map an arbitrary identifier to a filesystem-safe name."""
    s = _SAFE_NAME_RE.sub("_", str(name)).strip("_")
    return s or "_"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[pipeline_cache] failed to read {path}: {e}; ignoring cached entry")
        return None


def group_token(track_ids: Iterable[str]) -> str:
    """Stable per-group cache token: sorted track_ids joined, plus a short
    hash so the filename can't collide if the same sorted list is somehow
    constructed differently (e.g. with extra whitespace).
    """
    ids_sorted = sorted(str(t) for t in track_ids)
    joined = "|".join(ids_sorted)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    # Cap the readable prefix so very large groups don't blow up filenames.
    readable = "_".join(_safe(t) for t in ids_sorted[:6])
    if len(ids_sorted) > 6:
        readable += f"_plus{len(ids_sorted) - 6}"
    return f"{readable}__{digest}"


VALID_PHASES = frozenset({"scene_caption", "object_descriptions", "persistence"})


class PipelineCache:
    def __init__(
        self,
        cache_dir: Path,
        *,
        force_regen: bool = False,
        force_regen_phases: Optional[Iterable[str]] = None,
    ):
        """``force_regen`` skips reads for every phase (writes still happen so
        the next run can hit cache). ``force_regen_phases`` is a per-phase
        allow-list of names to force-regenerate; when set (non-empty), only
        those phases regenerate and ``force_regen`` is ignored. Valid phase
        names: ``scene_caption``, ``object_descriptions``, ``persistence``.
        """
        self.cache_dir = Path(cache_dir)
        if force_regen_phases:
            phases = {str(p) for p in force_regen_phases}
            unknown = phases - VALID_PHASES
            if unknown:
                raise ValueError(
                    f"Unknown force_regen_phases entries: {sorted(unknown)}. "
                    f"Valid phases: {sorted(VALID_PHASES)}"
                )
            self._force_regen_phases = phases
            self._force_regen_all = False
        else:
            self._force_regen_phases = set()
            self._force_regen_all = bool(force_regen)

    def _should_regen(self, phase: str) -> bool:
        if self._force_regen_phases:
            return phase in self._force_regen_phases
        return self._force_regen_all

    # -------------------------------------------------------- scene caption

    def _scene_path(self, captioner_name: str, image_id: str) -> Path:
        return (
            self.cache_dir
            / "scene_caption"
            / _safe(captioner_name)
            / f"{_safe(image_id)}.json"
        )

    def get_scene_caption(self, captioner_name: str, image_id: str) -> Optional[str]:
        if self._should_regen("scene_caption"):
            return None
        blob = _read_json(self._scene_path(captioner_name, image_id))
        if blob is None:
            return None
        return blob.get("value")

    def put_scene_caption(
        self,
        captioner_name: str,
        image_id: str,
        *,
        caption: str,
        elapsed_s: float,
    ) -> None:
        _atomic_write_json(
            self._scene_path(captioner_name, image_id),
            {
                "image_id": image_id,
                "captioner_name": captioner_name,
                "value": caption,
                "elapsed_s": elapsed_s,
            },
        )

    def cached_scene_caption_elapsed(self, captioner_name: str, image_id: str) -> Optional[float]:
        if self._should_regen("scene_caption"):
            return None
        blob = _read_json(self._scene_path(captioner_name, image_id))
        return None if blob is None else float(blob.get("elapsed_s", 0.0))

    # ---------------------------------------------------- object descriptions

    def _object_path(self, describer_name: str, image_id: str, track_id: str) -> Path:
        return (
            self.cache_dir
            / "object_descriptions"
            / _safe(describer_name)
            / _safe(image_id)
            / f"{_safe(track_id)}.json"
        )

    def get_object_description(
        self, describer_name: str, image_id: str, track_id: str,
    ) -> Optional[Tuple[Any, float]]:
        """Returns (description, elapsed_s) or None if missing."""
        if self._should_regen("object_descriptions"):
            return None
        blob = _read_json(self._object_path(describer_name, image_id, track_id))
        if blob is None:
            return None
        return blob.get("description"), float(blob.get("elapsed_s", 0.0))

    def put_object_description(
        self,
        describer_name: str,
        image_id: str,
        track_id: str,
        *,
        description: Any,
        elapsed_s: float,
    ) -> None:
        _atomic_write_json(
            self._object_path(describer_name, image_id, track_id),
            {
                "image_id": image_id,
                "describer_name": describer_name,
                "track_id": track_id,
                "description": description,
                "elapsed_s": elapsed_s,
            },
        )

    # -------------------------------------------------------- persistence

    def _persistence_path(
        self,
        querier_name: str,
        image_id: str,
        token: str,
        upstream_hash: str,
    ) -> Path:
        return (
            self.cache_dir
            / "persistence"
            / _safe(querier_name)
            / _safe(image_id)
            / f"{token}__{upstream_hash}.json"
        )

    def get_persistence(
        self,
        querier_name: str,
        image_id: str,
        track_ids: Iterable[str],
        *,
        scene_caption: str,
        object_descriptions: Dict[Any, Any],
    ) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """Returns (raw_estimates, elapsed_s) or None if missing.

        ``scene_caption`` and ``object_descriptions`` are folded into the
        cache key so any change to upstream phase outputs invalidates this
        entry automatically.
        """
        if self._should_regen("persistence"):
            return None
        token = group_token(track_ids)
        upstream_hash = upstream_fingerprint(scene_caption, object_descriptions)
        blob = _read_json(self._persistence_path(querier_name, image_id, token, upstream_hash))
        if blob is None:
            return None
        return list(blob.get("raw_estimates", [])), float(blob.get("elapsed_s", 0.0))

    def put_persistence(
        self,
        querier_name: str,
        image_id: str,
        track_ids: Iterable[str],
        *,
        scene_caption: str,
        object_descriptions: Dict[Any, Any],
        raw_estimates: List[Dict[str, Any]],
        elapsed_s: float,
    ) -> None:
        ids_list = [str(t) for t in track_ids]
        token = group_token(ids_list)
        upstream_hash = upstream_fingerprint(scene_caption, object_descriptions)
        _atomic_write_json(
            self._persistence_path(querier_name, image_id, token, upstream_hash),
            {
                "image_id": image_id,
                "querier_name": querier_name,
                "track_ids": ids_list,
                "upstream_fingerprint": {
                    "scene_caption_sha1": hashlib.sha1(scene_caption.encode("utf-8")).hexdigest(),
                    "object_descriptions_sha1": _hash_json_canonical(object_descriptions),
                    "combined_short": upstream_hash,
                },
                "raw_estimates": raw_estimates,
                "elapsed_s": elapsed_s,
            },
        )


def upstream_fingerprint(scene_caption: str, object_descriptions: Dict[Any, Any]) -> str:
    """Short hex digest of the upstream phase outputs used by the persistence
    cache key. Stable across processes — keys are sorted before serialization.
    """
    payload = json.dumps(
        {
            "scene_caption": scene_caption,
            "object_descriptions": {str(k): v for k, v in sorted(object_descriptions.items(), key=lambda kv: str(kv[0]))},
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _hash_json_canonical(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def replay_timing(name: str, elapsed_s: float) -> None:
    """Record a cached duration into the global timer so its summary stays
    representative even when most calls are served from cache.
    """
    timer.record(name, float(elapsed_s), success=True)
