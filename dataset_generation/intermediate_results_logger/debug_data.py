from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional


_VALID_KINDS = ("image", "numpy", "tensor", "json", "text", "pickle")
_KIND_EXT = {
    "image": ".png",
    "numpy": ".npy",
    "tensor": ".pt",
    "json": ".json",
    "text": ".txt",
    "pickle": ".pkl",
}

_MANIFEST_FILENAME = "_manifest.json"
_MANIFEST_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Debug entry name must be a non-empty string, got {name!r}")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        raise ValueError(f"Debug entry name {name!r} sanitizes to empty string")
    return cleaned


def _detect_kind(value: Any) -> str:
    try:
        from PIL import Image as _PILImage
        if isinstance(value, _PILImage.Image):
            return "image"
    except ImportError:
        pass

    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            if (
                value.ndim == 3
                and value.shape[-1] in (1, 3, 4)
                and value.dtype == _np.uint8
            ):
                return "image"
            return "numpy"
    except ImportError:
        pass

    try:
        import torch as _torch
        if isinstance(value, _torch.Tensor):
            return "tensor"
    except ImportError:
        pass

    if isinstance(value, str):
        return "text"

    if isinstance(value, (dict, list, tuple)):
        try:
            json.dumps(value)
            return "json"
        except TypeError:
            pass

    return "pickle"


@dataclass
class DebugEntryRecord:
    name: str
    value: Any
    kind: str
    # Caller's original defer choice — preserved on the record so a transient
    # leaf payload's policy is honored when later merged into a configured
    # parent. None = use the parent's smart default at write-decision time.
    defer: Optional[bool] = None


@dataclass
class DebugEntry:
    name: str
    kind: str
    path: str


class DebugPayload:
    """Per-(stage, image) debug entry container.

    Three modes:
      * Disabled (``enabled=False``): all add/merge calls are no-ops. The
        per-object PIL refs the caller passes in are dropped immediately, so
        debug-off runs don't pin memory.
      * Transient/buffer-only (``enabled=True``, ``target_dir=None``): add
        appends to ``_entries``; nothing is written. Used by leaf queriers
        that return their payload upward to be merged into a configured parent.
      * Configured (``enabled=True``, ``target_dir`` set): add either writes
        the entry to disk immediately or buffers it for ``flush()``, per the
        ``defer`` flag; ``flush()`` writes deferred entries and a single
        ``_manifest.json`` covering both immediate and deferred entries.

    Built via ``DebugPayloadFactory.instance().create(phase, image_id)`` at
    the same site where ``DebugPayload()`` is constructed today.
    """

    def __init__(
        self,
        *,
        target_dir: Optional[str] = None,
        kinds_filter: Optional[set[str]] = None,
        enabled: bool = True,
    ) -> None:
        self._entries: list[DebugEntryRecord] = []
        self._used_names: set[str] = set()
        self._merge_names: set[str] = set()
        self.target_dir = target_dir
        self.kinds_filter = kinds_filter
        self.enabled = enabled
        # Records of entries already written to disk (immediate-write path),
        # used to assemble the final manifest at flush time.
        self._immediate_manifest: list[dict] = []
        self._used_filenames: set[str] = set()

    @staticmethod
    def _resolve_write_now(defer: Optional[bool], kind: str) -> bool:
        if defer is True:
            return True
        if defer is False:
            return False
        # Smart default: image kind writes immediately (memory-heavy);
        # other kinds defer (cheap, batched into one manifest write).
        return kind == "image"

    def _allocate_filename(self, name: str, kind: str) -> str:
        stem = _sanitize_name(name)
        ext = _KIND_EXT[kind]
        filename = f"{stem}{ext}"
        collision = 2
        while filename in self._used_filenames:
            filename = f"{stem}_{collision}{ext}"
            collision += 1
        self._used_filenames.add(filename)
        return filename

    def _write_immediate(self, name: str, value: Any, kind: str) -> None:
        if self.kinds_filter is not None and kind not in self.kinds_filter:
            return
        os.makedirs(self.target_dir, exist_ok=True)
        filename = self._allocate_filename(name, kind)
        file_path = os.path.join(self.target_dir, filename)
        try:
            DebugWriter._write_entry(kind, value, file_path)
        except Exception as e:
            print(
                f"[debug-data] failed to write immediate entry {name!r} ({kind}): {e}"
            )
            return
        self._immediate_manifest.append({
            "name": name,
            "kind": kind,
            "filename": filename,
            "written_at": _utc_now_iso(),
        })

    def add(
        self,
        name: str,
        value: Any,
        kind: Optional[str] = None,
        defer: Optional[bool] = None,
    ) -> None:
        if not self.enabled:
            return

        if kind is None:
            kind = _detect_kind(value)
        elif kind not in _VALID_KINDS:
            raise ValueError(
                f"Invalid debug kind {kind!r}; must be one of {_VALID_KINDS}"
            )

        unique_name = name
        suffix = 2
        while unique_name in self._used_names:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        self._used_names.add(unique_name)

        # Transient mode (no target_dir): always buffer. Remember caller's
        # defer choice so a later merge into a configured parent honors it.
        if self.target_dir is None:
            self._entries.append(
                DebugEntryRecord(name=unique_name, value=value, kind=kind, defer=defer)
            )
            return

        if self._resolve_write_now(defer, kind):
            self._write_immediate(unique_name, value, kind)
        else:
            self._entries.append(
                DebugEntryRecord(name=unique_name, value=value, kind=kind, defer=defer)
            )

    def merge_json(
        self,
        name: str,
        value: dict,
        defer: Optional[bool] = None,
    ) -> None:
        if not self.enabled:
            return

        if not isinstance(value, dict):
            raise ValueError(
                f"merge_json requires a dict value; got {type(value).__name__}"
            )

        # merge_json always buffers — its whole point is to allow accumulation
        # of further dicts into the same entry; an immediately-written entry
        # would be unreachable for the next merge_json call. The defer arg is
        # preserved on the record only so a downstream merge into a configured
        # parent can re-evaluate (it'll still buffer, but the policy is intact).

        if name in self._merge_names:
            for rec in self._entries:
                if rec.name == name:
                    if not isinstance(rec.value, dict) or rec.kind != "json":
                        raise ValueError(
                            f"Existing entry {name!r} is not a merge-bound JSON dict"
                        )
                    rec.value.update(value)
                    return
            raise ValueError(f"merge-bound name {name!r} has no backing entry")

        if name in self._used_names:
            raise ValueError(
                f"Cannot merge_json onto name {name!r}: already added via add()"
            )

        self._used_names.add(name)
        self._merge_names.add(name)
        self._entries.append(
            DebugEntryRecord(name=name, value=dict(value), kind="json", defer=defer)
        )

    def merge(self, other: Optional["DebugPayload"]) -> None:
        if other is None or not self.enabled:
            return
        for rec in other._entries:
            if (
                rec.name in self._merge_names
                and rec.name in other._merge_names
                and isinstance(rec.value, dict)
            ):
                self.merge_json(rec.name, rec.value, defer=rec.defer)
            elif (
                rec.name in other._merge_names
                and rec.name not in self._used_names
                and isinstance(rec.value, dict)
            ):
                self.merge_json(rec.name, rec.value, defer=rec.defer)
            else:
                self.add(rec.name, rec.value, kind=rec.kind, defer=rec.defer)

    def extend(self, other: Optional["DebugPayload"]) -> None:
        self.merge(other)

    def flush(self) -> Optional[str]:
        """Write deferred entries and the combined manifest. No-op when
        disabled, when transient (no target_dir), or when there's nothing to
        write."""
        if not self.enabled or self.target_dir is None:
            return None
        if not self._entries and not self._immediate_manifest:
            return None

        deferred_manifest: list[dict] = []
        if self._entries:
            os.makedirs(self.target_dir, exist_ok=True)
            for rec in self._entries:
                if self.kinds_filter is not None and rec.kind not in self.kinds_filter:
                    continue
                filename = self._allocate_filename(rec.name, rec.kind)
                file_path = os.path.join(self.target_dir, filename)
                try:
                    DebugWriter._write_entry(rec.kind, rec.value, file_path)
                except Exception as e:
                    print(
                        f"[debug-data] failed to write deferred entry "
                        f"{rec.name!r} ({rec.kind}): {e}"
                    )
                    continue
                deferred_manifest.append({
                    "name": rec.name,
                    "kind": rec.kind,
                    "filename": filename,
                    "written_at": _utc_now_iso(),
                })
            self._entries.clear()

        all_entries = self._immediate_manifest + deferred_manifest
        if not all_entries:
            return None

        manifest_path = os.path.join(self.target_dir, _MANIFEST_FILENAME)
        manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "written_at": _utc_now_iso(),
            "entries": all_entries,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest_path

    def is_empty(self) -> bool:
        return not self._entries and not self._immediate_manifest

    def __iter__(self) -> Iterator[DebugEntryRecord]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def empty() -> "DebugPayload":
        return DebugPayload()

    @staticmethod
    def combine(*payloads: "DebugPayload") -> "DebugPayload":
        result = DebugPayload()
        for p in payloads:
            result.merge(p)
        return result


@dataclass
class DebugConfig:
    enabled: bool = False
    phases: Optional[list[str]] = None
    kinds: Optional[list[str]] = None

    def is_phase_enabled(self, phase: str) -> bool:
        return self.phases is None or phase in self.phases

    def kinds_filter(self) -> Optional[set[str]]:
        return None if self.kinds is None else set(self.kinds)


class DebugPayloadFactory:
    """Singleton that produces configured (or disabled) DebugPayloads.

    Configured once per run by the logger; stage code calls
    ``DebugPayloadFactory.instance().create(phase_name, image_id)`` at the
    site where ``DebugPayload()`` is constructed today. Holds only static
    per-run state (config, base dir, component names) — no per-image state,
    so it's safe to share across threads.
    """

    _instance: Optional["DebugPayloadFactory"] = None

    # Per-phase debug-dir layout. Each value is a function from
    # (component_names, image_id) to the list of path parts under
    # base_log_directory. Mirrors the dir scheme on the logger
    # (CachingResultsLoggerRevised.get*ResultsDirectory) and the read-side
    # layout in DebugDataReader._PHASE_LAYOUT.
    _PHASE_DIR_PARTS = {
        "segmentation": lambda c, image_id: [
            "segmentation_results", c["segmenter"], image_id, "debug",
        ],
        "scene_caption": lambda c, image_id: [
            "scene_caption_results", c["sceneCaptioner"], image_id, "debug",
        ],
        "object_descriptions": lambda c, image_id: [
            "object_description_results",
            c["segmenter"], c["objectDescriber"], image_id, "debug",
        ],
        "relationships": lambda c, image_id: [
            "relationship_description_results",
            c["segmenter"], c["objectDescriber"], c["relationshipDescriber"],
            image_id, "debug",
        ],
        "persistence": lambda c, image_id: [
            "full_results", c["configFileName"], image_id, "debug",
        ],
    }

    @classmethod
    def instance(cls) -> "DebugPayloadFactory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._debug_config: Optional[DebugConfig] = None
        self._base_log_directory: Optional[str] = None
        self._component_names: dict = {}

    def configure(
        self,
        *,
        debug_config: DebugConfig,
        base_log_directory: str,
        config_file_name: str,
        component_names: dict,
    ) -> None:
        self._debug_config = debug_config
        self._base_log_directory = base_log_directory
        self._component_names = dict(component_names)
        # configFileName is part of the persistence-phase dir layout; carry it
        # in the same dict so _PHASE_DIR_PARTS only has to look at one place.
        self._component_names["configFileName"] = config_file_name

    def is_enabled_for(self, phase_name: str) -> bool:
        if self._debug_config is None or not self._debug_config.enabled:
            return False
        return self._debug_config.is_phase_enabled(phase_name)

    def create(self, phase_name: str, image_id: str) -> DebugPayload:
        if not self.is_enabled_for(phase_name):
            return DebugPayload(enabled=False)
        if phase_name not in self._PHASE_DIR_PARTS:
            raise ValueError(
                f"Unknown phase name {phase_name!r}; expected one of "
                f"{sorted(self._PHASE_DIR_PARTS)}"
            )
        parts = self._PHASE_DIR_PARTS[phase_name](
            self._component_names, image_id
        )
        target_dir = os.path.join(self._base_log_directory, *parts)
        return DebugPayload(
            target_dir=target_dir,
            kinds_filter=self._debug_config.kinds_filter(),
            enabled=True,
        )


class DebugWriter:
    def __init__(self, target_dir: str) -> None:
        self.target_dir = target_dir

    def write(
        self,
        payload: DebugPayload,
        *,
        kinds_filter: Optional[set[str]] = None,
    ) -> Optional[str]:
        if payload is None or payload.is_empty():
            return None

        entries_to_write: list[DebugEntryRecord] = []
        for rec in payload:
            if kinds_filter is not None and rec.kind not in kinds_filter:
                continue
            entries_to_write.append(rec)
        if not entries_to_write:
            return None

        os.makedirs(self.target_dir, exist_ok=True)

        manifest_entries: list[dict] = []
        used_filenames: set[str] = set()

        for rec in entries_to_write:
            stem = _sanitize_name(rec.name)
            ext = _KIND_EXT[rec.kind]
            filename = f"{stem}{ext}"
            collision = 2
            while filename in used_filenames:
                filename = f"{stem}_{collision}{ext}"
                collision += 1
            used_filenames.add(filename)
            file_path = os.path.join(self.target_dir, filename)

            try:
                self._write_entry(rec.kind, rec.value, file_path)
            except Exception as e:
                print(
                    f"[debug-data] failed to serialize entry {rec.name!r} ({rec.kind}): {e}"
                )
                continue

            manifest_entries.append(
                {
                    "name": rec.name,
                    "kind": rec.kind,
                    "filename": filename,
                    "written_at": _utc_now_iso(),
                }
            )

        manifest_path = os.path.join(self.target_dir, _MANIFEST_FILENAME)
        manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "written_at": _utc_now_iso(),
            "entries": manifest_entries,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest_path

    @staticmethod
    def _write_entry(kind: str, value: Any, path: str) -> None:
        if kind == "image":
            from PIL import Image as _PILImage
            if isinstance(value, _PILImage.Image):
                value.save(path)
                return
            try:
                import numpy as _np
                if isinstance(value, _np.ndarray):
                    arr = value
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    _PILImage.fromarray(arr).save(path)
                    return
            except ImportError:
                pass
            raise TypeError(
                f"Cannot serialize value of type {type(value)!r} as image"
            )

        if kind == "numpy":
            import numpy as _np
            _np.save(path, value)
            return

        if kind == "tensor":
            import torch as _torch
            tensor = value
            if isinstance(tensor, _torch.Tensor):
                tensor = tensor.detach().cpu()
            _torch.save(tensor, path)
            return

        if kind == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            return

        if kind == "text":
            with open(path, "w", encoding="utf-8") as f:
                f.write(value)
            return

        if kind == "pickle":
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        raise ValueError(f"Unknown debug kind: {kind!r}")


class DebugDataReader:
    _PHASE_LAYOUT = {
        "segmentation":         ("segmentation_results", 1),
        "scene_caption":        ("scene_caption_results", 1),
        "object_descriptions":  ("object_description_results", 2),
        "relationships":        ("relationship_description_results", 3),
        "persistence":          ("full_results", 1),
    }

    def __init__(
        self,
        base_log_directory: str,
        config_file_name: Optional[str] = None,
    ) -> None:
        self.base = base_log_directory
        self.config_file_name = config_file_name

    def list_phases(self) -> list[str]:
        present = []
        for phase, (root_dir, _) in self._PHASE_LAYOUT.items():
            if os.path.isdir(os.path.join(self.base, root_dir)):
                present.append(phase)
        return present

    def list_image_ids(self, phase: str) -> list[str]:
        ids: set[str] = set()
        for image_dir in self._iter_image_dirs(phase):
            image_id = os.path.basename(image_dir.rstrip(os.sep))
            if os.path.isdir(os.path.join(image_dir, "debug")):
                ids.add(image_id)
        return sorted(ids)

    def list_entries(self, phase: str, image_id: str) -> list[DebugEntry]:
        entries: list[DebugEntry] = []
        for image_dir in self._iter_image_dirs(phase):
            if os.path.basename(image_dir.rstrip(os.sep)) != image_id:
                continue
            debug_dir = os.path.join(image_dir, "debug")
            manifest = self._read_manifest(debug_dir)
            if manifest is None:
                continue
            for entry in manifest.get("entries", []):
                entries.append(
                    DebugEntry(
                        name=entry["name"],
                        kind=entry["kind"],
                        path=os.path.join(debug_dir, entry["filename"]),
                    )
                )
        return entries

    def load(self, phase: str, image_id: str, name: str) -> Any:
        for entry in self.list_entries(phase, image_id):
            if entry.name == name:
                return self._load_entry(entry.kind, entry.path)
        raise KeyError(
            f"Debug entry not found: phase={phase!r}, image_id={image_id!r}, name={name!r}"
        )

    def _iter_image_dirs(self, phase: str) -> Iterable[str]:
        if phase not in self._PHASE_LAYOUT:
            raise ValueError(f"Unknown phase: {phase!r}")
        root_dir, depth_before_image_id = self._PHASE_LAYOUT[phase]
        root_path = os.path.join(self.base, root_dir)
        if not os.path.isdir(root_path):
            return

        if phase == "persistence" and self.config_file_name is not None:
            yield from self._iter_at_depth(
                os.path.join(root_path, self.config_file_name), 0
            )
            return

        yield from self._iter_at_depth(root_path, depth_before_image_id)

    def _iter_at_depth(self, current_dir: str, remaining_depth: int) -> Iterable[str]:
        if not os.path.isdir(current_dir):
            return
        if remaining_depth == 0:
            for child in sorted(os.listdir(current_dir)):
                child_path = os.path.join(current_dir, child)
                if os.path.isdir(child_path):
                    yield child_path
            return
        for child in sorted(os.listdir(current_dir)):
            child_path = os.path.join(current_dir, child)
            if os.path.isdir(child_path):
                yield from self._iter_at_depth(child_path, remaining_depth - 1)

    @staticmethod
    def _read_manifest(debug_dir: str) -> Optional[dict]:
        manifest_path = os.path.join(debug_dir, _MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[debug-data] failed to read manifest at {manifest_path}: {e}")
            return None

    @staticmethod
    def _load_entry(kind: str, path: str) -> Any:
        if kind == "image":
            from PIL import Image as _PILImage
            return _PILImage.open(path)
        if kind == "numpy":
            import numpy as _np
            return _np.load(path)
        if kind == "tensor":
            import torch as _torch
            return _torch.load(path, map_location="cpu")
        if kind == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if kind == "text":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        if kind == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"Unknown debug kind: {kind!r}")
