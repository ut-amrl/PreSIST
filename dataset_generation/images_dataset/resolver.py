# image_id_pipeline/resolver.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResolvedItem:
    """
    One resolved dataset item.

    relpath is a forward-slash relative path under image_root (e.g. "A/subset2/img.png")
    component is a provenance label (e.g. "A_train", "B_val", or "list").
    """
    relpath: str
    component: str


def _normalize_relpath(relpath: str) -> str:
    rel = relpath.replace("\\", "/").lstrip("/")
    if rel == "" or rel == ".":
        raise ValueError("Empty relpath")
    if rel == ".." or rel.startswith("../") or "/../" in rel:
        raise ValueError(f"Refusing relpath with parent traversal: {relpath}")
    return rel


def _read_relpaths_txt(path: Path) -> List[str]:
    relpaths: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        relpaths.append(_normalize_relpath(line))
    return relpaths


def _load_structured_spec(spec_path: Path) -> Dict[str, Any]:
    suffix = spec_path.suffix.lower()
    if suffix == ".json":
        return json.loads(spec_path.read_text(encoding="utf-8"))

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML spec provided but PyYAML is not installed. "
                "Install pyyaml or use JSON."
            ) from e
        return yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported spec suffix: {suffix}")


def resolve_paths(
    *,
    image_root: str,
    spec_path: str,
    lists_root: Optional[str] = None,
) -> Tuple[List[ResolvedItem], Dict[str, Any]]:
    """
    Resolve a dataset spec into concrete relpaths.

    Modes:
      1) List mode (.txt):
          spec_path is a text file with relpaths under image_root.

      2) Config mode (.json/.yaml/.yml):
          spec has:
            dataset_name: str (optional)
            components:
              - name: str
                root_subdir: str
                list: str   # list name (without .txt) OR filename (with .txt)
            dedupe: by_path | by_image_id (optional; default by_path)

          Each list file lives at:
            <lists_root>/<root_subdir>/<list>.txt   (if list has no suffix)
          and contains relpaths *within that root_subdir*.

          Final resolved relpath becomes:
            <root_subdir>/<line_from_list_file>
    """
    root = Path(image_root).resolve()
    spec_p = Path(spec_path)

    suffix = spec_p.suffix.lower()

    # -----------------------
    # Mode A: plain list
    # -----------------------
    if suffix == ".txt":
        rels = _read_relpaths_txt(spec_p)
        items = [ResolvedItem(relpath=r, component="list") for r in rels]
        meta: Dict[str, Any] = {
            "kind": "list",
            "image_root": str(root),
            "spec_path": str(spec_p),
            "dedupe": "by_path",
        }
        return items, meta

    # -----------------------
    # Mode B: structured config
    # -----------------------
    if suffix in (".json", ".yaml", ".yml"):
        if not lists_root:
            raise ValueError("lists_root is required for config mode (.json/.yaml/.yml).")

        cfg = _load_structured_spec(spec_p) or {}
        components = cfg.get("components") or []
        if not isinstance(components, list) or not components:
            raise ValueError("Config spec must contain a non-empty 'components' list.")

        dedupe = cfg.get("dedupe", "by_path")
        if dedupe not in ("by_path", "by_image_id"):
            raise ValueError(f"Unknown dedupe policy in spec: {dedupe}")

        lr = Path(lists_root).resolve()

        out: List[ResolvedItem] = []
        for comp in components:
            if not isinstance(comp, dict):
                raise ValueError("Each component must be a mapping/dict.")
            name = comp.get("name")
            root_subdir = comp.get("root_subdir")
            list_name = comp.get("list")

            if not name or not root_subdir or not list_name:
                raise ValueError("Each component must have 'name', 'root_subdir', and 'list'.")

            root_subdir_norm = _normalize_relpath(str(root_subdir))

            list_file = str(list_name)
            if not list_file.lower().endswith(".txt"):
                list_file = list_file + ".txt"

            list_path = lr / root_subdir_norm / list_file
            rels_in_component = _read_relpaths_txt(list_path)

            for rel_in_root in rels_in_component:
                rel_final = _normalize_relpath(f"{root_subdir_norm}/{rel_in_root}")
                out.append(ResolvedItem(relpath=rel_final, component=str(name)))

        meta2: Dict[str, Any] = {
            "kind": "config",
            "image_root": str(root),
            "spec_path": str(spec_p),
            "lists_root": str(lr),
            "dataset_name": cfg.get("dataset_name"),
            "dedupe": dedupe,
            "num_components": len(components),
            "num_items": len(out),
        }
        return out, meta2

    raise ValueError(
        f"Unsupported spec_path suffix '{suffix}'. "
        "Use .txt (list mode) or .json/.yaml/.yml (config mode)."
    )


# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# from .registry import normalize_relpath


# @dataclass(frozen=True)
# class ResolvedItem:
#     relpath: str          # relative to image_root
#     component: str        # provenance label


# def _read_lines_txt(path: Path) -> List[str]:
#     out: List[str] = []
#     for raw in path.read_text(encoding="utf-8").splitlines():
#         line = raw.strip()
#         if not line or line.startswith("#"):
#             continue
#         out.append(line.replace("\\", "/").lstrip("/"))
#     return out


# def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
#     suffix = path.suffix.lower()
#     if suffix == ".json":
#         return json.loads(path.read_text(encoding="utf-8"))

#     if suffix in (".yml", ".yaml"):
#         try:
#             import yaml  # type: ignore
#         except ImportError as e:
#             raise ImportError(
#                 "YAML config provided but PyYAML is not installed. "
#                 "Install pyyaml or use JSON."
#             ) from e
#         loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
#         if loaded is None:
#             return {}
#         if not isinstance(loaded, dict):
#             raise ValueError(f"YAML root must be a dict, got: {type(loaded)}")
#         return loaded

#     raise ValueError(f"Unsupported config format: {path} (use .txt, .json, .yaml/.yml)")


# def resolve_paths(
#     *,
#     image_root: str,
#     spec_path: str,
#     lists_root: Optional[str] = None,
# ) -> Tuple[List[ResolvedItem], Dict[str, Any]]:
#     """
#     Returns:
#       - resolved items (relpaths relative to image_root)
#       - metadata dict (mode, dataset_name, dedupe, etc.)

#     spec_path can be:
#       - .txt: a plain list of relpaths under image_root
#       - .json/.yaml: dataset config referencing per-root named lists
#     """
#     image_root_p = Path(image_root).resolve()
#     spec_p = Path(spec_path).resolve()

#     # ---- Mode 1: plain text list of relpaths under image_root ----
#     if spec_p.suffix.lower() == ".txt":
#         relpaths = [normalize_relpath(rp) for rp in _read_lines_txt(spec_p)]
#         items = [ResolvedItem(relpath=rp, component="default") for rp in relpaths]
#         meta = {"mode": "text_list", "dataset_name": spec_p.stem, "dedupe": "by_path"}
#         return items, meta

#     # ---- Mode 2: dataset config referencing per-root named lists ----
#     cfg = _load_json_or_yaml(spec_p)
#     dataset_name = str(cfg.get("dataset_name", spec_p.stem))
#     dedupe = str(cfg.get("dedupe", "by_path"))

#     components = cfg.get("components")
#     if not isinstance(components, list) or not components:
#         raise ValueError("dataset config must have non-empty 'components' list")

#     if lists_root is None:
#         raise ValueError("lists_root is required for dataset config mode")

#     lists_root_p = Path(lists_root).resolve()
#     resolved: List[ResolvedItem] = []

#     for comp in components:
#         if not isinstance(comp, dict):
#             raise ValueError("each component must be a dict")

#         comp_name = str(comp.get("name") or "component")
#         root_subdir = comp.get("root_subdir")
#         list_name = comp.get("list")

#         if not root_subdir or not list_name:
#             raise ValueError("each component must include 'root_subdir' and 'list'")

#         list_file = lists_root_p / str(root_subdir) / f"{str(list_name)}.txt"
#         if not list_file.exists():
#             raise FileNotFoundError(f"List file not found: {list_file}")

#         inner_relpaths = _read_lines_txt(list_file)
#         for inner_rp in inner_relpaths:
#             combined = normalize_relpath(f"{root_subdir}/{inner_rp}")
#             resolved.append(ResolvedItem(relpath=combined, component=comp_name))

#     meta = {
#         "mode": "dataset_config",
#         "dataset_name": dataset_name,
#         "dedupe": dedupe,
#         "spec_path": str(spec_p),
#         "lists_root": str(lists_root_p),
#         "image_root": str(image_root_p),
#     }
#     return resolved, meta



# # from __future__ import annotations

# # import sqlite3
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Optional, List, Tuple


# # SCHEMA_SQL = """
# # PRAGMA journal_mode=WAL;
# # PRAGMA synchronous=NORMAL;

# # CREATE TABLE IF NOT EXISTS images (
# #     root TEXT NOT NULL,
# #     relpath TEXT NOT NULL,
# #     abspath TEXT NOT NULL,
# #     image_id TEXT NOT NULL,
# #     file_size INTEGER NOT NULL,
# #     mtime_ns INTEGER NOT NULL,
# #     indexed_at_ns INTEGER NOT NULL,

# #     PRIMARY KEY (root, relpath)
# # );

# # CREATE INDEX IF NOT EXISTS idx_images_image_id ON images(image_id);
# # """


# # def normalize_relpath(relpath: str) -> str:
# #     rel = relpath.replace("\\", "/").lstrip("/")
# #     # refuse parent traversal
# #     if rel.startswith("../") or "/../" in rel or rel == "..":
# #         raise ValueError(f"Refusing relpath with parent traversal: {relpath}")
# #     return rel


# # @dataclass(frozen=True)
# # class RegistryConfig:
# #     registry_root: Path  # directory
# #     db_name: str = "image_registry.sqlite"


# # class ImageRegistry:
# #     """
# #     Canonical mapping store.
# #     - root + relpath -> image_id
# #     - image_id -> list of (root, relpath, abspath, stats)

# #     Worker-safe design:
# #     - Do NOT share an open sqlite connection across processes.
# #     - We open lazily (per-process) on first use.
# #     """

# #     def __init__(self, cfg: RegistryConfig, *, read_only: bool = False):
# #         self.cfg = cfg
# #         self.read_only = read_only

# #         self.cfg.registry_root.mkdir(parents=True, exist_ok=True)
# #         self.db_path: Path = self.cfg.registry_root / self.cfg.db_name

# #         # connection is opened lazily in this process
# #         self._conn: Optional[sqlite3.Connection] = None

# #         # Ensure schema exists (only if not read-only)
# #         if not self.read_only:
# #             self._ensure_open()
# #             self._init_db()

# #     @classmethod
# #     def from_db_path(cls, db_path: str | Path, *, read_only: bool = True) -> "ImageRegistry":
# #         """
# #         Convenience constructor for datasets/workers that only know the db path.
# #         """
# #         db_path = Path(db_path)
# #         cfg = RegistryConfig(registry_root=db_path.parent, db_name=db_path.name)
# #         return cls(cfg, read_only=read_only)

# #     @property
# #     def conn(self) -> sqlite3.Connection:
# #         self._ensure_open()
# #         assert self._conn is not None
# #         return self._conn

# #     def _ensure_open(self) -> None:
# #         if self._conn is not None:
# #             return

# #         # For read-only usage, prefer opening in read-only mode.
# #         # NOTE: even with read-only, SQLite still uses locks; WAL is fine for concurrent readers.
# #         if self.read_only:
# #             uri = f"file:{self.db_path.as_posix()}?mode=ro"
# #             self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
# #         else:
# #             self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

# #         self._conn.row_factory = sqlite3.Row

# #     def close(self) -> None:
# #         if self._conn is not None:
# #             self._conn.close()
# #             self._conn = None

# #     def _init_db(self) -> None:
# #         cur = self.conn.cursor()
# #         cur.executescript(SCHEMA_SQL)
# #         self.conn.commit()

# #     # ---- Upserts / indexing ----

# #     def get_record(self, root: Path, relpath: str) -> Optional[sqlite3.Row]:
# #         root_s = str(root.resolve())
# #         rel = normalize_relpath(relpath)
# #         cur = self.conn.cursor()
# #         return cur.execute(
# #             "SELECT * FROM images WHERE root=? AND relpath=?",
# #             (root_s, rel),
# #         ).fetchone()

# #     def upsert_record(
# #         self,
# #         root: Path,
# #         relpath: str,
# #         abspath: Path,
# #         image_id: str,
# #         file_size: int,
# #         mtime_ns: int,
# #         indexed_at_ns: int,
# #     ) -> None:
# #         if self.read_only:
# #             raise RuntimeError("ImageRegistry is read-only; cannot upsert_record")

# #         root_s = str(root.resolve())
# #         rel = normalize_relpath(relpath)
# #         cur = self.conn.cursor()
# #         cur.execute(
# #             """
# #             INSERT INTO images (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
# #             VALUES (?, ?, ?, ?, ?, ?, ?)
# #             ON CONFLICT(root, relpath) DO UPDATE SET
# #               abspath=excluded.abspath,
# #               image_id=excluded.image_id,
# #               file_size=excluded.file_size,
# #               mtime_ns=excluded.mtime_ns,
# #               indexed_at_ns=excluded.indexed_at_ns
# #             """,
# #             (root_s, rel, str(abspath), image_id, int(file_size), int(mtime_ns), int(indexed_at_ns)),
# #         )
# #         self.conn.commit()

# #     def bulk_upsert(self, rows: List[Tuple[str, str, str, str, int, int, int]]) -> None:
# #         if self.read_only:
# #             raise RuntimeError("ImageRegistry is read-only; cannot bulk_upsert")

# #         cur = self.conn.cursor()
# #         cur.executemany(
# #             """
# #             INSERT INTO images (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
# #             VALUES (?, ?, ?, ?, ?, ?, ?)
# #             ON CONFLICT(root, relpath) DO UPDATE SET
# #               abspath=excluded.abspath,
# #               image_id=excluded.image_id,
# #               file_size=excluded.file_size,
# #               mtime_ns=excluded.mtime_ns,
# #               indexed_at_ns=excluded.indexed_at_ns
# #             """,
# #             rows,
# #         )
# #         self.conn.commit()

# #     # ---- Queries ----

# #     def path_to_id(self, root: Path, relpath: str) -> Optional[str]:
# #         row = self.get_record(root, relpath)
# #         return row["image_id"] if row else None

# #     def id_to_paths(self, image_id: str) -> List[sqlite3.Row]:
# #         cur = self.conn.cursor()
# #         return cur.execute(
# #             "SELECT * FROM images WHERE image_id=? ORDER BY root, relpath",
# #             (image_id,),
# #         ).fetchall()

# #     def all_rows(self) -> List[sqlite3.Row]:
# #         cur = self.conn.cursor()
# #         return cur.execute("SELECT * FROM images ORDER BY root, relpath").fetchall()


# # # image_id_pipeline/resolver.py
# # from __future__ import annotations

# # import json
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Any, Dict, List, Optional, Tuple

# # from .registry import normalize_relpath


# # @dataclass(frozen=True)
# # class ResolvedItem:
# #     relpath: str          # relative to image_root
# #     component: str        # provenance label


# # def _read_lines_txt(path: Path) -> List[str]:
# #     out: List[str] = []
# #     for raw in path.read_text(encoding="utf-8").splitlines():
# #         line = raw.strip()
# #         if not line or line.startswith("#"):
# #             continue
# #         out.append(line.replace("\\", "/").lstrip("/"))
# #     return out


# # def _load_json_or_yaml(path: Path) -> Dict[str, Any]:
# #     suffix = path.suffix.lower()
# #     if suffix == ".json":
# #         return json.loads(path.read_text(encoding="utf-8"))

# #     if suffix in (".yml", ".yaml"):
# #         try:
# #             import yaml  # type: ignore
# #         except ImportError as e:
# #             raise ImportError(
# #                 "YAML config provided but PyYAML is not installed. "
# #                 "Install pyyaml or use JSON."
# #             ) from e
# #         return yaml.safe_load(path.read_text(encoding="utf-8"))

# #     raise ValueError(f"Unsupported config format: {path} (use .txt, .json, .yaml/.yml)")


# # def resolve_paths(
# #     image_root: str,
# #     spec_path: str,
# #     lists_root: Optional[str] = None,
# # ) -> Tuple[List[ResolvedItem], Dict[str, Any]]:
# #     """
# #     Returns:
# #       - resolved items (relpaths relative to image_root)
# #       - metadata dict (mode, dataset_name, dedupe, etc.)

# #     spec_path can be:
# #       - .txt: a plain list of relpaths under image_root
# #       - .json/.yaml: dataset config referencing per-root named lists
# #     """
# #     image_root_p = Path(image_root).resolve()
# #     spec_p = Path(spec_path).resolve()

# #     # ---- Mode 1: plain text list of relpaths under image_root ----
# #     if spec_p.suffix.lower() == ".txt":
# #         relpaths = [normalize_relpath(rp) for rp in _read_lines_txt(spec_p)]
# #         items = [ResolvedItem(relpath=rp, component="default") for rp in relpaths]
# #         meta = {"mode": "text_list", "dataset_name": spec_p.stem, "dedupe": "by_path"}
# #         return items, meta

# #     # ---- Mode 2: dataset config referencing per-root named lists ----
# #     cfg = _load_json_or_yaml(spec_p)
# #     dataset_name = cfg.get("dataset_name", spec_p.stem)
# #     dedupe = cfg.get("dedupe", "by_path")
# #     components = cfg.get("components")
# #     if not isinstance(components, list) or not components:
# #         raise ValueError("dataset config must have non-empty 'components' list")

# #     if lists_root is None:
# #         raise ValueError("lists_root is required for dataset config mode")

# #     lists_root_p = Path(lists_root).resolve()
# #     resolved: List[ResolvedItem] = []

# #     for comp in components:
# #         if not isinstance(comp, dict):
# #             raise ValueError("each component must be a dict")

# #         comp_name = str(comp.get("name") or "component")
# #         root_subdir = comp.get("root_subdir")
# #         list_name = comp.get("list")

# #         if not root_subdir or not list_name:
# #             raise ValueError("each component must include 'root_subdir' and 'list'")

# #         list_file = lists_root_p / str(root_subdir) / f"{str(list_name)}.txt"
# #         if not list_file.exists():
# #             raise FileNotFoundError(f"List file not found: {list_file}")

# #         inner_relpaths = _read_lines_txt(list_file)
# #         for inner_rp in inner_relpaths:
# #             combined = normalize_relpath(f"{root_subdir}/{inner_rp}")
# #             resolved.append(ResolvedItem(relpath=combined, component=comp_name))

# #     meta = {
# #         "mode": "dataset_config",
# #         "dataset_name": dataset_name,
# #         "dedupe": dedupe,
# #         "spec_path": str(spec_p),
# #         "lists_root": str(lists_root_p),
# #         "image_root": str(image_root_p),
# #     }
# #     return resolved, meta
