# image_id_pipeline/registry.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RegistryConfig:
    """
    Points to a directory that will contain the registry SQLite DB.

    In your hydra config you used:
      registry_root: /work/data/output/registry/
    so we store:
      <registry_root>/registry.sqlite3
    """
    registry_root: Path
    registry_db_name: str = "registry.sqlite3"
    strict: bool = True
    export_human_readable: bool = True



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_root(root: Path | str) -> str:
    return str(Path(root).resolve())


def _normalize_relpath(relpath: str) -> str:
    # normalize separators, strip any leading slashes
    rel = relpath.replace("\\", "/").lstrip("/")

    # Basic traversal protection
    if rel == ".." or rel.startswith("../") or "/../" in rel:
        raise ValueError(f"Refusing relpath with parent traversal: {relpath}")

    if rel == "":
        raise ValueError("relpath cannot be empty")
    return rel


class ImageRegistry:
    """
    SQLite-backed image registry.

    Schema:
      images(
        root TEXT NOT NULL,
        relpath TEXT NOT NULL,
        abspath TEXT NOT NULL,
        image_id TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        indexed_at_ns INTEGER NOT NULL,
        PRIMARY KEY(root, relpath)
      )
    """

    DEFAULT_DB_NAME = "registry.sqlite3"

    def __init__(self, config: RegistryConfig, *, read_only: bool = False):
        self.config = config
        self.registry_root = Path(config.registry_root)
        _ensure_dir(self.registry_root)

        self.db_path = self.registry_root / self.DEFAULT_DB_NAME
        self._conn = self._connect(self.db_path, read_only=read_only)
        self._init_schema()

    @classmethod
    def from_db_path(cls, db_path: str | Path, *, read_only: bool = False) -> "ImageRegistry":
        """
        Used by RegistryBackedImageDataset workers where you only have the db path.
        """
        db_path = Path(db_path)
        cfg = RegistryConfig(registry_root=db_path.parent)
        obj = cls.__new__(cls)  # bypass __init__
        obj.config = cfg
        obj.registry_root = cfg.registry_root
        obj.db_path = db_path
        obj._conn = obj._connect(db_path, read_only=read_only)
        obj._init_schema()  # safe even for RO; only reads schema if present
        return obj

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -----------------------
    # Connection / Schema
    # -----------------------

    def _connect(self, db_path: Path, *, read_only: bool) -> sqlite3.Connection:
        # For read-only, use sqlite URI mode to enforce it.
        if read_only:
            uri = f"file:{db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)

        conn.row_factory = sqlite3.Row

        # Reasonable defaults
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

        return conn

    def _init_schema(self) -> None:
        # If the DB is read-only and does not exist, sqlite will error on connect.
        # If it exists, CREATE TABLE IF NOT EXISTS is fine.
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    root TEXT NOT NULL,
                    relpath TEXT NOT NULL,
                    abspath TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    indexed_at_ns INTEGER NOT NULL,
                    PRIMARY KEY (root, relpath)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_images_image_id ON images(image_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_images_root_relpath ON images(root, relpath)")

    # -----------------------
    # Core ops
    # -----------------------

    def get_record(self, root: Path | str, relpath: str) -> Optional[Dict[str, Any]]:
        root_s = _normalize_root(root)
        rel = _normalize_relpath(relpath)
        cur = self._conn.execute(
            """
            SELECT root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns
            FROM images
            WHERE root = ? AND relpath = ?
            """,
            (root_s, rel),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def path_to_id(self, root: Path | str, relpath: str) -> Optional[str]:
        row = self.get_record(root, relpath)
        return None if row is None else str(row["image_id"])

    def id_to_paths(self, image_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT root, relpath, abspath
            FROM images
            WHERE image_id = ?
            ORDER BY root ASC, relpath ASC
            """,
            (image_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def upsert_record(
        self,
        *,
        root: Path | str,
        relpath: str,
        abspath: Path | str,
        image_id: str,
        file_size: int,
        mtime_ns: int,
        indexed_at_ns: int,
    ) -> None:
        root_s = _normalize_root(root)
        rel = _normalize_relpath(relpath)
        abs_s = str(Path(abspath).resolve())

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO images(root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(root, relpath) DO UPDATE SET
                    abspath=excluded.abspath,
                    image_id=excluded.image_id,
                    file_size=excluded.file_size,
                    mtime_ns=excluded.mtime_ns,
                    indexed_at_ns=excluded.indexed_at_ns
                """,
                (root_s, rel, abs_s, image_id, int(file_size), int(mtime_ns), int(indexed_at_ns)),
            )

    def bulk_upsert(self, rows: Sequence[Tuple[str, str, str, str, int, int, int]]) -> None:
        """
        rows are:
          (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)

        Matches preprocess.py exactly:
          to_upsert.append((str(root), it.relpath, str(abspath), image_id, file_size, mtime_ns, now_ns))
        """
        if not rows:
            return

        normalized: List[Tuple[str, str, str, str, int, int, int]] = []
        for root_s, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns in rows:
            rr = _normalize_root(root_s)
            rel = _normalize_relpath(relpath)
            abs_s = str(Path(abspath).resolve())
            normalized.append((rr, rel, abs_s, image_id, int(file_size), int(mtime_ns), int(indexed_at_ns)))

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO images(root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(root, relpath) DO UPDATE SET
                    abspath=excluded.abspath,
                    image_id=excluded.image_id,
                    file_size=excluded.file_size,
                    mtime_ns=excluded.mtime_ns,
                    indexed_at_ns=excluded.indexed_at_ns
                """,
                normalized,
            )

    def all_rows(self) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns
            FROM images
            ORDER BY root ASC, relpath ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]



# from __future__ import annotations

# import sqlite3
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, List, Tuple


# SCHEMA_SQL = """
# PRAGMA journal_mode=WAL;
# PRAGMA synchronous=NORMAL;

# CREATE TABLE IF NOT EXISTS images (
#     root TEXT NOT NULL,
#     relpath TEXT NOT NULL,
#     abspath TEXT NOT NULL,
#     image_id TEXT NOT NULL,
#     file_size INTEGER NOT NULL,
#     mtime_ns INTEGER NOT NULL,
#     indexed_at_ns INTEGER NOT NULL,

#     PRIMARY KEY (root, relpath)
# );

# CREATE INDEX IF NOT EXISTS idx_images_image_id ON images(image_id);
# """


# def normalize_relpath(relpath: str) -> str:
#     rel = relpath.replace("\\", "/").lstrip("/")
#     # refuse parent traversal
#     if rel.startswith("../") or "/../" in rel or rel == "..":
#         raise ValueError(f"Refusing relpath with parent traversal: {relpath}")
#     return rel


# @dataclass(frozen=True)
# class RegistryConfig:
#     registry_root: Path  # directory
#     db_name: str = "image_registry.sqlite"


# class ImageRegistry:
#     """
#     Canonical mapping store.
#     - root + relpath -> image_id
#     - image_id -> list of (root, relpath, abspath, stats)

#     Worker-safe design:
#     - Do NOT share an open sqlite connection across processes.
#     - We open lazily (per-process) on first use.
#     """

#     def __init__(self, cfg: RegistryConfig, *, read_only: bool = False):
#         self.cfg = cfg
#         self.read_only = read_only

#         self.cfg.registry_root.mkdir(parents=True, exist_ok=True)
#         self.db_path: Path = self.cfg.registry_root / self.cfg.db_name

#         # connection is opened lazily in this process
#         self._conn: Optional[sqlite3.Connection] = None

#         # Ensure schema exists (only if not read-only)
#         if not self.read_only:
#             self._ensure_open()
#             self._init_db()

#     @classmethod
#     def from_db_path(cls, db_path: str | Path, *, read_only: bool = True) -> "ImageRegistry":
#         """
#         Convenience constructor for datasets/workers that only know the db path.
#         """
#         db_path = Path(db_path)
#         cfg = RegistryConfig(registry_root=db_path.parent, db_name=db_path.name)
#         return cls(cfg, read_only=read_only)

#     @property
#     def conn(self) -> sqlite3.Connection:
#         self._ensure_open()
#         assert self._conn is not None
#         return self._conn

#     def _ensure_open(self) -> None:
#         if self._conn is not None:
#             return

#         # For read-only usage, prefer opening in read-only mode.
#         # NOTE: even with read-only, SQLite still uses locks; WAL is fine for concurrent readers.
#         if self.read_only:
#             uri = f"file:{self.db_path.as_posix()}?mode=ro"
#             self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
#         else:
#             self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

#         self._conn.row_factory = sqlite3.Row

#     def close(self) -> None:
#         if self._conn is not None:
#             self._conn.close()
#             self._conn = None

#     def _init_db(self) -> None:
#         cur = self.conn.cursor()
#         cur.executescript(SCHEMA_SQL)
#         self.conn.commit()

#     # ---- Upserts / indexing ----

#     def get_record(self, root: Path, relpath: str) -> Optional[sqlite3.Row]:
#         root_s = str(root.resolve())
#         rel = normalize_relpath(relpath)
#         cur = self.conn.cursor()
#         return cur.execute(
#             "SELECT * FROM images WHERE root=? AND relpath=?",
#             (root_s, rel),
#         ).fetchone()

#     def upsert_record(
#         self,
#         root: Path,
#         relpath: str,
#         abspath: Path,
#         image_id: str,
#         file_size: int,
#         mtime_ns: int,
#         indexed_at_ns: int,
#     ) -> None:
#         if self.read_only:
#             raise RuntimeError("ImageRegistry is read-only; cannot upsert_record")

#         root_s = str(root.resolve())
#         rel = normalize_relpath(relpath)
#         cur = self.conn.cursor()
#         cur.execute(
#             """
#             INSERT INTO images (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             ON CONFLICT(root, relpath) DO UPDATE SET
#               abspath=excluded.abspath,
#               image_id=excluded.image_id,
#               file_size=excluded.file_size,
#               mtime_ns=excluded.mtime_ns,
#               indexed_at_ns=excluded.indexed_at_ns
#             """,
#             (root_s, rel, str(abspath), image_id, int(file_size), int(mtime_ns), int(indexed_at_ns)),
#         )
#         self.conn.commit()

#     def bulk_upsert(self, rows: List[Tuple[str, str, str, str, int, int, int]]) -> None:
#         if self.read_only:
#             raise RuntimeError("ImageRegistry is read-only; cannot bulk_upsert")

#         cur = self.conn.cursor()
#         cur.executemany(
#             """
#             INSERT INTO images (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#             ON CONFLICT(root, relpath) DO UPDATE SET
#               abspath=excluded.abspath,
#               image_id=excluded.image_id,
#               file_size=excluded.file_size,
#               mtime_ns=excluded.mtime_ns,
#               indexed_at_ns=excluded.indexed_at_ns
#             """,
#             rows,
#         )
#         self.conn.commit()

#     # ---- Queries ----

#     def path_to_id(self, root: Path, relpath: str) -> Optional[str]:
#         row = self.get_record(root, relpath)
#         return row["image_id"] if row else None

#     def id_to_paths(self, image_id: str) -> List[sqlite3.Row]:
#         cur = self.conn.cursor()
#         return cur.execute(
#             "SELECT * FROM images WHERE image_id=? ORDER BY root, relpath",
#             (image_id,),
#         ).fetchall()

#     def all_rows(self) -> List[sqlite3.Row]:
#         cur = self.conn.cursor()
#         return cur.execute("SELECT * FROM images ORDER BY root, relpath").fetchall()


# # # image_id_pipeline/registry.py
# # from __future__ import annotations

# # import sqlite3
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Iterable, Optional, List, Tuple, Dict, Any


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
# #     registry_root: str
# #     export_human_readable: bool = True
# #     strict: bool = True
# #     # registry_root: Path  # directory
# #     db_name: str = "image_registry.sqlite"


# # class ImageRegistry:
# #     """
# #     Canonical mapping store.
# #     - root + relpath -> image_id
# #     - image_id -> list of (root, relpath, abspath, stats)
# #     """

# #     def __init__(self, cfg: RegistryConfig):
# #         self.cfg = cfg
# #         self.cfg.registry_root.mkdir(parents=True, exist_ok=True)
# #         self.db_path = self.cfg.registry_root / self.cfg.db_name
# #         self.conn = sqlite3.connect(str(self.db_path))
# #         self.conn.row_factory = sqlite3.Row
# #         self._init_db()

# #     def close(self) -> None:
# #         self.conn.close()

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
# #         """
# #         rows: list of (root, relpath, abspath, image_id, file_size, mtime_ns, indexed_at_ns)
# #         """
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
