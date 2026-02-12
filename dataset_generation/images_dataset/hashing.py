# image_id_pipeline/hashing.py
from __future__ import annotations


import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HashSpec:
    alg: str = "sha256"
    chunk_size: int = 1024 * 1024  # 1MB


class ImageIdComputer:
    """
    Unified image_id logic used by:
      - preprocess indexer
      - pytorch dataset
      - any other pipeline components
    """

    def __init__(self, spec: HashSpec | None = None):
        self.spec = spec or HashSpec()

    def compute(self, path: Path) -> str:
        if self.spec.alg != "sha256":
            raise ValueError(f"Unsupported hash alg: {self.spec.alg}")

        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(self.spec.chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
