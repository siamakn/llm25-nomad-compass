from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Any, List

class FileIndexStore:
    """Tiny on-disk persistence for the vector index (docs + vecs)."""
    def __init__(self, cache_dir: Path, index_filename: str):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / index_filename

    def save(self, docs: List[Dict[str, Any]], vecs: List[Dict[str, float]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            pickle.dump({"docs": docs, "vecs": vecs}, f)

    def load(self) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        with self.index_path.open("rb") as f:
            data = pickle.load(f)
        return data["docs"], data["vecs"]

    def exists(self) -> bool:
        return self.index_path.exists()
