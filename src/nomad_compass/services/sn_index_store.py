# sn_index_store.py
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Tuple, List, Any

class FileIndexStore:
    def __init__(self, cache_dir: Path, index_filename: str = "vector_index.pkl") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / index_filename

    def exists(self) -> bool:
        return self.index_path.exists()

    def save(self, docs: List[dict], vecs: List[Any] | None) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump({"docs": docs, "vecs": vecs}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> Tuple[List[dict], List[Any] | None]:
        with open(self.index_path, "rb") as f:
            obj = pickle.load(f)
        return obj.get("docs", []), obj.get("vecs", None)
