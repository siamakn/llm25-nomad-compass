from __future__ import annotations
from typing import Any, List, Tuple, Sequence

class SimpleVectorStore:
    def __init__(self) -> None:
        self._docs: List[dict] = []
        self._vecs: List[Any] | None = None

    def set_index(self, docs: Sequence[dict] | None, vecs: Sequence[Any] | None) -> None:
        self._docs = list(docs or [])
        self._vecs = list(vecs or [])

    def get_index(self) -> Tuple[List[dict], List[Any] | None]:
        return self._docs, self._vecs

    async def build_index(self, docs: Sequence[dict]) -> None:
        # trivial vector per doc; replace with real embeddings later
        vecs = [0] * len(docs)
        self.set_index(docs, vecs)

    def is_ready(self) -> bool:
        return self._vecs is not None and len(self._docs) > 0 and len(self._vecs) == len(self._docs)

    async def search(self, query: str, top_k: int = 5):
        """
        Trivial search: return first k docs with dummy score 1.0
        Replace with real similarity search later.
        """
        k = min(top_k, len(self._docs))
        return [(i, 1.0) for i in range(k)]

    async def aclose(self) -> None:
        pass
