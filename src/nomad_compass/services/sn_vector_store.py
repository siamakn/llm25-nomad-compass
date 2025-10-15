from __future__ import annotations
import re, math, asyncio
from collections import Counter
from typing import Dict, List, Tuple

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

def _tf_vector(text: str) -> Dict[str, float]:
    toks = _tokenize(text)
    c = Counter(toks)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b: return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    den = (sum(v*v for v in a.values()) ** 0.5) * (sum(v*v for v in b.values()) ** 0.5)
    return (num / den) if den else 0.0

class SimpleVectorStore:
    def __init__(self):
        self._docs: List[Dict] = []
        self._vecs: List[Dict[str, float]] = []

    async def build_index(self, docs: List[Dict]):
        self._docs = docs
        tasks = [asyncio.to_thread(_tf_vector, d.get("text", "")) for d in docs]
        self._vecs = await asyncio.gather(*tasks)

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        qv = await asyncio.to_thread(_tf_vector, query)
        tasks = [asyncio.to_thread(_cosine, qv, dv) for dv in self._vecs]
        scores = await asyncio.gather(*tasks)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # For persistence
    def set_index(self, docs: List[Dict], vecs: List[Dict[str, float]]):
        self._docs, self._vecs = docs, vecs

    def get_index(self) -> tuple[List[Dict], List[Dict[str, float]]]:
        return self._docs, self._vecs

    async def aclose(self):
        return
