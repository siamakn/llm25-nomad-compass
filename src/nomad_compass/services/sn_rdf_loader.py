from __future__ import annotations
import json, asyncio
from pathlib import Path
from typing import Dict, List

class RDFLoader:
    def __init__(self, rdf_dir: Path):
        self.rdf_dir = Path(rdf_dir)

    async def load_all_jsonld(self) -> List[Dict]:
        files = sorted(self.rdf_dir.glob("*.jsonld"))
        tasks = [asyncio.to_thread(self._read_one, fp) for fp in files]
        return [d for d in await asyncio.gather(*tasks) if d]

    def _read_one(self, fp: Path) -> Dict | None:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            text = self._extract_text(data)
            return {
                "id": fp.name,
                "text": text,
                "meta": {
                    "path": str(fp),
                    "title": data.get("name") or data.get("headline") or fp.stem,
                    "dateModified": data.get("dateModified"),
                },
            }
        except Exception:
            return None

    def _extract_text(self, data: Dict) -> str:
        fields = []
        for k in ("name", "headline", "description", "text", "abstract"):
            v = data.get(k)
            if isinstance(v, str):
                fields.append(v)
        for k in ("about", "keywords"):
            v = data.get(k)
            if isinstance(v, list):
                fields.extend([str(x) for x in v if isinstance(x, (str, int, float))])
        return "\n".join(fields)
