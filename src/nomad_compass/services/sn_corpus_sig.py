from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    mtime_ns: int

def _gather_files(root: Path, pattern: str = "*.jsonld") -> List[FileInfo]:
    files = []
    root = Path(root)
    for fp in sorted(root.glob(pattern)):
        try:
            st = fp.stat()
            files.append(FileInfo(path=str(fp.name), size=st.st_size, mtime_ns=st.st_mtime_ns))
        except FileNotFoundError:
            continue
    return files

def compute_signature(rdf_dir: Path) -> Dict:
    files = _gather_files(rdf_dir)
    payload = "|".join(f"{f.path}:{f.size}:{f.mtime_ns}" for f in files)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "algo": "sha256(path:size:mtime_ns)",
        "digest": digest,
        "count": len(files),
    }

def save_signature(sig: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sig, indent=2), encoding="utf-8")

def load_signature(path: Path) -> Dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
