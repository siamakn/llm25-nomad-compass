from __future__ import annotations
import os
from pathlib import Path

class Settings:
    """
    Lightweight settings without extra deps.
    Override via environment variables:
      NOMAD_COMPASS_TOP_K, NOMAD_COMPASS_CACHE_DIR, etc.
    """
    def __init__(self):
        self.package_dir: Path = Path(__file__).resolve().parent
        self.data_dir: Path = self.package_dir / "data"
        self.rdf_dir: Path = Path(os.getenv("NOMAD_COMPASS_RDF_DIR", str(self.data_dir / "rdf_files")))
        self.cache_dir: Path = Path(os.getenv("NOMAD_COMPASS_CACHE_DIR", str(self.package_dir / "cache")))
        self.index_filename: str = os.getenv("NOMAD_COMPASS_INDEX_FILENAME", "vector_index.pkl")
        self.signature_filename: str = os.getenv("NOMAD_COMPASS_SIGNATURE_FILENAME", "corpus_signature.json")
        self.top_k: int = int(os.getenv("NOMAD_COMPASS_TOP_K", "5"))
        self.min_score: float = float(os.getenv("NOMAD_COMPASS_MIN_SCORE", "0.0"))
