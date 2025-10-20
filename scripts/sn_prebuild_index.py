import asyncio
from pathlib import Path
from nomad_compass.sn_config import Settings
from nomad_compass.services.sn_rdf_loader import RDFLoader
from nomad_compass.services.sn_vector_store import SimpleVectorStore
from nomad_compass.services.sn_index_store import FileIndexStore
from nomad_compass.services.sn_corpus_sig import compute_signature, save_signature

async def main():
    s = Settings()
    cache_dir = Path(s.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"rdf_dir:   {s.rdf_dir}")
    print(f"cache_dir: {cache_dir}")

    loader = RDFLoader(s.rdf_dir)
    docs = await loader.load_all_jsonld()
    print(f"docs: {len(docs)}")

    vs = SimpleVectorStore()
    await vs.build_index(docs)
    print(f"is_ready after build: {getattr(vs, 'is_ready', lambda: False)()}")

    store = FileIndexStore(cache_dir, s.index_filename)
    store.save(*vs.get_index())
    sig_path = cache_dir / s.signature_filename
    save_signature(compute_signature(s.rdf_dir), sig_path)
    print(f"saved: {store.index_path}")
    print(f"saved: {sig_path}")

if __name__ == "__main__":
    asyncio.run(main())
