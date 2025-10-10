import yaml
from langchain_chroma import Chroma

def load_cfg():
    with open("sn_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_embeddings(cfg):
    backend = cfg.get("embedding_backend", "hf").lower()
    if backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.get("embedding_model", "nomic-embed-text"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434")
        )

if __name__ == "__main__":
    cfg = load_cfg()
    vs = Chroma(
        embedding_function=make_embeddings(cfg),
        persist_directory=cfg["persist_directory"],
        collection_name="nomad_sn"
    )
    # Count vectors
    try:
        count = vs._collection.count()  # internal but handy
        print(f"[DEBUG] Collection 'nomad_sn' vectors: {count}")
    except Exception as e:
        print("[DEBUG] Could not read collection count:", e)

    # Peek top-3 docs for a generic query
    print("\n[DEBUG] Try a probe query: 'NOMAD onboarding workshop'")
    results = vs.similarity_search_with_score("NOMAD onboarding workshop", k=3)
    for i, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title") or doc.metadata.get("source")
        url = doc.metadata.get("url", "")
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"  {i}. title={title!r}  score={score:.4f}  url={url}")
        print(f"     preview: {preview} ...\n")
