import json, os, glob, yaml, re
from typing import Any, Dict, List, DefaultDict, Tuple
from collections import defaultdict
from tqdm import tqdm

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ---------- CONFIG & VOCAB ----------

def get_config_path(filename: str = "sn_config.yaml") -> str:
    """Find the configuration file automatically relative to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Could not find {filename} at {path}")
    return path


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vocab(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load vocabulary file from known paths or warn if missing."""
    candidates = []
    if "vocabulary_path" in cfg:
        candidates.append(cfg["vocabulary_path"])
    candidates.extend([
        "vocabulary.json",
        os.path.join(os.path.dirname(__file__), "vocabulary.json"),
        os.path.join(os.path.dirname(__file__), "data", "vocabulary.json"),
    ])
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                v = json.load(f)
            print(f"[INFO] Loaded vocabulary from {p}")
            return v
    print("[WARN] No vocabulary.json found; proceeding without normalization.")
    return {
        "subjects": [], "keywords": [], "educational_levels": [],
        "instructional_methods": [], "learning_resource_types": [], "formats": []
    }


# ---------- JSON-LD EXTRACTION ----------

WANTED = {
    "name", "title", "headline", "description", "abstract", "summary", "text",
    "subject", "keywords", "inlanguage", "audience",
    "creator", "author", "contributor", "publisher", "datepublished",
    "url", "identifier", "typicalagerange", "educationallevel", "license",
    "ispartof", "type", "@type", "learningresourcetype", "instructionalmethod",
    "sameas", "isformatof", "references", "hasformat", "haspart", "isreferencedby"
}
CURIE_OR_IRI = re.compile(r"^(schema:|rdf:|rdfs:|https?://)", re.I)


def norm_key(k: Any) -> str:
    if not isinstance(k, str):
        return ""
    k = k.strip()
    if k.startswith("@"):
        return k.lower()
    if ":" in k:
        k = k.split(":", 1)[1]
    return k.lower()


def extract_text_value(v: Any) -> str:
    if isinstance(v, dict):
        if "@value" in v:
            return str(v["@value"]).strip()
        if "value" in v:
            return str(v["value"]).strip()
        if "@id" in v and all(k.startswith("@") for k in v.keys()):
            return ""
        parts = []
        for _, vv in v.items():
            if isinstance(vv, (str, int, float, bool)):
                s = str(vv).strip()
                if s and not CURIE_OR_IRI.match(s):
                    parts.append(s)
        return ", ".join(parts).strip()
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        return "" if CURIE_OR_IRI.match(s) else s
    if isinstance(v, list):
        items = [extract_text_value(it) for it in v]
        items = [it for it in items if it]
        return ", ".join(dict.fromkeys(items))
    return ""


def collect_fields(obj: Any, bag: DefaultDict[str, List[str]]):
    if isinstance(obj, dict):
        if "@context" in obj and len(obj) == 1:
            return
        for k, v in obj.items():
            nk = norm_key(k)
            if nk in WANTED:
                text = extract_text_value(v)
                if text:
                    if nk in {"subject", "keywords"}:
                        parts = [p.strip() for p in text.split(",") if p.strip()]
                        bag[nk].extend(parts)
                    else:
                        bag[nk].append(text)
            collect_fields(v, bag)
    elif isinstance(obj, list):
        for item in obj:
            collect_fields(item, bag)


# ---------- MAIN PIPELINE ----------

def make_embeddings(cfg):
    backend = cfg.get("embedding_backend", "hf").lower()
    if backend == "hf":
        print(f"[INFO] Using HuggingFace embeddings: {cfg.get('embedding_model')}")
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        print(f"[INFO] Using Ollama embeddings: {cfg.get('embedding_model')} at {cfg.get('ollama_base_url')}")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.get("embedding_model", "nomic-embed-text"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434")
        )


def main():
    print("[INFO] Starting NOMAD Compass index builder...")

    # Load config and vocabulary
    cfg_path = get_config_path()
    cfg = load_cfg(cfg_path)
    vocab = load_vocab(cfg)

    # --- Resolve JSON-LD directory relative to this script ---
    here = os.path.dirname(os.path.abspath(__file__))
    jsonld_dir = cfg.get("jsonld_dir", "data/rdf_files")

    # If it's already absolute, keep it. Otherwise, make it relative to this file.
    if not os.path.isabs(jsonld_dir):
        jsonld_dir = os.path.join(here, jsonld_dir)
    jsonld_dir = os.path.normpath(jsonld_dir)

    if not os.path.isdir(jsonld_dir):
        raise FileNotFoundError(f"[ERROR] JSON-LD directory not found: {jsonld_dir}")
    else:
        print(f"[INFO] Using JSON-LD directory: {jsonld_dir}")

    # --- Load JSON-LD documents ---
    docs = []
    for p in sorted(glob.glob(os.path.join(jsonld_dir, "*.json*"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            bag = defaultdict(list)
            collect_fields(data, bag)
            text = "\n".join([f"{k}: {', '.join(v)}" for k, v in bag.items()])
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    print(f"[INFO] Loaded {len(docs)} JSON-LD files.")

    # --- Split documents into chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 1000),
        chunk_overlap=cfg.get("chunk_overlap", 200),
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    # --- Build embeddings and persist database ---
    embeddings = make_embeddings(cfg)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nomad_sn",
        persist_directory=cfg["persist_directory"]
    )

    print(f"[SUCCESS] Chroma DB ready at {cfg['persist_directory']} (auto-persisted).")


if __name__ == "__main__":
    main()
