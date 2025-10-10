import yaml
from typing import Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

def load_cfg() -> Dict[str, Any]:
    with open("sn_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_embeddings(cfg):
    backend = cfg.get("embedding_backend", "hf").lower()
    if backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[INFO] Query: Using HuggingFace embeddings: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        model = cfg.get("embedding_model", "nomic-embed-text")
        print(f"[INFO] Query: Using Ollama embeddings: {model} at {cfg.get('ollama_base_url')}")
        return OllamaEmbeddings(
            model=model,
            base_url=cfg.get("ollama_base_url", "http://localhost:11434")
        )

def load_vectorstore(cfg: Dict[str, Any]) -> Chroma:
    embeddings = make_embeddings(cfg)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=cfg["persist_directory"],
        collection_name="nomad_sn"
    )

def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title") or d.metadata.get("source", f"doc{i}")
        url = d.metadata.get("url", "")

        # Read as strings (fallback to join if a list slips through)
        subjects = d.metadata.get("subjects", "")
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)

        keywords = d.metadata.get("keywords", "")
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        rt  = d.metadata.get("resource_type", "")
        lvl = d.metadata.get("level", "")

        header = f"### {title}" + (f" ({url})" if url else "")
        lines = [header]
        if subjects: lines.append(f"**Subjects**: {subjects}")
        if keywords: lines.append(f"**Keywords**: {keywords}")
        if rt:       lines.append(f"**Type**: {rt}")
        if lvl:      lines.append(f"**Level**: {lvl}")
        lines.append(d.page_content)
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)



def get_sources(docs) -> List[str]:
    uniq = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source")
        url = d.metadata.get("url", "")
        s = f"{title} - {url}" if url else title
        if s not in uniq:
            uniq.append(s)
    return uniq

def build_chain():
    cfg = load_cfg()
    vs = load_vectorstore(cfg)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg.get("k", 5), "fetch_k": 20, "lambda_mult": 0.5}
    )
    system_msg = (
        "You are NOMAD Training Assistant. Answer concisely using only the provided context. "
        "If unsure or out of scope, say you don't know. After the answer, provide a short 'Sources:' list."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    model = ChatOllama(
        model=cfg.get("chat_model", "gpt-oss:20b"),
        base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
        temperature=0.2
    )

    # Use retriever.invoke (no deprecation)
    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"question": x["question"], "docs": retriever.invoke(x["question"])})
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "context": format_docs(x["docs"]),
            "sources": get_sources(x["docs"])
        })
        | RunnableLambda(lambda x: {
            "answer": model.invoke(prompt.format_messages(question=x["question"], context=x["context"])).content,
            "sources": x["sources"]
        })
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    out = chain.invoke("What is the NOMAD onboarding workshop?")
    print(out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print("-", s)
