from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI

from ..sn_config import Settings
from ..services.sn_rdf_loader import RDFLoader
from ..services.sn_vector_store import SimpleVectorStore
from ..services.sn_chatbot_service import ChatbotService
from ..services.sn_corpus_sig import compute_signature, save_signature, load_signature
from ..services.sn_index_store import FileIndexStore


async def ensure_services(app: FastAPI) -> Dict[str, Any]:
    """
    Lazily initialize and cache services on first use.
    Safe to call on every request; it returns app.state.services.
    """
    if getattr(app.state, "services", None):
        return app.state.services

    settings = getattr(app.state, "settings", None) or Settings()
    app.state.settings = settings

    cache_dir: Path = settings.cache_dir
    index_store = FileIndexStore(cache_dir, settings.index_filename)
    sig_path = cache_dir / settings.signature_filename

    current_sig = compute_signature(settings.rdf_dir)
    cached_sig = load_signature(sig_path)

    vs = SimpleVectorStore()

    if cached_sig and cached_sig.get("digest") == current_sig.get("digest") and index_store.exists():
        try:
            docs, vecs = index_store.load()
            vs.set_index(docs, vecs)
        except Exception:
            loader = RDFLoader(settings.rdf_dir)
            docs = await loader.load_all_jsonld()
            await vs.build_index(docs)
            index_store.save(*vs.get_index())
            save_signature(current_sig, sig_path)
    else:
        loader = RDFLoader(settings.rdf_dir)
        docs = await loader.load_all_jsonld()
        await vs.build_index(docs)
        index_store.save(*vs.get_index())
        save_signature(current_sig, sig_path)

    chatbot = ChatbotService(vector_store=vs, settings=settings)
    app.state.services = {"vector_store": vs, "chatbot": chatbot, "index_store": index_store}
    return app.state.services


# 🔹 Add this block below — required by the NOMAD app loader
async def init_services(app: FastAPI) -> Dict[str, Any]:
    """
    Explicit service initializer (used by NOMAD app startup).
    Simply wraps ensure_services for compatibility.
    """
    return await ensure_services(app)
