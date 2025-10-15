from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, APIRouter
from nomad.config import config

from ..sn_config import Settings
from ..services.sn_rdf_loader import RDFLoader
from ..services.sn_vector_store import SimpleVectorStore
from ..services.sn_chatbot_service import ChatbotService
from ..services.sn_corpus_sig import compute_signature, save_signature, load_signature
from ..services.sn_index_store import FileIndexStore
from .sn_status_api import router as status_router
from .sn_chatbot_api import router as chatbot_router

router = APIRouter()
router.include_router(status_router)
router.include_router(chatbot_router)

# --- Standalone-friendly root_path handling ---
try:
    _ep = config.get_plugin_entry_point('nomad_compass.apis:myapi')
    _root_path = f'{config.services.api_base_path}/{_ep.prefix}'
except Exception:
    # Not running under NOMAD -> use empty root_path
    _root_path = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
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
    yield
    await vs.aclose()

app = FastAPI(
    root_path=_root_path,
    title="NOMAD Compass Plugin",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}
