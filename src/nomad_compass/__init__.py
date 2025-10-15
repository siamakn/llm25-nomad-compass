from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .sn_config import Settings
from .services.sn_chatbot_service import ChatbotService
from .services.sn_rdf_loader import RDFLoader
from .services.sn_vector_store import SimpleVectorStore
from .apis import sn_status_api, sn_chatbot_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings
    settings = Settings()
    app.state.settings = settings

    # Initialize services (async load on startup)
    loader = RDFLoader(settings.rdf_dir)
    docs = await loader.load_all_jsonld()  # [{id, text, meta}, ...]

    vector_store = SimpleVectorStore()
    await vector_store.build_index(docs)   # async-friendly build

    chatbot = ChatbotService(vector_store=vector_store, settings=settings)
    app.state.services = {
        "chatbot": chatbot,
        "vector_store": vector_store,
        "loader": loader,
    }

    yield

    # Teardown if needed (e.g., close pools)
    await vector_store.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NOMAD Compass Plugin",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(sn_status_api.router)
    app.include_router(sn_chatbot_api.router)
    return app


# NOMAD and ASGI servers will import `app`
app = create_app()
