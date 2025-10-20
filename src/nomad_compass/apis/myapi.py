import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from ..sn_config import Settings
from ..services.sn_chatbot_service import ChatbotService
from .sn_bootstrap import init_services
from .sn_status_api import router as status_router
from .sn_chatbot_api import router as chatbot_router

router = APIRouter()
router.include_router(status_router)
router.include_router(chatbot_router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings

    base = await init_services(settings)
    chatbot = ChatbotService(vector_store=base["vector_store"], settings=settings)
    app.state.services = {**base, "chatbot": chatbot}
    try:
        yield
    finally:
        await base["vector_store"].aclose()

app = FastAPI(
    title="NOMAD Compass Plugin",
    version="0.1.0",
    lifespan=lifespan,
)

origins = os.getenv('NOMAD_COMPASS_CORS_ORIGINS', 'http://localhost:3000').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}
