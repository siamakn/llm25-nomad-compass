from fastapi import FastAPI, APIRouter
from nomad.config import config
from nomad_compass.apis.chatbot_api import router as chatbot_router

router = APIRouter()
router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])

myapi_entry_point = config.get_plugin_entry_point('nomad_compass.apis:myapi')

app = FastAPI(
    root_path=f'{config.services.api_base_path}/{myapi_entry_point.prefix}'
)

@app.get('/')
async def root():
    return {"message": "Hello World"}