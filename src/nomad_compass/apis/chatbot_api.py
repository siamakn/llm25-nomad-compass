from fastapi import APIRouter
from nomad_compass.chatbot_backend.sn_app import chat, ChatRequest

router = APIRouter()

@router.post("/chat")
def chatbot_reply(req: ChatRequest):
    return chat(req)

@router.get("/")
def chatbot_status():
    return {"status": "Chatbot endpoint is active within NOMAD."}
