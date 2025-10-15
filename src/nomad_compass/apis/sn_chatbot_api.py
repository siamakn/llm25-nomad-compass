from __future__ import annotations
from fastapi import APIRouter, Depends, Request, HTTPException
from ..models.sn_chatbot_models import AskRequest, AskResponse

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

def _get_chatbot(request: Request):
    try:
        return request.app.state.services["chatbot"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service not initialized") from exc

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, chatbot = Depends(_get_chatbot)) -> AskResponse:
    return await chatbot.answer(req)

@router.get("/search")
async def search(q: str, chatbot = Depends(_get_chatbot)):
    return await chatbot.search(q, top_k=10)
