from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter(prefix="/status", tags=["Status"])

@router.get("/ping")
async def ping():
    return {"status": "ok"}

@router.get("/ready")
async def ready(request: Request):
    ok = hasattr(request.app.state, "services") and "chatbot" in request.app.state.services
    return {"ready": bool(ok)}
