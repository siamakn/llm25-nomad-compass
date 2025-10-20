from fastapi import APIRouter, Request
from .sn_bootstrap import ensure_services

router = APIRouter(prefix="/status")

@router.get("/ping")
async def ping():
    return {"status": "ok"}

@router.get("/ready")
async def ready(request: Request):
    services = await ensure_services(request.app)  # <-- await
    vs = services.get("vector_store")
    try:
        return {"ready": bool(vs and getattr(vs, "is_ready", lambda: False)())}
    except Exception:
        return {"ready": False}

@router.get("/debug")
async def debug(request: Request):
    services = getattr(request.app.state, "services", None)
    if not services:
        return {"has_services": False}
    vs = services.get("vector_store")
    docs, vecs = ([], None)
    try:
        docs, vecs = vs.get_index() if vs else ([], None)
    except Exception:
        pass
    return {
        "has_services": True,
        "docs_len": len(docs),
        "vecs_len": (len(vecs) if isinstance(vecs, list) else None),
        "ready": bool(vs and getattr(vs, "is_ready", lambda: False)()),
    }
