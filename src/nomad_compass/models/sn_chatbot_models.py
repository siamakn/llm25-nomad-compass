from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50)

class Snippet(BaseModel):
    id: str
    text: str
    score: float
    meta: Dict[str, Any] = {}

class AskResponse(BaseModel):
    answer: str
    sources: List[Snippet] = []
