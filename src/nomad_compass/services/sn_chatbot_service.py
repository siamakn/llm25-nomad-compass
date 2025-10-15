from __future__ import annotations
from ..sn_config import Settings
from ..models.sn_chatbot_models import AskRequest, AskResponse, Snippet

class ChatbotService:
    def __init__(self, vector_store, settings: Settings):
        self.vs = vector_store
        self.settings = settings

    async def search(self, query: str, top_k: int | None = None):
        k = top_k or self.settings.top_k
        hits = await self.vs.search(query, top_k=k)
        out = []
        for idx, score in hits:
            doc = self.vs._docs[idx]
            out.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": float(score),
                "meta": doc["meta"],
            })
        return out

    async def answer(self, req: AskRequest) -> AskResponse:
        top_k = req.top_k or self.settings.top_k
        docs = await self.search(req.question, top_k=top_k)
        if not docs:
            return AskResponse(answer="No relevant training resource found.", sources=[])
        lines = []
        for d in docs[:3]:
            title = d["meta"].get("title", d["id"])
            if d["text"]:
                lines.append(f"- {title}: {d['text'].splitlines()[0][:240]}")
        answer = "Here are relevant training resources:\n" + "\n".join(lines)
        return AskResponse(
            answer=answer,
            sources=[Snippet(id=d["id"], text=d["text"], score=d["score"], meta=d["meta"]) for d in docs],
        )
