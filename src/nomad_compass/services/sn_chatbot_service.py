from ..sn_config import Settings
from ..models.sn_chatbot_models import AskRequest, AskResponse, Snippet
import httpx

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

        # 🔹 Try querying HU LLM server for a better answer
        answer_text = None
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/generate",
                    json={"model": self.settings.chat_model, "prompt": req.question},
                )
                if response.status_code == 200:
                    result = response.json()
                    answer_text = result.get("response") or result.get("message")
        except Exception:
            pass  # fallback below if LLM unreachable

        if not answer_text:
            # fallback heuristic
            lines = []
            for d in docs[:3]:
                title = d["meta"].get("title", d["id"])
                if d["text"]:
                    lines.append(f"- {title}: {d['text'].splitlines()[0][:240]}")
            answer_text = "Here are relevant training resources:\n" + "\n".join(lines)

        return AskResponse(
            answer=answer_text,
            sources=[Snippet(id=d["id"], text=d["text"], score=d["score"], meta=d["meta"]) for d in docs],
        )
