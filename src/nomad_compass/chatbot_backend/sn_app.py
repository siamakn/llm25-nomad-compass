from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .sn_rag_chain import build_chain
import uvicorn

# Initialize FastAPI
app = FastAPI(title="NOMAD RAG Chatbot")

# Enable CORS for local or frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chain variable (initialized on startup)
chain = None


@app.on_event("startup")
def init_chain():
    """Build RAG chain when FastAPI starts."""
    global chain
    print("[INFO] Initializing RAG chain...")
    chain = build_chain()
    print("[INFO] RAG chain initialized successfully.")


# Request model
class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Receive a user message and return chatbot reply."""
    try:
        if chain is None:
            return {"error": "RAG chain not initialized yet."}

        result = chain.invoke(req.message)
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        return {"reply": answer, "sources": sources}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "NOMAD RAG Chatbot is running."}


# Always run on port 8001
if __name__ == "__main__":
    uvicorn.run("sn_app:app", host="0.0.0.0", port=8001, reload=True)
