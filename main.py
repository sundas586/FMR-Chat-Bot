"""FastAPI service for the FMR RAG learning project."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingest import ingest
from rag import answer

app = FastAPI(title="FMR RAG Chatbot", version="0.1.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the FastAPI service is running."""
    return {"status": "ok"}


@app.post("/ingest")
def run_ingest(reset: bool = False) -> dict[str, int]:
    """Trigger PDF ingestion and return the number of indexed chunks."""
    return {"chunks_indexed": ingest(reset=reset)}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Handle a chatbot question and convert processing errors to HTTP errors."""
    try:
        return answer(request.question, count=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
