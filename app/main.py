import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.drafting import draft_reply
from app.retrieval import RetrievalIndex

load_dotenv()

app = FastAPI(title="Support Copilot", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

index = RetrievalIndex()


class DraftRequest(BaseModel):
    ticket_text: str
    tone: str = "default"


class SourceOut(BaseModel):
    id: str
    source_type: str
    title: str
    score: float


class DraftResponse(BaseModel):
    draft_reply: str
    sources_used: list[str]
    category: str
    urgency: str
    sentiment: str
    notes_for_agent: str
    retrieved_sources: list[SourceOut]


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "kb_articles": len(index.kb_articles), "past_tickets": len(index.past_tickets)}


@app.post("/api/draft", response_model=DraftResponse)
def api_draft(req: DraftRequest):
    if not req.ticket_text.strip():
        raise HTTPException(status_code=400, detail="ticket_text is required")

    top_k = int(os.environ.get("RETRIEVAL_TOP_K", 3))
    retrieved = index.search(req.ticket_text, top_k=top_k)

    try:
        result = draft_reply(req.ticket_text, retrieved, tone=req.tone)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DraftResponse(
        draft_reply=result.draft_reply,
        sources_used=result.sources_used,
        category=result.category,
        urgency=result.urgency,
        sentiment=result.sentiment,
        notes_for_agent=result.notes_for_agent,
        retrieved_sources=[
            SourceOut(id=d["id"], source_type=d["source_type"], title=d["title"], score=d["score"])
            for d in retrieved
        ],
    )
