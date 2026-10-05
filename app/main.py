"""
FastAPI application entrypoint for Support Copilot.

Exposes a small HTTP API that takes a raw customer support ticket, retrieves
relevant knowledge-base articles and past tickets (see app.retrieval), and
asks an LLM to draft a ready-to-review reply plus lightweight triage
metadata (see app.drafting). Also serves the static single-page frontend
from app/static.
"""
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

# Built once at startup: loads the knowledge-base / past-ticket corpus and
# fits the TF-IDF vectorizer used for retrieval on every /api/draft request.
index = RetrievalIndex()


class DraftRequest(BaseModel):
    """Request body for POST /api/draft.

    Attributes:
        ticket_text: The raw customer message to draft a reply for.
        tone: Optional reply register. One of "default", "formal", or
            "casual"; unrecognized values fall back to "default".
    """

    ticket_text: str
    tone: str = "default"


class SourceOut(BaseModel):
    """A single retrieved source (knowledge-base article or past ticket)
    returned alongside a drafted reply, so an agent can see what the draft
    was grounded in.

    Attributes:
        id: Document ID, e.g. "kb-002" or "t-1015".
        source_type: Either "knowledge_base" or "past_ticket".
        title: Human-readable title for display in the UI.
        score: Cosine-similarity relevance score in [0, 1]; higher means
            more relevant.
    """

    id: str
    source_type: str
    title: str
    score: float


class DraftResponse(BaseModel):
    """Response body for POST /api/draft.

    Attributes:
        draft_reply: The model-drafted reply text for an agent to review.
        sources_used: IDs of the retrieved documents the model says it
            actually drew on (a subset of retrieved_sources).
        category: One of billing | sync | account | sharing | mobile | other.
        urgency: One of low | medium | high.
        sentiment: One of positive | neutral | frustrated | angry.
        notes_for_agent: Anything the agent should double-check, or context
            the knowledge base didn't cover.
        retrieved_sources: All documents retrieval surfaced for this ticket,
            regardless of whether the model cited them.
    """

    draft_reply: str
    sources_used: list[str]
    category: str
    urgency: str
    sentiment: str
    notes_for_agent: str
    retrieved_sources: list[SourceOut]


@app.get("/")
def root():
    """Serve the single-page frontend.

    Returns:
        FileResponse: app/static/index.html.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Basic liveness/readiness check.

    Returns:
        dict: Service status plus a quick sanity count of how many
        knowledge-base articles and past tickets were loaded into the
        retrieval index at startup.
    """
    return {"status": "ok", "kb_articles": len(index.kb_articles), "past_tickets": len(index.past_tickets)}


@app.post("/api/draft", response_model=DraftResponse)
def api_draft(req: DraftRequest):
    """Draft a reply to a customer support ticket.

    Retrieves the most relevant knowledge-base articles and past tickets for
    the given ticket text, then asks the LLM (see app.drafting.draft_reply)
    to produce a grounded draft reply plus triage metadata.

    Args:
        req: The incoming ticket text and desired reply tone.

    Returns:
        DraftResponse: The drafted reply, triage metadata, and the sources
        retrieval surfaced.

    Raises:
        HTTPException: 400 if ticket_text is empty/whitespace-only; 500 if
            the drafting call fails (e.g. missing API key or model error).
    """
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
