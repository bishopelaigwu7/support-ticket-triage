"""
Drafting layer: turns a raw customer ticket + retrieved context into a
grounded, ready-to-review reply, plus lightweight triage metadata
(urgency / sentiment / category) in a single model call.

Design choices worth calling out in an interview:
- The model is asked to cite which knowledge-base IDs it drew on, and to
  say so explicitly if the retrieved context doesn't cover the question.
  That's the single biggest lever against confident-sounding wrong
  answers in a support context -- silent hallucination is the failure
  mode that actually damages customer trust.
- Structured output (tags/urgency/sentiment) is requested as JSON in the
  same call as the draft, rather than as a second API call, to keep
  latency and cost down for what is meant to be an interactive tool.
- Nothing here auto-sends to the customer. This produces a draft for a
  human agent to review and edit -- matching how every real product in
  this space (Zendesk Copilot, Intercom Copilot, Help Scout AI Drafts,
  etc.) is designed. Keeping a human in the loop is a feature, not a
  limitation.
"""
import json
import os
from dataclasses import dataclass, field

from anthropic import Anthropic

SYSTEM_PROMPT = """You are a support-response drafting assistant for CloudSync, \
a cloud file sync and backup product. You help human support agents by \
drafting a reply to a customer ticket. You do NOT send anything to the \
customer yourself -- an agent will review and edit your draft before sending.

Rules:
- Match the tone of the example past replies you're given: warm, direct, \
concise, no corporate filler, no over-apologizing.
- Only state policy facts (refund windows, storage limits, retention periods, \
etc.) that are supported by the provided knowledge-base context. If the \
context doesn't cover something the customer asked, say so plainly in the \
"notes_for_agent" field instead of guessing.
- Keep replies short: 3-6 sentences unless the issue genuinely requires more.
- Always respond with ONLY a single JSON object matching this shape, no \
other text:
{
  "draft_reply": "the drafted reply text",
  "sources_used": ["kb-001", "t-1002"],
  "category": "billing | sync | account | sharing | mobile | other",
  "urgency": "low | medium | high",
  "sentiment": "positive | neutral | frustrated | angry",
  "notes_for_agent": "anything the agent should double check or that the \
retrieved context didn't cover"
}
"""


@dataclass
class DraftResult:
    """Structured result of a drafting call.

    Mirrors the JSON shape the model is instructed to return (see
    SYSTEM_PROMPT above), with safe defaults so a partially-parsed or
    malformed model response still produces a usable object instead of
    raising.

    Attributes:
        draft_reply: The drafted reply text for an agent to review.
        sources_used: IDs of retrieved documents the model says it drew on.
        category: One of billing | sync | account | sharing | mobile | other.
        urgency: One of low | medium | high.
        sentiment: One of positive | neutral | frustrated | angry.
        notes_for_agent: Anything the agent should double-check, or context
            the knowledge base didn't cover.
    """

    draft_reply: str
    sources_used: list = field(default_factory=list)
    category: str = "other"
    urgency: str = "medium"
    sentiment: str = "neutral"
    notes_for_agent: str = ""


def _strip_markdown_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` (or plain ``` ... ```) fence.

    Some models wrap JSON output in a markdown code fence even when told
    not to. This strips it so the remaining text can be passed straight to
    json.loads without raising on the fence markers.

    Args:
        text: Raw text returned by the model.

    Returns:
        str: The text with a leading/trailing markdown fence removed, if
        one was present. Text with no fence is returned unchanged
        (aside from surrounding whitespace being trimmed).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _build_context_block(retrieved_docs: list) -> str:
    """Render retrieved knowledge-base articles / past tickets as plain text.

    Formats each retrieved document into a labeled block the model can cite
    by ID in its response (see SYSTEM_PROMPT's "sources_used" instruction).

    Args:
        retrieved_docs: Documents returned by
            app.retrieval.RetrievalIndex.search(), each a dict with at
            least "source_type" and "raw" keys.

    Returns:
        str: A newline-separated block of formatted documents, ready to
        drop into the user prompt. Returns a plain "no results" sentence
        if retrieved_docs is empty, so the model is told explicitly rather
        than silently prompted with nothing.
    """
    if not retrieved_docs:
        return "No relevant knowledge-base articles or past tickets were found."

    lines = []
    for doc in retrieved_docs:
        if doc["source_type"] == "knowledge_base":
            article = doc["raw"]
            lines.append(
                f"[{article['id']}] KB ARTICLE: {article['title']}\n{article['body']}"
            )
        else:
            ticket = doc["raw"]
            lines.append(
                f"[{ticket['id']}] PAST TICKET\n"
                f"Customer said: {ticket['customer_message']}\n"
                f"Agent replied: {ticket['agent_reply']}"
            )
    return "\n\n".join(lines)


def draft_reply(ticket_text: str, retrieved_docs: list, tone: str = "default") -> DraftResult:
    """Call Claude to draft a reply to a customer ticket, grounded in retrieved_docs.

    Builds a single prompt containing the ticket text, the formatted
    retrieved context, and an optional tone instruction, then asks the
    model for the JSON-shaped response described in SYSTEM_PROMPT.

    Args:
        ticket_text: The raw customer message to draft a reply for.
        retrieved_docs: Knowledge-base articles / past tickets from
            app.retrieval.RetrievalIndex.search() to ground the reply in.
        tone: Reply register to request. One of "default", "formal", or
            "casual"; any other value is treated as "default" (no extra
            instruction added).

    Returns:
        DraftResult: The drafted reply and triage metadata. If the model's
        response isn't valid JSON, returns a DraftResult with the raw model
        text as draft_reply and a note flagging the parse failure, rather
        than raising, so callers always get a usable object back.

    Raises:
        RuntimeError: If the ANTHROPIC_API_KEY environment variable is not
            set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    client = Anthropic(api_key=api_key)

    tone_instruction = {
        "default": "",
        "formal": "Use a more formal, professional register than the examples.",
        "casual": "Use a more casual, friendly register than the examples.",
    }.get(tone, "")

    context_block = _build_context_block(retrieved_docs)

    user_prompt = f"""CUSTOMER TICKET:
{ticket_text}

RETRIEVED CONTEXT:
{context_block}

{tone_instruction}

Draft a reply to this customer now, following the JSON output format from \
your instructions."""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    try:
        parsed = json.loads(_strip_markdown_fence(raw_text))
    except json.JSONDecodeError:
        # Model didn't return clean JSON -- fail soft with the raw text
        # visible, rather than crashing the request.
        return DraftResult(
            draft_reply=raw_text,
            notes_for_agent="Model response was not valid JSON; showing raw output.",
        )

    return DraftResult(
        draft_reply=parsed.get("draft_reply", ""),
        sources_used=parsed.get("sources_used", []),
        category=parsed.get("category", "other"),
        urgency=parsed.get("urgency", "medium"),
        sentiment=parsed.get("sentiment", "neutral"),
        notes_for_agent=parsed.get("notes_for_agent", ""),
    )
