"""
Tests for app.retrieval.RetrievalIndex.

Covers that the index loads the full knowledge-base/past-ticket corpus on
construction, and that search() returns relevant, correctly bounded results
for representative queries.
"""
from app.retrieval import RetrievalIndex


def test_index_loads_all_documents():
    """The index should load every KB article and past ticket, combined
    into a single searchable documents list of the expected total size."""
    index = RetrievalIndex()
    assert len(index.kb_articles) == 10
    assert len(index.past_tickets) == 8
    assert len(index.documents) == 18


def test_search_finds_relevant_kb_article_for_sync_issue():
    """A sync-related query should surface the sync-troubleshooting KB
    article (kb-002) among the top results."""
    index = RetrievalIndex()
    results = index.search("my files are stuck syncing and won't upload", top_k=3)
    ids = [r["id"] for r in results]
    assert "kb-002" in ids


def test_search_finds_relevant_article_for_refund_question():
    """A refund-related query should surface the refund-policy KB article
    (kb-004) among the top results."""
    index = RetrievalIndex()
    results = index.search("can I get my money back, I want a refund", top_k=3)
    ids = [r["id"] for r in results]
    assert "kb-004" in ids


def test_search_returns_empty_for_blank_query():
    """A blank/whitespace-only query should short-circuit to an empty
    result list instead of erroring or returning arbitrary matches."""
    index = RetrievalIndex()
    assert index.search("   ", top_k=3) == []


def test_search_respects_top_k():
    """search() should never return more than top_k results, even when
    more documents would otherwise match."""
    index = RetrievalIndex()
    results = index.search("account sync billing storage sharing", top_k=2)
    assert len(results) <= 2
