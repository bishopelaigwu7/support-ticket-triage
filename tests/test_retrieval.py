from app.retrieval import RetrievalIndex


def test_index_loads_all_documents():
    index = RetrievalIndex()
    assert len(index.kb_articles) == 10
    assert len(index.past_tickets) == 8
    assert len(index.documents) == 18


def test_search_finds_relevant_kb_article_for_sync_issue():
    index = RetrievalIndex()
    results = index.search("my files are stuck syncing and won't upload", top_k=3)
    ids = [r["id"] for r in results]
    assert "kb-002" in ids


def test_search_finds_relevant_article_for_refund_question():
    index = RetrievalIndex()
    results = index.search("can I get my money back, I want a refund", top_k=3)
    ids = [r["id"] for r in results]
    assert "kb-004" in ids


def test_search_returns_empty_for_blank_query():
    index = RetrievalIndex()
    assert index.search("   ", top_k=3) == []


def test_search_respects_top_k():
    index = RetrievalIndex()
    results = index.search("account sync billing storage sharing", top_k=2)
    assert len(results) <= 2
