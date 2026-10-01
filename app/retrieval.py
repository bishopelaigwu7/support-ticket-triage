"""
Lightweight retrieval layer.

Grounding an LLM's draft reply in *real* prior answers and knowledge-base
articles is what separates a useful support copilot from a generic
chatbot -- it keeps the tone consistent with how your team actually talks
to customers, and it keeps facts (refund windows, storage limits, etc.)
accurate instead of hallucinated.

This uses plain TF-IDF + cosine similarity rather than an embeddings API.
That's a deliberate choice for this project: it needs zero extra API keys
or vector DB infrastructure to run, it's fast, and for a knowledge base of
this size it retrieves just as reliably as embeddings would. Swap in a
proper embedding model (see README "Extending this project") once the
corpus grows past a few hundred documents or you need semantic matches
across paraphrases TF-IDF can't catch.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


class RetrievalIndex:
    """Search over knowledge-base articles and past resolved tickets.

    Loads the JSON corpus once at construction time and fits a single
    TF-IDF vectorizer over both knowledge-base articles and past tickets
    combined, so a query can surface whichever document type is most
    relevant.

    Attributes:
        kb_articles: Raw knowledge-base article records loaded from
            data/knowledge_base.json.
        past_tickets: Raw past-ticket records loaded from
            data/past_tickets.json.
        documents: Normalized, searchable view of kb_articles + past_tickets
            (one dict per document, with a combined "text" field used for
            TF-IDF).
        vectorizer: Fitted sklearn TfidfVectorizer for this corpus.
    """

    def __init__(self):
        """Load the knowledge base and past tickets and fit the TF-IDF index.

        Raises:
            FileNotFoundError: If knowledge_base.json or past_tickets.json
                is missing from app/data.
            json.JSONDecodeError: If either data file contains invalid JSON.
        """
        self.kb_articles = self._load(DATA_DIR / "knowledge_base.json")
        self.past_tickets = self._load(DATA_DIR / "past_tickets.json")

        self.documents = []
        for article in self.kb_articles:
            self.documents.append({
                "source_type": "knowledge_base",
                "id": article["id"],
                "title": article["title"],
                "text": f"{article['title']}. {article['body']}",
                "raw": article,
            })
        for ticket in self.past_tickets:
            self.documents.append({
                "source_type": "past_ticket",
                "id": ticket["id"],
                "title": f"Similar past ticket ({ticket['id']})",
                "text": f"{ticket['customer_message']} {ticket['agent_reply']}",
                "raw": ticket,
            })

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform(
            doc["text"] for doc in self.documents
        )

    @staticmethod
    def _load(path: Path):
        """Read and parse a JSON data file.

        Args:
            path: Path to a JSON file containing a list of records.

        Returns:
            The parsed JSON content (a list of dicts for this project's
            data files).

        Raises:
            FileNotFoundError: If path does not exist.
            json.JSONDecodeError: If path's contents are not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search(self, query: str, top_k: int = 3):
        """Return the top_k most relevant documents for a query string.

        Scores every document in the corpus against the query using
        cosine similarity over TF-IDF vectors, then returns the highest-
        scoring matches with a positive score, most relevant first.

        Args:
            query: Free-text query, typically the customer's ticket text.
            top_k: Maximum number of documents to return.

        Returns:
            list[dict]: Up to top_k documents from self.documents, each
            with an added "score" key (float, rounded to 4 decimal places).
            Documents with a similarity score of 0 or less are excluded.
            Returns an empty list if query is blank/whitespace-only or the
            corpus is empty.
        """
        if not query.strip() or not self.documents:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            zip(self.documents, scores), key=lambda pair: pair[1], reverse=True
        )

        results = []
        for doc, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append({**doc, "score": round(float(score), 4)})
        return results
