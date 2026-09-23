"""Semantic search over an item catalog: retrieve-then-rerank with Jev.

The pattern behind Aayan-DEV/aayans-yc-indexor: cheap vector retrieval narrows the
whole catalog to a shortlist, then ONE parallel Jev call scores every finalist and
returns a calibrated probability per item. Interactive search (~1 s, ~$0.002) that
keyword matching can't express ("dev-tools startup with a blue logo").

Jev never generates text here — it only judges relevance. The UI does the rest.
"""

from typesafe_sdk import Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

MAX_FINALISTS = 320  # vector recall set; Jev scores all of them in one request


def semantic_search(query: str, catalog: list[dict], embed_fn, top_k: int = 10) -> list[dict]:
    """catalog: items with at least 'id', 'title', 'description'. embed_fn: your
    local embedding function (bge-small, MobileCLIP, whatever fits the domain)."""
    query_vec = embed_fn(query)

    # Stage 1 — cheap retrieval: cosine similarity over the whole catalog.
    ranked = sorted(
        catalog,
        key=lambda item: cosine(query_vec, embed_fn(item["description"])),
        reverse=True,
    )[:MAX_FINALISTS]

    # Stage 2 — Jev re-scores every finalist in one parallel call.
    # State carries the query ONCE; each question is one candidate item.
    response = client.system_one(
        state={"query": query},
        questions={
            item["id"]: Score(
                f"How well does this item match the query: {item['title']} — "
                f"{item['description'][:300]}",
                ["Irrelevant", "Weak match", "Good match", "Excellent match"],
            )
            for item in ranked
        },
    )

    # Calibrated probabilities out, sorted list back. No LLM text anywhere.
    scored = [
        {
            "id": item["id"],
            "title": item["title"],
            "probability": round(response.answers[item["id"]].score / 3.0, 3),
            "confidence": round(response.answers[item["id"]].confidence, 3),
        }
        for item in ranked
    ]
    scored.sort(key=lambda r: r["probability"], reverse=True)
    return scored[:top_k]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0
