# examples/code_search.py
# The siftr pattern (Bentlybro/siftr, Sep 2026): ask Jev "is this relevant?"
# about hundreds of code chunks at once, in two passes — rank broadly, then
# zoom into the best files for exact lines.
#
# pip install typesafe-sdk   (reads TYPESAFE_API_KEY from the environment)
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")  # pin the version; jev-latest moves


def rank_chunks(issue: str, chunks: list[str]) -> list[tuple[str, float]]:
    """Score every chunk's relevance to the issue in one parallel Jev request.

    In production, batch hundreds of chunks per request — Jev answers all of
    them in a single pass, so a 4,000-file repo becomes a few dozen requests.
    """
    questions = {f"c{i}": Noul("This code chunk is relevant to fixing the issue")
                 for i in range(len(chunks))}
    answers = client.system_one(
        state={"issue": issue, "chunks": chunks}, questions=questions).answers
    scored = [(chunks[i], answers[f"c{i}"].noul) for i in range(len(chunks))]
    return sorted(scored, key=lambda t: t[1], reverse=True)


def search_repo(issue: str, files: list[dict]) -> list[str]:
    """Two-pass search: rank files by names/definitions, then zoom into the best 30.

    Pass 2 (not shown) feeds the exact line ranges of the top-30 files back
    through the same question to point at the lines to change.
    """
    overview = [f"{f['path']}: {f['definitions']}" for f in files]
    top = [chunk for chunk, _ in rank_chunks(issue, overview)[:30]]
    return top


if __name__ == "__main__":
    files = [
        {"path": "src/billing.py", "definitions": "calculate_refund, apply_discount"},
        {"path": "src/auth.py", "definitions": "login, refresh_token"},
    ]
    print(search_repo("refund amount is doubled on retry", files))
