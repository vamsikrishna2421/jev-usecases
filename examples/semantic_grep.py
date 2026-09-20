"""Semantic grep: filter lines by meaning, not by pattern.

Each line becomes one yes/no (Noul) question to Jev. Decisions arrive in ~200 ms
at about a thousandth of a cent per line, so judging runs concurrently in the
pipeline — fast and cheap enough to sit in `tail -f`.

Pattern: retrieve, then judge — one typed decision per atomic unit, threshold in
code. Inspired by keltokhy/jgrep (measured: 994 HN titles in 4.6 s for $0.012).
"""

from concurrent.futures import ThreadPoolExecutor

from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

MAX_WORKERS = 32  # jgrep's default: 32 calls in flight


def line_matches(line: str, description: str, threshold: float = 0.5) -> bool:
    response = client.system_one(
        state={"line": line[:8000]},  # ordinary records are truncated, not bloated
        questions={
            "hit": Noul(
                instructions=f"This line matches the description: {description}"
            )
        },
    )
    return response.answers["hit"].noul >= threshold


def semantic_grep(lines: list[str], description: str, threshold: float = 0.5) -> list[str]:
    """Return the lines whose meaning matches `description`, in input order."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        verdicts = pool.map(lambda ln: line_matches(ln, description, threshold), lines)
    return [ln for ln, ok in zip(lines, verdicts) if ok]


if __name__ == "__main__":
    sample = [
        "user 12: this is the third time checkout has failed, I am done with this app",
        "user 77: WHY does it log me out every five minutes??",
        "deploys: rolling out v2.14 to staging",
    ]
    for hit in semantic_grep(sample, "a user is getting frustrated"):
        print(hit)
