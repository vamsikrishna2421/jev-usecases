"""Entity-resolution match adjudication with Jev.

Production shape this mirrors:
  Snowflake / Snowpark deterministic + fuzzy stage  ->  ~75% of pairs resolved
  Jev typed-decision adjudication                  ->  ambiguous pairs, ~$0.0004/case
  Confidence-gated routing                        ->  auto-act / human review / LLM rationale

Keep the LLM (or a human) for the low-confidence slice where you need explainable
reasoning or an audit trail. Validate Jev's calibration on YOUR labeled match pairs
before trusting a threshold.

API shape follows the published typesafe-sdk guides (Sep 2026); verify against
current docs before building on it.
"""

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

# Pin the version: jev-latest moves, and answers can change under you.
client = TypeSafeClient(model="jev-1.13.0")


def adjudicate_pair(record_a: dict, record_b: dict, deterministic_score: float):
    """Ask Jev one parallel, typed question-set about a candidate pair.

    record_a / record_b: the two candidate records (dicts of identity attributes).
    deterministic_score: fuzzy-match score from the deterministic stage (0..1).
    """
    response = client.system_one(
        state={
            "record_a": record_a,
            "record_b": record_b,
            "deterministic_score": deterministic_score,
        },
        questions={
            # The core adjudication: one bounded decision, asked explicitly.
            "verdict": Choice(
                instructions="Do these two records describe the same real-world entity?",
                criteria={
                    "match": (
                        "Same person, company, or location; differences are only "
                        "formatting, spelling variants, abbreviations, or field ordering"
                    ),
                    "no_match": "Different real-world entities",
                    "needs_review": "Cannot decide from the given fields",
                },
            ),
            # Independent signal, scored atomically (composite-scoring pattern).
            "match_strength": Score(
                instructions="How strong is the evidence that these records match?",
                criteria=[
                    "No meaningful overlap",
                    "Weak overlap on one or two attributes",
                    "Strong overlap on most attributes",
                    "Conclusive overlap across attributes",
                ],
            ),
            # Individual match signals as yes/no probabilities — cheap, parallel.
            "name_is_variant": Noul(
                instructions="The names differ only by spelling, abbreviation, or ordering"
            ),
            "address_agrees": Noul(
                instructions="The addresses refer to the same location"
            ),
            "contact_agrees": Noul(
                instructions="Phone or email identifiers agree"
            ),
        },
    )
    a = response.answers
    return a["verdict"], a["match_strength"], a


def route_pair(record_a: dict, record_b: dict, deterministic_score: float) -> str:
    """Confidence-gated routing: auto-act above threshold, escalate below."""
    verdict, strength, all_answers = adjudicate_pair(record_a, record_b, deterministic_score)

    # High-confidence decisions: let code act. (Tune 0.9 on your labeled pairs.)
    if verdict.choice == "match" and verdict.confidence >= 0.9:
        return "auto_merge"
    if verdict.choice == "no_match" and verdict.confidence >= 0.9:
        return "auto_reject"

    # Everything else — low confidence, needs_review, conflicting signals —
    # goes where a rationale exists: human review, or the LLM for explainable reasoning.
    return "human_review"


if __name__ == "__main__":
    a = {"name": "Acme Corp", "address": "71 Maywood St, Worcester MA 01603",
         "phone": "+1 (774) 525-7098"}
    b = {"name": "ACME Corporation", "address": "71 Maywood Street, Worcester, MA",
         "phone": "+17745257098"}
    print(route_pair(a, b, deterministic_score=0.82))
