"""Lead scoring: composite scoring pattern.

Break a fuzzy judgment ("is this a good lead?") into independent dimensions, score
each atomically, and combine with weights YOU control. Re-weighting becomes a code
change — not a re-prompt — so you can A/B it.
"""

from typesafe_sdk import Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

# Weights live in code, not in the prompt. Change them without touching the model.
WEIGHTS = {"budget": 0.30, "need": 0.30, "timing": 0.20, "fit": 0.20}


def score_lead(lead: dict) -> dict:
    response = client.system_one(
        state={
            "title": lead["title"],
            "company": lead["company"],
            "company_description": lead.get("company_description", ""),
            "message": lead.get("message", ""),
        },
        questions={
            "budget": Score(
                "Likelihood the prospect has budget authority",
                ["No signal", "Individual contributor", "Influencer", "Decision maker"]),
            "need": Score(
                "How clearly the prospect expresses a need we serve",
                ["No need stated", "Vague interest", "Concrete pain", "Actively evaluating"]),
            "timing": Score(
                "How soon they are likely to buy",
                ["No timeline", "Someday", "This quarter", "Now"]),
            "fit": Score(
                "Fit with our ideal customer profile",
                ["Poor fit", "Partial fit", "Good fit", "Ideal fit"]),
        },
    )
    a = response.answers
    composite = sum(WEIGHTS[k] * (a[k].score / 3.0) for k in WEIGHTS)  # normalize 0..3 -> 0..1

    # Gate the expensive action (sales call) on the composite, not on vibes.
    action = "route_to_sales" if composite >= 0.65 else "nurture"
    if min(a[k].confidence for k in WEIGHTS) < 0.5:
        action = "enrich_first"  # low confidence anywhere -> get more data, don't guess
    return {"score": round(composite, 3), "action": action}
