"""Support ticket triage: one parallel Jev call routes, scores, and flags.

Pattern: speculative fan-out — ask every question up front (a tenth question costs
tokens but almost no time), then let code decide what was relevant.
"""

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")


def triage(ticket: dict) -> dict:
    response = client.system_one(
        state={
            "subject": ticket["subject"],
            "body": ticket["body"][:2000],  # trim: accuracy falls on bloated state
            "customer_tier": ticket.get("customer_tier", "standard"),
        },
        questions={
            "queue": Choice(
                instructions="Which team should handle this ticket",
                criteria={
                    "billing": "Payment, subscription, invoice, or refund issues",
                    "technical": "Bugs, errors, or integration problems",
                    "sales": "Pricing, plans, or account questions",
                    "spam": "Unsolicited or abusive content",
                    "other": "Anything else",
                },
            ),
            "urgency": Score(
                instructions="How urgent is this ticket",
                criteria=["Routine", "Time-sensitive", "Blocking the customer"],
            ),
            "frustration": Score(
                instructions="How frustrated the customer appears",
                criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry"],
            ),
            "needs_human": Noul(
                instructions="This ticket needs a human rather than an automated reply"
            ),
        },
    )
    a = response.answers

    # One threshold per action, scaled to what being wrong costs.
    if a["needs_human"].noul > 0.6 or a["queue"].confidence < 0.7:
        return {"action": "human_queue", "queue": a["queue"].choice,
                "priority": "high" if a["urgency"].score > 1.5 else "normal"}
    return {"action": "auto_reply", "queue": a["queue"].choice,
            "template": f"reply_{a['queue'].choice}"}
