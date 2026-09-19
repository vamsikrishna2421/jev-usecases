"""Agent-loop cascade: Jev decides which requests deserve a frontier model.

Pattern: Jev is not a replacement for the LLM — it's the thing that decides which
requests deserve one. One branch never touches a model; two load different
specialists; one escalates to a human.
"""

from typesafe_sdk import Choice, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")


def handle(message: str) -> str:
    r = client.system_one(
        state=message,
        questions={
            "intent": Choice(
                "Primary intent of this message",
                {"order_status": "Asking about an existing order",
                 "product_question": "Asking about a product",
                 "return_exchange": "Wants to return or exchange",
                 "complaint": "Unhappy, wants resolution",
                 "other": "Anything else"}),
            "complexity": Score(
                "How complex this is to resolve",
                ["Simple lookup or standard procedure",
                 "Requires judgment or multiple steps",
                 "Unusual edge case, escalation needed"]),
        },
    )
    intent, cx = r.answers["intent"], r.answers["complexity"]

    # Floor: genuinely unsure -> human, no matter what.
    if intent.confidence < 0.5:
        return route_to_human(message)

    if intent.choice == "order_status":
        return lookup_order(message)                      # pure code, no model at all
    if intent.choice == "product_question":
        return handle_with_llm(message, "PRODUCT_SPECIALIST")
    if intent.choice == "return_exchange":
        return handle_with_llm(message, "RETURNS_SPECIALIST")
    if intent.choice == "complaint":
        if cx.score > 1 or cx.confidence < 0.5:
            return route_to_human(message)
        return handle_with_llm(message, "COMPLAINT_RESOLUTION")
    return route_to_human(message)


# ---- stubs: wire these to your own code/LLM calls ----
def route_to_human(message): return f"[human] {message}"
def lookup_order(message): return f"[lookup] {message}"
def handle_with_llm(message, specialist): return f"[{specialist}] {message}"
