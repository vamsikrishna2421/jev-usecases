# Use case 21: live broadcast audience-message triage.
#
# Pattern: Jev classifies EVERY message cheaply; only messages that clear the
# "on-air" bar get an expensive LLM extraction call. Low-confidence decisions
# surface as "review" instead of being decided silently.
#
# See: https://github.com/marcemarin/radio-chat

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")


def classify_message(transcript: str, show_context: dict):
    answers = client.system_one(
        state={
            "transcript": transcript,
            "show": show_context,  # program, segment, topic of the day
        },
        questions={
            "intent": Choice(
                "What this listener message is about",
                {
                    "complaint": "Problem with the station or show",
                    "song_request": "Asking for a song or artist",
                    "greeting": "Hello / shout-out only",
                    "opinion": "Comment on the topic of the day",
                    "contest": "Entry for a contest or giveaway",
                    "question": "Asking the hosts something",
                    "spam": "Promo, scam, or repeated junk",
                    "other": "Does not fit the above",
                },
            ),
            "sentiment": Score(
                "Listener sentiment toward the show",
                ["Hostile or upset", "Neutral", "Positive and engaged"],
            ),
            "on_air_score": Score(
                "How good this message would be to read live on air",
                ["Not worth airtime", "Might work", "Must go on air"],
            ),
            "needs_review": Noul(
                "A producer should review this before it airs "
                "(sensitive, ambiguous, or risky content)"
            ),
        },
    ).answers
    return answers


def route_to_board(transcript: str, show_context: dict):
    answers = classify_message(transcript, show_context)

    if answers["intent"].choice == "spam":
        return "drop"
    if answers["needs_review"].noul > 0.5 or answers["on_air_score"].confidence < 0.6:
        return "producer_review"  # surface as "review" — never decided silently
    if answers["on_air_score"].score >= 2.5:
        # Only now spend the expensive LLM call (topic, name, summary)
        extract_with_llm(transcript)
        return "on_air_queue"
    return "archive"
