"""Voice-driven browser control: free speech in, typed action out.

Transcription stays local; one Jev call per utterance returns a Choice over the
action vocabulary plus a Choice over on-page targets collected by code from the
DOM. Deterministic code executes — Jev never hallucinates an action.

Pattern: confidence-gated routing — below threshold, keep listening instead of
acting. Inspired by moritzkremb/jev-voice-browser (measured: ~330 ms avg
decision latency, 27/27 integration cases, ~$0.01 for a 16-command demo).
"""

from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

ACTIONS = {
    "click": "Activate the chosen page element",
    "type_text": "Type dictated text into the chosen field",
    "go_back": "Navigate back one page",
    "scroll": "Scroll the page",
    "wait": "Pause briefly",
    "ignore": "No clear action — do nothing",
}

CONFIDENCE_BAR = 0.7


def interpret(transcript: str, targets: dict[str, str]) -> dict | None:
    """Return {'action', 'target'} for a spoken command, or None to keep listening."""
    response = client.system_one(
        state={
            "transcript": transcript,  # partial transcripts are fine — closed-set actions commit early
            "targets": targets,  # {"t1": "Search button", ...} built by code from the DOM
        },
        questions={
            "action": Choice(
                instructions="What the user wants done",
                criteria={**ACTIONS, "other": "Anything else"},
            ),
            "target": Choice(
                instructions="Which page element the action applies to, by id",
                criteria=targets,
            ),
            "commits": Noul(
                instructions="The transcript clearly commits to an action"
            ),
        },
    )
    answers = response.answers
    if answers["commits"].noul < CONFIDENCE_BAR:
        return None  # ambiguous: wait for more speech, never guess
    return {"action": answers["action"].choice, "target": answers["target"].choice}


if __name__ == "__main__":
    cmd = interpret(
        "click the search button",
        {"t1": "Search button", "t2": "Cart link", "t3": "Search field"},
    )
    print(cmd)  # -> {'action': 'click', 'target': 't1'}  (then Playwright acts)
