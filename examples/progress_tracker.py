"""Live checklist tracking: score every open talking point against the transcript.

Each open point is one Jev Score question; all of them are scored in parallel in
one request, roughly twice a second, against what has been said so far. Code
checks a point off at threshold and stops sending it.

Pattern: speculative fan-out — one call covers every open point, order does not
matter. Inspired by finetuningsingh/intelliprompter (measured: 383-word talk,
26 calls averaging 236 ms, $0.0011 total; original LLM build cost ~$40/hour,
Jev version ~$0.50/hour).
"""

import time

from typesafe_sdk import Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

CHECK_OFF_AT = 1.5  # between "mentioned" and "discussed"
RUBRIC = ["No mention of this topic at all", "Topic is mentioned",
          "Topic is discussed", "Topic is thoroughly discussed"]
POLL_SECONDS = 0.5
CONTEXT_CHARS = 6000


def track(transcript_stream, points: list[str], threshold: float = CHECK_OFF_AT):
    """Yield (point, score) each time a talking point is checked off."""
    open_points = dict(enumerate(points))
    said_so_far = ""
    while open_points:
        chunk = next(transcript_stream, None)  # speech arrives in chunks
        if chunk is None:
            break
        said_so_far += chunk
        response = client.system_one(
            state={"transcript": said_so_far[-CONTEXT_CHARS:]},
            questions={
                f"p{i}": Score(
                    instructions=f"How much has the speaker covered this talking point: {p}",
                    criteria=RUBRIC,
                )
                for i, p in open_points.items()
            },
        )
        for i, p in list(open_points.items()):
            if response.answers[f"p{i}"].score >= threshold:
                yield p, response.answers[f"p{i}"].score
                del open_points[i]  # checked points stay checked, no longer sent
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    def fake_speech():
        yield "Welcome everyone, I'm your host. "
        yield "Quick note: the emergency exits are behind you. "

    for point, score in track(fake_speech(), ["Welcome the audience",
                                              "Where the emergency exits are",
                                              "Tip the performers"]):
        print(f"checked off: {point} (score {score:.2f})")
