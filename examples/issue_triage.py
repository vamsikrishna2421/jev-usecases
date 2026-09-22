# Use case 23: abstention-gated issue triage.
#
# Pattern: one calibrated question per label, each with its own threshold scaled
# to the cost of being wrong. Below threshold -> stay quiet and say why.
# A bot that doesn't label is better than a bot that mislabels.
#
# See: https://github.com/emreozyoruk/hush

from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

# Threshold per label, scaled to the cost of a wrong action.
THRESHOLDS = {
    "bug": 0.80,
    "possible-duplicate": 0.85,
    "spam": 0.90,  # accusing someone of spam is expensive
    "needs-info": 0.85,
}


def triage_issue(title: str, body: str, repo_labels: list[str]) -> dict:
    answers = client.system_one(
        state={"title": title, "body": body[:3000], "labels_available": repo_labels},
        questions={
            "bug": Noul("This issue describes a defect in the software"),
            "possible-duplicate": Noul("This is a duplicate of an existing issue"),
            "spam": Noul("This is spam, promo, or a scam"),
            "needs-info": Noul("More information is required before this can be actioned"),
        },
    ).answers

    applied, abstained = [], []
    for label, question in answers.items():
        threshold = THRESHOLDS[label]
        belief = question.noul
        if belief >= threshold:
            applied.append(label)
        else:
            # Stay quiet, but report WHY — this is the whole point of hush.
            abstained.append(f"{label} {belief:.0%} < {threshold:.0%} (too unsure)")

    return {"applied": applied, "abstained": abstained}
