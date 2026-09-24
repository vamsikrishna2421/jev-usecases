"""Document-packet splitting with Jev: find the boundaries between sub-documents.

The pattern behind jerryjliu/docjev (reported 6x faster than gpt-5.6-luna at
equivalent accuracy): OCR each page once (liteparse for digital docs, LlamaParse
for complex pages), then ask Jev ONE question per page — "does a new document
start here?" — as a Noul plus a Choice over the page's document type. All pages
are judged in a single parallel call, and boundaries fall out of the Noul
threshold in code. Jev never writes text; your splitter does the assembly.

Jev never explains *why* a page is a boundary — it only returns the judgment.
In regulated flows (claims, legal packets), keep the low-confidence pages for
human review and log every answer for audit.
"""

from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")


def split_packet(pages: list[str], new_doc_threshold: float = 0.75) -> list[list[str]]:
    """pages: OCR text of each page in order. Returns lists of pages per sub-document."""
    questions = {}
    for i, text in enumerate(pages):
        questions[f"page_{i}_is_boundary"] = Noul(
            f"A new document begins on this page (not a continuation of the previous one). "
            f"Page {i + 1} of {len(pages)}:\n{text[:1500]}"
        )
        questions[f"page_{i}_doctype"] = Choice(
            f"Document type of page {i + 1}:\n{text[:800]}",
            {
                "invoice": "A bill requesting payment",
                "claim_form": "An insurance or benefit claim form",
                "contract": "A legal agreement or contract page",
                "supporting": "Receipt, photo, or attachment supporting another document",
                "other": "Does not fit the above",
            },
        )

    response = client.system_one(
        state={"packet": f"{len(pages)} pages, scanned in order"},
        questions=questions,
    )
    answers = response.answers

    packets, current = [], []
    for i, text in enumerate(pages):
        is_boundary = answers[f"page_{i}_is_boundary"].noul
        doctype = answers[f"page_{i}_doctype"].choice
        conf = answers[f"page_{i}_doctype"].confidence
        if i > 0 and is_boundary >= new_doc_threshold:
            packets.append(current)
            current = []
        current.append({"text": text, "doctype": doctype, "doctype_confidence": conf,
                        "boundary_prob": is_boundary})
    packets.append(current)
    return packets


if __name__ == "__main__":
    demo = ["INVOICE #2041\nTotal due: $412.00", "Proof of delivery — signed J. Doe"]
    for n, pkt in enumerate(split_packet(demo)):
        print(f"--- sub-document {n + 1}: {[p['doctype'] for p in pkt]} ---")
