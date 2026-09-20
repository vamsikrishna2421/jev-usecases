# Jev Use Cases — cutting AI inference costs with a "System One" decision model

Study notes on **Jev**, the decision-only model released by [TypeSafe AI](https://typesafe.ai) on
September 15, 2026: what it is, where it fits, what early builders actually shipped, what it
costs, and where the vendor claims don't hold up. Written to answer one question:

> *Where can a typed-decision model replace an expensive LLM call in a real pipeline?*

**How to read the numbers:** anything tagged **(vendor)** is TypeSafe's own claim, self-run and
unreproduced. Anything tagged **(reported)** is a community self-report from launch week, not an
independent benchmark. The [Reality check](#6-reality-check) section puts both side by side.

Not affiliated with TypeSafe AI. Last updated: September 20, 2026.

## Contents

1. [What Jev is](#1-what-jev-is)
2. [How it works — the three primitives](#2-how-it-works--the-three-primitives)
3. [Five patterns worth stealing](#3-five-patterns-worth-stealing)
4. [Use-case catalog](#4-use-case-catalog) — 17 use cases with code sketches
5. [Cost math](#5-cost-math)
6. [Reality check](#6-reality-check)
7. [When NOT to use Jev](#7-when-not-to-use-jev)
8. [Getting access](#8-getting-access)
9. [Sources](#9-sources)

---

## 1. What Jev is

Jev is TypeSafe AI's first **"System One" model** — named after Kahneman's fast, intuitive
thinking in *Thinking, Fast and Slow*. It does not generate text at all. You hand it
unstructured program state (raw text, JSON, or arrays) plus a set of typed questions; it returns
schema-constrained decisions with calibrated probabilities in **one parallel pass**, in
70–500 ms end to end (vendor; community median 76 ms, reported).

The bet: most decisions inside software are System 1 judgments — *"which bucket is this?"*,
*"is this urgent?"*, *"does this match?"* — and we've been renting System 2 (slow,
deliberative LLMs) to make them.

| | Jev | Frontier LLM |
|---|---|---|
| Output | Typed values, schema-guaranteed | Generated strings, need parsing |
| Sampling | Parallel, single pass | Sequential, token by token |
| Input price | $0.042 / 1M tokens | $0.20–$10 / 1M tokens |
| Output price | Free | ~5x input |
| Structured-output errors | 0% by construction (vendor) | 0.58%–45.5% on vendor's test |
| Confidence | Calibrated per output | Overconfident, inconsistent |
| Latency | 70–500 ms (vendor) | 3 s – 329 s (vendor) |

Background: founded by **Diogo Almeida**, co-inventor of RLHF and InstructGPT at OpenAI;
$40M seed round led by DCVC; named after 19th-century economist William Stanley Jevons
(Jevons paradox). Trained with a method TypeSafe calls **RLCD — Reinforcement Learning for
Calibrated Decisions** — which optimizes probabilities against outcomes rather than human
preference. That's why calibration is a first-class property instead of something bolted on
via prompting.

## 2. How it works — the three primitives

The whole API is three question types. State can be a string, a JSON object, or an array of
text. **Text only** — no images, audio, or video (transcribe or caption first). Context limits
(vendor): 64k tokens for state + all questions together; 32k for state + the single longest
question.

- **Choice** — one option from a set (up to 255 options). Returns `.choice`, `.probabilities`
  (one per option), `.confidence`. Tip from the field: always include an explicit `other`
  option so the model can say nothing fits instead of picking the closest wrong thing.
- **Score** — a position on a 2–10 level rubric, described in words; the returned score can
  land *between* levels. Returns `.score`, `.probabilities`, `.confidence`.
- **Noul** — yes/no as a single probability from 0 to 1. Returns `.noul` (no separate
  confidence field — the number *is* the belief).

Every answer ships with a **calibrated probability**: when it says 0.9, it's right about nine
times in ten (vendor claim — validate the calibration on your own traffic before trusting a
threshold with money or safety).

```python
# pip install typesafe-sdk   (reads TYPESAFE_API_KEY from the environment)
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")  # pin the version; jev-latest moves

response = client.system_one(
    state={
        "ticket": {"subject": "Duplicate charge",
                   "text": "I was charged twice for order A-104. Please refund the duplicate."},
        "refund_policy": "Duplicate charges are eligible for a refund.",
    },
    questions={
        "department": Choice(
            instructions="Which team should handle this",
            criteria={"billing": "Payment or subscription issues",
                      "technical": "Bugs or integration problems",
                      "sales": "Pricing or account questions",
                      "other": "Anything else"}),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=["Calm, just stating facts",
                      "Frustrated but civil",
                      "Very angry, strong language"]),
        "refund_requested": Noul(
            instructions="The customer is explicitly asking for a refund"),
    },
)

print(response.answers["department"].choice,      # "billing"
      response.answers["department"].confidence)
print(response.answers["frustration"].score)
print(response.answers["refund_requested"].noul)  # e.g. 0.94
```

API shape follows the published SDK guides (Sep 2026); verify against current docs before
building on it. There is also `@typesafe-ai/sdk` for TypeScript and a raw
`POST https://api.typesafe.ai/v1/systemone` endpoint.

## 3. Five patterns worth stealing

Distilled from the community's practical guide (dev.to, Sep 2026):

1. **Speculative fan-out.** Questions are evaluated in parallel, so a tenth question costs
   tokens but almost no time. Ask everything up front and let code decide what was relevant —
   the opposite of the usual "cheap call first, follow up if needed" instinct. TypeSafe's
   cookbook reports batching 13 questions into one call is 12.2x cheaper and 10x faster than
   asking one at a time (vendor).
2. **Confidence-gated routing.** Set one threshold *per action*, scaled to what being wrong
   costs: auto-act above it, escalate below. Read-only lookup? Low bar. Moving money? High
   bar. See `examples/agent_router.py`.
3. **Composite scoring.** Break a fuzzy judgment into independent dimensions, score each
   atomically, combine with weights *you* control. Re-weighting becomes a code change, not a
   re-prompt. See `examples/lead_scoring.py`.
4. **The cascade.** Jev is not a replacement for a frontier model — it's the thing that
   decides which requests deserve one. Route the easy cases to code, the medium cases to a
   small LLM, and only the hard slice to the expensive model. See
   `examples/agent_router.py`.
5. **Retrieve, then judge.** Jev has no knowledge beyond the state you hand it, and accuracy
   falls as state fills with irrelevant material ("context rot"). Fetch precisely in code
   first, then judge cheaply.

**The meta-rule from TypeSafe's own docs:** *avoid asking the model something code can
compute exactly; avoid hiding several judgments inside one question.*

## 4. Use-case catalog

Seventeen use cases, ordered roughly by how much real-world evidence exists. Every code sketch
follows the SDK pattern from Section 2. The original twelve were added 2026-09-19 from
launch-week coverage (sources published Sep 15–19, 2026); entries added later carry their own
add date and source publish date.

---

### 1. Support / email triage & routing

**The problem.** Every ticket, email, or chat message currently costs an LLM call just to
decide *who should look at it* — before any actual work happens.

**How Jev fits.** One parallel call per message: a `Choice` over queues, a `Score` for
urgency/frustration, `Noul`s for flags like "refund requested" or "PII present". Route on
confidence; below threshold goes to a human queue. This is the canonical System One task.

**Real-world signal.** Launch-week builders reported sorting 500 emails for $0.035, and an
email-fraud detector screening 100 emails in 1.42 s with uncertain cases passed on for
further review (reported). TypeSafe's own eval included a customer-service workflow where
Jev tied GPT-5.6 Terra on accuracy at ~1/76th the cost per case (vendor).

**Sketch.**

```python
response = client.system_one(
    state={"subject": email.subject, "body": email.body[:2000]},
    questions={
        "queue": Choice("Which queue owns this",
                        {"billing": "...", "technical": "...", "sales": "...", "spam": "...", "other": "..."}),
        "urgency": Score("How urgent is this", ["Routine", "Time-sensitive", "Blocking the customer"]),
        "needs_human": Noul("This message needs a human rather than an automated reply"),
    },
)
if response.answers["needs_human"].noul > 0.6 or response.answers["queue"].confidence < 0.7:
    enqueue_for_human(email)
else:
    auto_reply(email, template_for(response.answers["queue"].choice))
```

---

### 2. Lead scoring & qualification

**The problem.** Scoring inbound leads with an LLM is accurate but bankrupts you at volume;
rule-based scoring misses nuance in titles, company descriptions, and intent signals.

**How Jev fits.** Composite scoring (Pattern 3): score budget authority, need, timing, and
fit as independent `Score`s, combine with weights you control in code. Re-weighting is a
config change, not a re-prompt — you can A/B it.

**Real-world signal.** A launch-week build reported scoring 700 leads for $0.09 (reported).
Full sketch in `examples/lead_scoring.py`.

---

### 3. Document classification — invoices, claims, legal & compliance checks

**The problem.** Invoices, insurance claims, contracts, and regulatory filings arrive as
unstructured text and need to land in the right bucket *and* get flagged for review —
millions of times a year.

**How Jev fits.** `Choice` over document types / review outcomes, `Score` for risk or
completeness, `Noul` for "missing required field" or "contains regulated data". Batch many
questions per document (Pattern 1): a 13-question regulatory briefing run as one call was
reported 12.2x cheaper and 10x faster than one question at a time (vendor cookbook).

**Real-world signal.** TypeSafe's 4-workflow eval included invoice processing and a
customer-service flow (vendor). Community: a plain-English rule engine reported $0.0001 and
0.7 s per check (reported). **Caveat:** Jev returns a number, not a rationale — in
regulated domains, reserve it for the high-volume routing layer and escalate flagged cases
to a model (or human) that can produce a written explanation.

**New signal — added 2026-09-20.** A community BANKING77 banking-intent classification
eval (published Sep 18, 2026) reported 92.40% accuracy vs 93.66% for a fine-tuned BERT
(−1.26 pt) at US$0.44 total test cost; an IRS tax-document page classifier reported 261
forms at 100% strict accuracy on the author's corpus, ~$0.001/page, self-reported 34×
cheaper and 6× faster than the author's prior LLM pipeline — surveyed in
[this Sep 20, 2026 finance-projects roundup](https://gist.github.com/drillan/6916b16e8ea31a8ec36c8f59d6483150)
(reported).

---

### 4. Content moderation

**The problem.** Every comment, review, listing, or upload needs a policy decision in
milliseconds — LLM moderation is too slow and too expensive at feed scale.

**How Jev fits.** `Noul` per policy ("contains hate speech", "is spam", "is sexual
content") plus a `Score` for severity; auto-remove above threshold, human-review queue
below. Parallel questions mean one call covers the whole policy list.

**Real-world signal.** Builders reported classifying 20,700 YouTube comments in 2 min 27 s
for $0.20, and skipping sponsor segments at $0.005 per video (reported). DataCamp notes
TypeSafe's pitch of scoring all 50M reviews in a product table for ~$20 in decision calls
(vendor).

---

### 5. LLM-as-judge / eval & rubric scoring

**The problem.** Teams already pay a frontier model to grade other models' outputs
(RLHF-style evals, agent benchmarks, code review) — the judge call is often *more*
expensive than the generation it grades.

**How Jev fits.** Replace the judge with `Score` on a written rubric + `Noul`s for
specific failure modes. The dev.to guide's staged code reviewer does exactly this: a
`Noul` risk matrix first, then `Choice`/`Score` file profiles, evidence selection, and
severity — with conditional routing on the results.

**Real-world signal.** A PR-review build reported $0.00007 per PR with an answer in half a
second; an ESLint-rule checker reported 90% agreement with the rule's own verdict
(reported). On TypeSafe's workflow eval, Jev matched GPT-5.6 Terra's agreement score
(67.8% vs 67.9%) at ~1/76th the cost (vendor).

---

### 6. Agent-loop routing, tool-use guardrails & safety checks

**The problem.** In an agent loop, *every* decision — which tool next, which model next,
is this action safe — currently burns a full LLM call, most of whose output is discarded.

**How Jev fits.** Jev picks the next action from a `Choice` over tools/models and runs
`Noul` safety checks ("does this tool call look malicious?", "does the plan match the
user's intent?") at 70–500 ms. This is the cascade (Pattern 4): Jev decides which requests
deserve the expensive model.

**Real-world signal.** The standout launch-week build: a browser agent (Browser Use) that
turned each page into a numbered element table, had Jev pick the operation (`CLICK`,
`TYPE_TEXT`, `SELECT`, `SCROLL`, `WAIT`, `DONE`, `BLOCKED`) *and* its target in one call,
and only invoked a small LLM for free-text typing — booking a real Zürich→London flight in
7.1 s for $0.0039 (reported). A computer-use agent drove a Mac at $0.0002 per decision vs
$0.032 for a bare-screenshot Opus 5 call (reported). A Claude Code session was reported
cut from ~1M tokens to 86K by routing decisions through Jev (reported).

**New signal — added 2026-09-20.** Vercel's open agent framework *eve* ships Jev
(`typesafe-ai/jev`) as the default evaluation model in its `auto()` human-in-the-loop
tool-approval path
([eve docs](https://github.com/vercel/eve/blob/HEAD/docs/tools/human-in-the-loop.md),
updated Sep 17, 2026); Vercel engineer Pranit Sharma reported 5–18x faster responses vs
ChatGPT Luna 5.6 for command safety checks with improved classification accuracy
(reported in
[this Sep 18, 2026 writeup](https://www.omegatechnologysolutionsgroupinc.com/blog/typesafe-ais-jev-model-chooses-actions-not-words-ca35b3)).

**Sketch** — the cascade in `examples/agent_router.py`:

```python
r = client.system_one(state=user_message, questions={
    "intent": Choice("Primary intent", {"order_status": "...", "product_question": "...",
                                        "return_exchange": "...", "complaint": "..."}),
    "complexity": Score("How complex to resolve", ["Simple lookup", "Needs judgment", "Escalation needed"]),
})
intent, cx = r.answers["intent"], r.answers["complexity"]
if intent.confidence < 0.5:
    route_to_human(user_message)
elif intent.choice == "order_status":
    lookup_order(user_message)                    # pure code, no model at all
elif intent.choice == "complaint" and (cx.score > 1 or cx.confidence < 0.5):
    route_to_human(user_message)
else:
    handle_with_llm(user_message, specialist_for(intent.choice))  # only the hard slice
```

---

### 7. RAG relevance scoring & context trimming

**The problem.** RAG pipelines retrieve wide and then feed everything — including junk
passages — to an expensive generator. Context is the most expensive thing in the stack.

**How Jev fits.** Retrieve-then-judge (Pattern 5): a `Noul` per passage ("is this passage
relevant to the question?") filters *before* anything expensive sees it. At $0.042/MTok
input with free output, the filter costs less than the context window it saves.

**Real-world signal.** TypeSafe ships RAG-passage-classification and citation-check
cookbooks for exactly this (vendor). A community plugin scored each tool call in an agent's
session history and pruned records the agent no longer needed — reported as a 50% context
reduction, though what gets lost in pruning vs. summarization is genuinely contested
(reported, debated). One caution from the field: a well-calibrated judgment about badly
retrieved material is still a judgment about bad material — retrieval quality sets the
ceiling.

---

### 8. Data-pipeline quality checks

**The problem.** Pipelines need per-row or per-batch judgments — "is this record sane?",
"which schema does this blob match?", "is this anomaly real?" — that are too fuzzy for
SQL but too numerous for an LLM.

**How Jev fits.** Map a Jev decision over rows: `Choice` for classification, `Noul` for
validity checks, `Score` for anomaly severity. ~$0.0004 per case (vendor) makes per-row
judgment economically viable for the first time.

**Real-world signal.** A DuckDB extension ran 1,000 row classifications in SQL in about
10 seconds (reported). TypeSafe pitches map-reducing decisions across petabytes of raw
data into features (vendor).

---

### 9. Entity-resolution match adjudication ⭐

**The problem.** Record-linkage pipelines match inbound leads against customer/account/
location data. Deterministic + fuzzy matching resolves the easy majority, but the
ambiguous remainder gets routed to an LLM for "do these two records describe the same
entity?" — at LLM prices, on every run.

**How Jev fits.** Keep the deterministic stage exactly as is. Replace the LLM
adjudication call with one Jev call per ambiguous pair: a `Choice`
(`match` / `no_match` / `needs_review`), a `Score` for match strength, and `Noul`s for
individual signals ("names differ only by spelling/abbreviation", "addresses agree").
Then confidence-gate the outcome: auto-merge above 0.9, auto-reject below, and send only
the uncertain slice to human review (or to the LLM for a written rationale when you need
an audit trail).

**Why this matters.** If you resolve ~75% of records deterministically today and route the
rest to Claude-class models, the adjudication layer is where the budget burns. A typed
decision at ~$0.0004/case (vendor) vs. a multi-thousand-token LLM call turns a
cost-center step into a rounding error — while calibrated confidence gives you the same
escalation semantics your human-in-the-loop review app already uses. Keep the LLM for the
hard 5–10% where you need explainable reasoning; let Jev clear the middle.

**Worked sketch** in `examples/entity_resolution.py` — mirrors a real production shape:
Snowflake/Snowpark deterministic stage → Jev adjudication → confidence-gated routing.

```python
verdict, strength, _, _ = adjudicate_pair(record_a, record_b, deterministic_score)
if verdict.choice == "match" and verdict.confidence >= 0.9:
    action = "auto_merge"
elif verdict.choice == "no_match" and verdict.confidence >= 0.9:
    action = "auto_reject"
else:
    action = "human_review"   # low-confidence + needs_review land here
```

**Caveats before migrating a production pipeline:** Jev returns no rationale (keep the
LLM/human for cases needing explanations); validate its calibration on *your* labeled
match pairs before trusting a threshold; early access is waitlisted, so this is a design
to prototype, not a drop-in replacement today.

---

### 10. Batch labeling & annotation

**The problem.** Training data, eval sets, and research corpora need thousands of
consistent labels. Human labeling is slow; LLM labeling is expensive and inconsistent.

**How Jev fits.** One `Choice` per item over your label taxonomy, run as a batch job.
Deterministic schema means zero malformed labels by construction.

**Real-world signal.** The cleanest economics demo of launch week: 1,018 research papers
summarized with a generative model ($3.99) then classified into 24 topics with one Jev
`Choice` each — **$0.08 total, 256 ms median per paper** (reported). A simulated survey
(150 personas × 12 questions) ran for about ¥1.8 in ~5 seconds (reported) — cheap
synthetic labeling, though it says nothing about agreement with real respondents.

---

###  11. Fraud & risk scoring

**The problem.** Transactions, claims, signups, and support messages need a risk verdict in
real time. Rules are brittle; ML models need training data; LLMs are too slow for the hot
path.

**How Jev fits.** `Score` for risk + `Noul` for specific fraud signals, evaluated in
milliseconds inside the request path; escalate on low confidence or high score. Zero-shot
means no training-data project to get started — though for a stable task *with* labeled
data, a specialized classifier remains the baseline to beat.

**Real-world signal.** Email fraud detection over 100 emails in 1.42 s with uncertain
cases passed on (reported); insurance-claim classification and a plain-English rule
engine at $0.0001 per check (reported); one market-making bot making a buy/sell decision
per ~300 ms block at ~81 ms model latency (reported).

---

### 12. Real-time judgment in tight loops

**The problem.** Games, robotics, trading, and interactive agents need a judgment call
every few hundred milliseconds — a 3–30 s LLM call is a non-starter.

**How Jev fits.** 70–500 ms typed decisions inside the loop, with the safety-critical
layers kept in deterministic code. The drone build is the model citizen here: geometric
flight control at 500 Hz and safety reflexes at 50 Hz stay in code; classical CV
compresses the camera feed to symbolic state; Jev contributes *advisory* tactical
judgment at ~2.5 Hz (one `Choice` over manoeuvres, one risk `Score`, one `Noul`).

**Real-world signal.** TypeSafe's own demo has Jev playing Doom from structured game
state at ~10 decisions/second for ~$7/hour of inference (vendor). Community: an
autonomous drone built in 15 minutes on $0.10 of inference; Super Mario played from
emulator RAM translated to object-centric JSON (reported). The honest lesson from the
computer-use build: *"every piece of reasoning the frontier model does for free has to be
rebuilt here as deterministic state."*

---

### 13. Voice command interpretation → typed actions

**Added 2026-09-20.** Source published Sep 17, 2026 —
[moritzkremb/jev-voice-browser](https://github.com/moritzkremb/jev-voice-browser).

**The problem.** Voice interfaces need sub-second interpretation of free speech into
*executable* actions — intent plus target — and an LLM call adds seconds and dollars per
utterance.

**How Jev fits.** Transcribe locally, then one Jev call per utterance: a `Choice` over the
action vocabulary plus a `Choice` over on-page targets (collected by code from the DOM),
decided in ~300 ms per spoken word; deterministic Playwright code executes. A voice-drawing
build reports ~350 ms per spoken word deciding action, target, and place (reported in the
[awesome-jev directory](https://github.com/hellogumbo/awesome-jev/blob/main/README.md),
refreshed Sep 20, 2026).

**Real-world signal.** The browser build's integration suite (27 real-API cases on captured
page fixtures, Sep 2026) passed 27/27, with Jev latency averaging ≈330 ms (p50 ≈300 ms,
3–6k input tokens per request); a 16-command headed demo against real sites cost ≈$0.01
(reported).

Full sketch in `examples/voice_router.py`.

---

### 14. Semantic grep — filter lines by meaning

**Added 2026-09-20.** Source published Sep 18, 2026 —
[keltokhy/jgrep](https://github.com/keltokhy/jgrep).

**The problem.** `grep` matches patterns; teams need to match *meaning* across logs,
titles, diffs, and records — "a user is getting frustrated", "removes error handling for a
persistent write" — work that previously needed an LLM per line.

**How Jev fits.** Each line (or diff hunk, function, paragraph, CSV row) becomes one `Noul`
question — "does this match the description?" — judged concurrently in input order. Works
on live streams (`tail -f`) because decisions arrive in ~200 ms. Judgment quality scales
with description quality; a `--estimate` mode previews calls and cost with no key.

**Real-world signal.** Measured on 994 Hacker News titles: 4.6 seconds and $0.012 for one
description — about 200 ms and a thousandth of a cent per line — and the same time for
three descriptions at once (reported).

Full sketch in `examples/semantic_grep.py`.

---

### 15. Decision-to-UI — Jev picks the interface

**Added 2026-09-20.** Sources published Sep 17–19, 2026 —
[Instinct](https://github.com/joevidev/ui-generator-instinct-jev) (repo created Sep 17,
2026; live site ui-generator-instinct-jev.vercel.app); Chris Tate's Jev + json-render
experiment ([video posted Sep 19, 2026](https://www.instagram.com/reel/DdeCIFxoZgL/)).

**The problem.** "AI UI generators" ask a model to write code or copy and hope it's valid
and on-brand. For dashboards, forms, and admin surfaces the UI space is really a *catalog*
— the decision is which component or block to render and how to configure it.

**How Jev fits.** Describe the case in free text; Jev never generates code. Hierarchical
`Choice` calls pick a component family, then a leaf (48 real shadcn components / 17 page
blocks in Instinct), then `Noul`/`Score` questions configure props and content — all from
bounded, real option sets; code renders the winner. Zero free text anywhere in the
pipeline except the user's own input. Confidence-gated escalation (call 3 tiebreak) only
fires when the margin is thin.

**Real-world signal.** Instinct verified live against the real Jev API across components,
blocks, and the tiebreak/escalation path (reported; not every catalog entry individually
visually spot-checked). A separate Sep 19 build demonstrated typed decisions rendered as
UI in milliseconds — e.g. a London→Edinburgh train-ticket interface (reported).

---

### 16. Live checklist / progress tracking

**Added 2026-09-20.** Source published Sep 19, 2026 —
[finetuningsingh/intelliprompter](https://github.com/finetuningsingh/intelliprompter)
(independent rebuild of a TypeSafe Discord town-hall demo).

**The problem.** Speakers, support agents, and operators work from talking-point lists and
need to know — in real time, in any order — what they've already covered. The original
LLM implementation cost ~$40/hour.

**How Jev fits.** Each open talking point is a `Score` question ("how much has the speaker
covered this point?"); all open points are scored in parallel in one request every 500 ms
against the transcript so far. Code checks a point off at threshold (1.5 default) and
stops sending it. A passing mention can check a point off at low thresholds — tune the
threshold per list.

**Real-world signal.** With Jev the same job costs at most ~$0.50/hour (two calls/sec at
~$0.00007 each), ~80x cheaper than the LLM version. Measured replay: a 383-word talk with
7 points — all 6 covered points checked off, the uncovered point peaking at 0.03 (26
calls, avg 236 ms, $0.0011 total); a 1,581-word talk — all 6 covered points checked, 106
calls averaging 252 ms, $0.0074 (reported).

Full sketch in `examples/progress_tracker.py`.

---

### 17. Financial news desk — trade-idea triage

**Added 2026-09-20.** Source published Sep 18, 2026 —
[0xnairb/research_desk](https://github.com/0xnairb/research_desk).

**The problem.** Research desks want fast, repeatable first-pass analysis of news and
tickers — ranked, grounded, routed trade ideas — without paying a reasoning model per
headline.

**How Jev fits.** One market snapshot → 12 independent Jev questions (trend, momentum,
sentiment, risk, …) → an in-code rule engine (BUY gate: trend > 0.75 AND momentum > 0.70,
…). Every number the UI shows is a typed answer from one request; thresholds, risk vetoes,
and order placement stay deterministic. Dry-run/paper by default.

**Real-world signal.** Working demo with live yfinance data, a five-stage pipeline, and a
call-recording tab showing the exact state and questions behind every number (reported;
thresholds are starting guesses, not values fitted to real outcomes).

**Relationship to use case 11.** That section's market-making bot is hot-path, per-block
trading; this is analyst-desk triage of news and tickers into ranked trade ideas — a
slower, research-shaped loop.

---

## 5. Cost math

**Published pricing (vendor):** $0.042 per million input tokens ($42 per billion).
**Output tokens are free** — "too cheap to meter," because the model returns structured
decisions, not long text.

### Worked example: 10,000-item classification batch

Assumptions: ~1,000 input tokens per item (a ticket + a few questions), one Jev call each.
LLM baseline uses TypeSafe's own per-case figure for GPT-5.6 Terra ($0.0304/case), which
includes its output tokens.

| | Jev | Frontier LLM (Terra-class) |
|---|---|---|
| Input tokens | 10M → 10 × $0.042 = **$0.42** | (bundled in per-case figure) |
| Output tokens | free | (bundled in per-case figure) |
| **Total for 10K items** | **≈ $0.42** | **≈ $304** (10,000 × $0.0304) |
| Implied ratio | — | **~720x cheaper** |

Two honest footnotes on that 720x:

1. It uses **vendor per-case figures** — treat it as the optimistic bound, not a measured
   result. The best *measured* community median is **30x** cost reduction (see Reality
   check).
2. It's also the *wrong-shaped* comparison for most real migrations: you rarely replace an
   LLM 1:1 with Jev. The realistic pattern is the **cascade** — Jev handles the bulk,
   the LLM keeps the hard slice. On a million-ticket cascade using TypeSafe's figures,
   the bill was ~$6,480 instead of ~$30,400 (vendor) — a 4.7x saving on the *whole
   system*, with ~800K tickets answered in under half a second instead of ten.

### A real bill, not a model

The 1,018-paper pipeline (reported, launch week): summarization with a generative model
cost **$3.99**; the Jev classification step over all 1,018 papers cost **$0.08** — about
$0.00008 per paper. That's the shape to aim for: generative models where generation is
needed, decision models everywhere else.

## 6. Reality check

| Metric | TypeSafe's claim | Community-measured |
|---|---|---|
| Speed-up | 193.6x (homepage), 20–200x (launch) | **Median 7x** across 215 reported figures (quartiles 2x–20x) |
| Cost reduction | 444.6x (homepage), 40–400x (launch) | **Median 30x** across 180 reported figures (quartiles 5x–85x) |
| Latency | 70–500 ms end to end | **Median 76 ms** across 333 figures (quartiles 2–270 ms) |
| Accuracy (4 workflows) | 67.8% agreement, ≈ Terra's 67.9% | No independent reproduction yet |
| Structured-output errors | 0% by construction | Asserted, not measured ("schema matching is guaranteed, thus we can confidently add 0% into the plots" — vendor) |
| "Cannot hallucinate" | Mathematically impossible | **Narrower than it sounds:** Jev cannot emit an *invalid* value. It can absolutely return a schema-valid answer that is factually *wrong*. |

Source for the community column: an analysis of 12,759 launch-week posts (Sep 15–18, 2026)
that separated authors' own measurements from repeated vendor figures. It's a survey of
public reports, not an independent benchmark — but it's the best evidence available, and
it says the gains are real and much smaller than the homepage multipliers.

**Why the gap?** A speed-up means little until you know the baseline. Replacing a long
reasoning call with a short classification call is a different comparison from replacing
a small model already configured to return one short answer. The practical comparison is
*the shortest reliable call your application can already make* — measure against that,
not against a homepage number.

**Accuracy, precisely:** the 67.8% is not accuracy against ground truth. TypeSafe built
consensus labels by averaging two frontier models at high thinking effort, then scored
everyone against those — so it measures *agreement with two frontier LLMs*, which biases
toward OpenAI/Anthropic-style answers. And the workflows were written by TypeSafe's own
team. Promising, not settled — run your own evals on your own traffic (several launch-week
builders did exactly this before migrating).

**Price sustainability:** TypeSafe itself says it can't prove the pricing isn't
subsidized, though it expects prices to fall rather than rise. Don't architect as if
$0.042/MTok is a law of nature.

## 7. When NOT to use Jev

- **You need text, code, or a summary out.** Jev generates nothing. (People have built
  text generation *around* it — 29 yes/no questions per character — as a stunt, not a
  strategy.)
- **You need a written rationale.** No explanations come back — a problem for debugging
  and for audits in regulated domains. Escalate those cases to a model or human that can
  explain.
- **The answer space isn't bounded and known up front.** Jev can't write its own schema.
- **You need exact computation.** It doesn't count reliably, dates are text to it (not
  ordered quantities — extract with a `Choice` over enumerated values, do the arithmetic
  in code), and contradictory instructions confuse it.
- **The state is adversarial.** User-controlled text engineered to argue for its own
  classification can move the answer — that's your threat model to handle.
- **The state is bloated.** Accuracy falls as irrelevant material fills the context
  ("context rot"). Retrieve and filter in code first.
- **You have a stable task with labeled data.** A fine-tuned specialized classifier is
  still the baseline to beat — Jev's advantage there is setup speed, not necessarily
  quality or cost.
- **You need images, audio, or video understanding.** Text-only; transcribe/caption first.

## 8. Getting access

- **Direct API:** waitlisted early access at `console.typesafe.ai` → API keys; endpoint
  `POST https://api.typesafe.ai/v1/systemone`; early-access route `jev-latest`
  (currently resolves to `jev-1.13.0` — **pin the version** if you tune thresholds, and
  log the `model` field from responses).
- **No waitlist:** Vercel AI Gateway (reported as the fastest-adopted model in Gateway
  history — ~13% of paid teams within 24h), OpenRouter (beta), AI/ML API.
- **SDKs:** `pip install typesafe-sdk` (Python 3.10+) / `npm install @typesafe-ai/sdk`
  (Node 20+); both read `TYPESAFE_API_KEY` and default to `jev-latest`. There is also a
  coding-agent skill (`typesafe-ai/skills`).
- **Rate limits** for `jev-1.13` (vendor, "moving without notice"): 250,000 tokens/sec,
  1,200 requests/min; SDKs retry with exponential backoff.

## 9. Sources

Source of truth, roughly in order of usefulness:

- TypeSafe AI — Introducing System One Models and Jev (official announcement):
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe docs — the "jaggedness" page (unusually honest failure-mode list):
  https://docs.typesafe.ai *(linked from the guides below; verify current paths)*
- "How to Use Jev: A practical guide to TypeSafe's System One model" (setup, 5 patterns,
  failure modes, launch-week builds) — https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- "Jev: TypeSafe's System One Model Explained" (benchmarks, pricing, when-to-use table) —
  https://www.datacamp.com/blog/system-one-models-jev
- "Jev by TypeSafe AI: What 12,759 tweets tell us" (vendor claims vs community
  measurements — the reality-check data) — https://openchamber.dev/blog/jev-typesafe-ai/
- "Jev by TypeSafe AI: System One Model Guide for Startups" (startup use cases, Vercel
  Gateway patterns) — https://saascity.io/blog/system-one-models-jev-typesafe-ai-2026
- "Jev Makes Fast and Cheap Decisions" (decision-model framing, enterprise use cases) —
  https://patmcguinness.substack.com/p/jev-makes-fast-and-cheap-decisions
- LangChain — "Building a harness with Jev" (Jev inside the agent loop) —
  https://www.langchain.com/blog/building-a-harness-with-jev
- "Jev: The ChatGPT Co-Creator's System One Model Can't Talk" (limits, edges) —
  https://dev.to/lukeocodes/jev-the-chatgpt-co-creators-system-one-model-cant-talk-3774
- "TypeSafe's Jev: An AI Model That Answers in Types, Not Text" (reading the launch
  closely) — https://logicdecode.in/blog/typesafe-jev-system-one-model-2026
- Latent Space AI News — Jev launch coverage (community reactions, DSPy parallels) —
  https://www.latent.space/p/ainews-jev-a-system-one-model-that
- Wikipedia — Jev (AI model) (launch facts, funding, background) —
  https://en.wikipedia.org/wiki/Jev_(AI_model)
- Doomers — TypeSafe AI launch case study (adoption numbers) —
  https://doomers.ai/work/typesafe-ai-case-study

Community builds referenced above (all launch-week, self-reported): `browser-use/jev-ultrafast`,
`awlevin/typesafe-computer-use`, `jarrodwatts/jev-trader`, `RomanSlack/jev-drone`,
`fhshaik/typesafe-mario`, `devagrawal09/jev-review`, `TheoLeeCJ/openjev`,
`AbdelStark/awesome-typesafe`, `1kpapers.com`.
