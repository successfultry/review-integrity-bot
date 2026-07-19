# Review Integrity Bot

Async FastAPI app that computes an **honest ("true") rating** from noisy public reviews.

## Day 35 Task

Automate the integrity check of a place rating: fetch live reviews, filter noise/manipulation/irrelevant content, and recompute a trustworthy score that can be compared with the official source rating.

How AI is used:

- The LLM classifies each review quality into strict labels (`valid`, `empty`, `speculative`, `spam_offtopic`, `low_effort`) using strict JSON output.
- The LLM generates a Russian high-level summary for relevant reviews: `summary_ru`, `pros_ru`, `cons_ru`.
- The service computes `true_rating` from LLM-assisted filtering and weighting (with Bayesian shrinkage), instead of using a raw star average.

## Why this project exists

Public star ratings are easy to game and easy to misread:

- Businesses buy or beg for 5-star reviews, so the average is inflated.
- Many "reviews" are star-only, one word ("good"), off-topic spam, or written by people who never used the place.
- The sample you actually see (first page, "most relevant") is usually **not** representative of all reviews.

The goal of this agent is to answer one question a human actually cares about:

> "If I strip away the noise and only trust real, first-hand, textual feedback, what rating does this place deserve?"

It does that by fetching reviews, classifying the quality of each one with an LLM, throwing away the untrustworthy ones, and recomputing a weighted, statistically-shrunk rating you can compare against the official number.

## What it produces

For any place you search, you get three comparable numbers plus an explanation:

- **`official_rating`** — what the source (Google/SerpApi) reports over ALL reviews.
- **`naive_rating`** — simple average of just the sample we fetched (a bias check: shows how skewed the fetched page is).
- **`true_rating`** — the honest rating: computed ONLY from reviews with meaningful, non-spam text, weighted by quality and confidence, then Bayesian-shrunk for small samples.

Plus (when LLM is on): a Russian summary with pros/cons, a per-class breakdown, and per-review labels with reasons.

## How the LLM is used

The LLM is the "quality judge", not a rating generator. It never invents a score — it only labels the *nature* of each review. Pipeline:

1. **Ingest** — an adapter pulls reviews from a source (`serpapi` or `google_maps`) and normalizes them into an internal `Review` model. Services never call external APIs directly; everything goes through adapters.
2. **Sanitize** — review text is treated as 100% untrusted input. We normalize unicode, strip invisible/control chars, and neutralize any fake delimiters (prompt-injection defense).
3. **Classify (LLM)** — each review is sent to the model with a strict system prompt and wrapped in unique nonce markers (`<<REVIEW:xxxx>> ... <<END:xxxx>>`) so the model can't be tricked by instructions hidden inside the review text. The model returns **strict JSON** (enforced via `json_schema`, `strict: true`) with:
   - `label` ∈ `valid | empty | speculative | spam_offtopic | low_effort`
   - `reason` (always English, quotes the evidence)
   - `confidence` (0..1)
   If confidence is below the threshold, the label is downgraded to `uncertain`. If the LLM is disabled or fails, a deterministic **heuristic fallback** classifies instead (regex rules), so the app never hard-crashes.
4. **Score** — `services/score.py` computes `true_rating` using only contributing classes:
   - `valid` → weight 1.0
   - `low_effort` → weight 0.5
   - everything else (`empty`, `speculative`, `spam_offtopic`, `uncertain`) → excluded (weight 0)
   Each contributing review is further weighted by its (capped) confidence, and the result is blended with a Bayesian prior so a tiny sample can't produce a wild rating.
5. **Summarize (LLM, optional)** — one batched call over the contributing reviews produces a Russian `summary_ru` + `pros_ru` / `cons_ru`.

### Why exclude star-only / speculative / spam
A rating with no verifiable text is not trustworthy: you can't check if the person actually used the place. The whole point of `true_rating` is "only count opinions you could, in principle, verify from the text." That's an intentional, opinionated design choice.

## Two run modes

- **`USE_LLM=false` (offline)** — no OpenAI calls; classification uses the heuristic engine only. Great for tests, eval, and demos without spending money.
- **`USE_LLM=true` (live)** — real OpenAI classification + Russian summary. Needs `OPENAI_API_KEY`.

## Observability

Every classification and summary call logs structured JSON with one `trace_id` per request: model, prompt/completion tokens, estimated cost (USD), and latency. The final `AnalysisResult.usage` aggregates total tokens and cost for the whole run so you can see exactly what an analysis cost.

## Architecture (layered)

```
adapters/  -> normalize external payloads (serpapi, google_maps, fixture) into internal Review models
services/  -> business logic: classify, score, summarize, analyze (NO direct external API calls)
models/    -> strict Pydantic contracts at API + LLM boundaries
core/      -> config, cost math, structured logging
web/       -> FastAPI endpoints + Jinja UI
eval/      -> golden dataset + accuracy / precision / recall / F1 / confusion matrix
tests/     -> unit tests for each layer
data/      -> fixture reviews for offline/eval only (never a live source)
```

At a glance:

- Ingests reviews from Google Places API (New) and SerpApi.
- Classifies each review (`valid`, `empty`, `speculative`, `spam_offtopic`, `low_effort`, `uncertain`).
- Recomputes rating with quality weights, confidence cap, and Bayesian shrinkage.
- Logs per-review token/cost/latency with one trace id per request.
- Fixture JSON is for local/eval only — never used as a live API/UI source.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and set your source keys.

Live sources:
- `google_maps`: set `GOOGLE_MAPS_API_KEY` (official Places API, max 5 reviews).
- `serpapi`: set `SERPAPI_KEY` (+ optional `SERPAPI_REVIEWS_LIMIT`, default 200).

Optional: `OPENAI_API_KEY` + `USE_LLM=true`.

### Source behavior

| Source | Coverage | Cost profile | Notes |
|---|---|---|---|
| `google_maps` | Up to 5 reviews | Low | Official API limit is strict |
| `serpapi` | Many reviews via pagination | Paid quota | Best for robust sample sizes |

## Quick start for testers (5 minutes)

1. `uv sync`
2. Copy `.env.example` → `.env`, put in `SERPAPI_KEY` (and `OPENAI_API_KEY` if you want real LLM output).
3. Start the app:

```powershell
$env:USE_LLM="true"          # or "false" to skip OpenAI
$env:OPENAI_API_KEY="..."    # only needed if USE_LLM=true
$env:SERPAPI_KEY="..."       # needed for source=serpapi
uv run uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```

4. Open `http://127.0.0.1:8000` and try:
   - Source `serpapi`, query `Gorky Park Moscow`, sort `newest`, `reviews_limit=100`.
5. Read the result cards top-to-bottom: place header → Official vs True → (Итог RU) → Class Breakdown → Reviews.

What to look for:
- `True` differs from `Official` (that's the whole point — noise removed).
- `Excluded` count > 0 (empty/spam/speculative got dropped).
- Each review has a `label`, `method` (`llm`/`heuristic`/`fallback`), and an English `reason`.
- With `USE_LLM=true`, an `Итог (RU)` card with pros/cons appears.

## Run

### Offline classifier checks (no Google key)

```powershell
$env:USE_LLM="false"
uv run pytest -q
uv run python -m eval.run_eval
```

### API / UI (live source)

```powershell
$env:GOOGLE_MAPS_API_KEY="your_key_here"
$env:SERPAPI_KEY="your_key_here"   # required for source=serpapi
$env:USE_LLM="false"   # or true + OPENAI_API_KEY
uv run uvicorn web.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"source\":\"google_maps\",\"source_id\":\"Central Park New York\",\"sort\":\"newest\",\"reviews_limit\":5}"
```

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d "{\"source\":\"serpapi\",\"source_id\":\"Central Park New York\",\"sort\":\"newest\",\"reviews_limit\":50}"
```

On source failure: HTTP 503 `{"detail":{"error":"source_error",...}}`.

Source notes:
- `google_maps`: max 5 reviews per place from official API (cheap, strict limit).
- `serpapi`: more reviews via paid quota (results size controlled by `SERPAPI_REVIEWS_LIMIT` or request `reviews_limit`).
- `sort`: one of `newest`, `most_relevant`, `highest_rating`, `lowest_rating`.

## Reading the result

UI/API now surface three comparable rating views:

- `official_rating`: source-reported average over all available source reviews.
- `naive_rating`: average over the fetched sample only (bias check).
- `true_rating`: filtered and weighted rating after quality classification.

Also check:

- `official_review_count` vs `sample_size` to see sampling gap.
- `warning` and fallback method tags when LLM is unavailable.

### Current UI behavior

- Reviews list has two modes: `Compact` and `Detailed` (auto-default: compact for larger samples).
- `reviews_limit` is constrained both in UI and backend:
  - UI input enforces `max=SERPAPI_REVIEWS_LIMIT`
  - backend clamps incoming values to `1..SERPAPI_REVIEWS_LIMIT`
- Review `reason` is always English, while review `text` stays in its original language.

### Russian summary block (`Итог (RU)`)

When LLM mode is enabled (`USE_LLM=true` + `OPENAI_API_KEY`), results include:

- `summary_ru`: short Russian overview of contributing reviews.
- `pros_ru`: key positive points.
- `cons_ru`: key negative points.

Summary is built only from contributing classes (`valid`, `low_effort`) and skipped gracefully when LLM is disabled/unavailable.

### SerpApi metadata note

For some places SerpApi returns metadata in `local_results` instead of `place_results`.
The adapter now falls back to `local_results[0]` for:

- `official_rating`
- `official_review_count`

This reduces `n/a` cases for official source metrics.

## Evaluation

```powershell
$env:USE_LLM="false"
uv run python -m eval.run_eval
```

For LLM-mode eval set `USE_LLM=true` and `OPENAI_API_KEY`.

## Demo flow

1. Set SerpApi key and analyze a specific place query.
2. Compare official vs true, and review sample bias using naive.
3. Check injection / mismatch / speculative labels and `method`.
4. Inspect usage block + JSON logs (`trace_id`).

## Troubleshooting / FAQ

**`official_rating` shows `n/a`.**
The source didn't return aggregate metadata in the expected section. SerpApi sometimes returns it in `local_results` instead of `place_results`; the adapter falls back to `local_results[0]`, but for ambiguous queries even that can be missing. Fix: make the query more specific (name + city/address).

**`true_rating` is `n/a`.**
No contributing reviews (no `valid`/`low_effort`). This happens when the fetched sample is all star-only/spam/speculative, or the sample is tiny. Try a higher `reviews_limit` or a different `sort`.

**Why is `true_rating` lower/higher than `official_rating`?**
Because it removes noise. Inflated averages usually drop after filtering; genuinely good places with lots of lazy 5-star "good" reviews may stay similar (those count at half weight).

**Every review shows `method=fallback` and a warning about LLM.**
The LLM couldn't be used. Check `USE_LLM=true`, a valid `OPENAI_API_KEY`, model access, and network. Fallback = deterministic heuristic, so results are still produced, just less nuanced.

**`naive_rating` doesn't match Google's number.**
Expected. `naive_rating` is only the fetched sample (often positively biased by "most relevant" sorting); `official_rating` is over all reviews. The gap itself is a signal.

**Reviews are in Turkish/Chinese/etc.**
Review text is intentionally kept in its original language. Only the `reason` field is always English, and the `Итог (RU)` summary is Russian.

**`reviews_limit` is ignored above some number.**
It's clamped to `SERPAPI_REVIEWS_LIMIT` (backend) and bounded by `max` in the UI, to protect your SerpApi quota/spend.

**How much does one analysis cost?**
See the `usage` block / footer: it aggregates total tokens and estimated USD across all classification + summary calls for that request.
