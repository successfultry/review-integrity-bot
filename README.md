# Review Integrity Bot

Async FastAPI app that computes a true rating from noisy reviews.

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
