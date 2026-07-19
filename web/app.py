from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.config import Settings, get_settings
from core.logging import get_logger, new_trace_id, setup_logging
from models.review import AnalysisResult, AnalyzeRequest
from services.analyze import ReviewAnalyzer
from services.errors import SourceError

setup_logging()
LOG = get_logger(__name__)
app = FastAPI(title="Review Integrity Bot")
templates = Jinja2Templates(directory="web/templates")


def get_analyzer(settings: Settings = Depends(get_settings)) -> ReviewAnalyzer:
    return ReviewAnalyzer(settings)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "source": settings.default_source if settings.default_source in {"google_maps", "serpapi"} else "serpapi",
            "source_id": "",
            "sort": "newest",
            "reviews_limit": 50,
            "serpapi_reviews_limit": max(1, settings.serpapi_reviews_limit),
            "error": "",
            "fallback_count": 0,
            "bayes_prior_strength": settings.bayes_prior_strength,
            "bayes_prior_mean": settings.bayes_prior_mean,
        },
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: AnalyzeRequest, analyzer: ReviewAnalyzer = Depends(get_analyzer)) -> AnalysisResult:
    trace_id = new_trace_id()
    try:
        return await analyzer.analyze(request=request, trace_id=trace_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "source_error", "detail": exc.detail, "trace_id": trace_id},
        ) from exc


@app.post("/analyze-ui", response_class=HTMLResponse)
async def analyze_ui(
    request: Request,
    analyzer: ReviewAnalyzer = Depends(get_analyzer),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    form = await request.form()
    source = str(form.get("source", "serpapi"))
    source_id = str(form.get("source_id", "")).strip()
    sort = str(form.get("sort", "newest")).strip() or "newest"
    reviews_limit_raw = str(form.get("reviews_limit", "50")).strip()
    trace_id = new_trace_id()
    error = ""
    result = None
    fallback_count = 0
    max_limit = max(1, settings.serpapi_reviews_limit)
    try:
        reviews_limit = 50
        if reviews_limit_raw:
            try:
                reviews_limit = int(reviews_limit_raw)
            except ValueError as exc:
                raise SourceError("reviews_limit must be an integer") from exc
        reviews_limit = min(max(1, reviews_limit), max_limit)
        if source not in {"google_maps", "serpapi"}:
            raise SourceError(f"unsupported source: {source}")
        if sort not in {"most_relevant", "newest", "highest_rating", "lowest_rating"}:
            raise SourceError(f"unsupported sort: {sort}")
        if not source_id:
            raise SourceError("source_id is required")
        result = await analyzer.analyze(
            AnalyzeRequest(source=source, source_id=source_id, sort=sort, reviews_limit=reviews_limit),
            trace_id=trace_id,
        )
        fallback_count = sum(1 for item in result.reviews if item.method.value == "fallback")
    except SourceError as exc:
        error = f"source_error: {exc.detail}"
    except Exception:  # noqa: BLE001
        LOG.exception(
            "analyze_ui_unexpected_error",
            extra={"extra_payload": {"trace_id": trace_id, "source": source, "source_id": source_id}},
        )
        error = f"unexpected_error: internal error, trace_id={trace_id}"
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "source": source,
            "source_id": source_id,
            "sort": sort,
            "reviews_limit": reviews_limit,
            "serpapi_reviews_limit": max_limit,
            "error": error,
            "fallback_count": fallback_count,
            "bayes_prior_strength": settings.bayes_prior_strength,
            "bayes_prior_mean": settings.bayes_prior_mean,
        },
    )
