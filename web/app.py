from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.config import Settings, get_settings
from core.logging import new_trace_id, setup_logging
from models.review import AnalysisResult, AnalyzeRequest
from services.analyze import ReviewAnalyzer
from services.errors import SourceError

setup_logging()
app = FastAPI(title="Review Integrity Bot")
templates = Jinja2Templates(directory="web/templates")


def get_analyzer(settings: Settings = Depends(get_settings)) -> ReviewAnalyzer:
    return ReviewAnalyzer(settings)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "source": "google_maps", "source_id": "", "error": ""},
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
async def analyze_ui(request: Request, analyzer: ReviewAnalyzer = Depends(get_analyzer)) -> HTMLResponse:
    form = await request.form()
    source = str(form.get("source", "google_maps"))
    source_id = str(form.get("source_id", "")).strip()
    trace_id = new_trace_id()
    error = ""
    result = None
    try:
        if source not in {"google_maps", "serpapi"}:
            raise SourceError(f"unsupported source: {source}")
        if not source_id:
            raise SourceError("source_id is required")
        result = await analyzer.analyze(
            AnalyzeRequest(source=source, source_id=source_id),
            trace_id=trace_id,
        )
    except SourceError as exc:
        error = f"source_error: {exc.detail}"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result, "source": source, "source_id": source_id, "error": error},
    )
