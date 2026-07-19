from __future__ import annotations

import httpx
import pytest

from core.config import Settings
from models.review import AnalysisResult, AnalyzeRequest, Usage
from web.app import app, get_analyzer, get_settings


class _DummySettings(Settings):
    def __init__(self) -> None:
        super().__init__()
        self.serpapi_reviews_limit = 25
        self.bayes_prior_strength = 5.0
        self.bayes_prior_mean = 3.5


class _DummyAnalyzer:
    def __init__(self) -> None:
        self.last_request: AnalyzeRequest | None = None

    async def analyze(self, request: AnalyzeRequest, trace_id: str) -> AnalysisResult:
        del trace_id
        self.last_request = request
        return AnalysisResult(
            source=request.source,
            source_id=request.source_id,
            place_name="Test Place",
            naive_rating=4.0,
            true_rating=4.1,
            delta=0.1,
            total=1,
            sample_size=1,
            official_rating=4.2,
            official_review_count=100,
            source_limit=request.reviews_limit,
            warning=None,
            excluded_count=0,
            excluded_by_class={},
            per_class_counts={
                "valid": 1,
                "empty": 0,
                "speculative": 0,
                "spam_offtopic": 0,
                "low_effort": 0,
                "uncertain": 0,
            },
            usage=Usage(),
            reviews=[],
        )


@pytest.mark.asyncio
async def test_analyze_ui_reviews_limit_bad_input_returns_ui_error() -> None:
    analyzer = _DummyAnalyzer()
    settings = _DummySettings()
    app.dependency_overrides[get_analyzer] = lambda: analyzer
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/analyze-ui",
                data={
                    "source": "serpapi",
                    "source_id": "coffee",
                    "sort": "newest",
                    "reviews_limit": "abc",
                },
            )
        assert response.status_code == 200
        assert "source_error: reviews_limit must be an integer" in response.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analyze_ui_reviews_limit_is_clamped_to_settings_cap() -> None:
    analyzer = _DummyAnalyzer()
    settings = _DummySettings()
    app.dependency_overrides[get_analyzer] = lambda: analyzer
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/analyze-ui",
                data={
                    "source": "serpapi",
                    "source_id": "coffee",
                    "sort": "newest",
                    "reviews_limit": "999",
                },
            )
        assert response.status_code == 200
        assert analyzer.last_request is not None
        assert analyzer.last_request.reviews_limit == 25
    finally:
        app.dependency_overrides.clear()
