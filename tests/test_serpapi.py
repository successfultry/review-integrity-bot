from __future__ import annotations

from typing import Any

import pytest

from adapters.serpapi_reviews import SerpApiReviewSource


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses: list[dict[str, Any]], calls: list[dict[str, Any]], timeout: float) -> None:
        self._responses = responses
        self._calls = calls
        self._index = 0

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any]) -> _FakeResponse:
        self._calls.append({"url": url, "params": params})
        payload = self._responses[self._index]
        self._index += 1
        return _FakeResponse(payload)


@pytest.mark.asyncio
async def test_serpapi_pagination_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "local_results": [{"data_id": "abc123", "title": "Best Cafe"}],
        },
        {
            "reviews": [
                {"rating": 4.7, "snippet": "Great coffee", "user": {"name": "Alice"}},
                {"rating": 5, "extracted_snippet": "Loved it", "user": {"name": "Bob"}},
            ],
            "serpapi_pagination": {"next_page_token": "tok-2"},
        },
        {
            "reviews": [
                {"rating": 3.1, "snippet": "Okay place", "user": {"name": "Carol"}},
                {"rating": 2.0, "snippet": "Noisy", "user": {"name": "Dan"}},
            ],
        },
    ]
    calls: list[dict[str, Any]] = []

    def _client_factory(timeout: float) -> _FakeAsyncClient:
        return _FakeAsyncClient(responses=responses, calls=calls, timeout=timeout)

    monkeypatch.setattr("adapters.serpapi_reviews.httpx.AsyncClient", _client_factory)
    source = SerpApiReviewSource(api_key="k", reviews_limit=3)

    out = await source.fetch("best cafe nyc")

    assert len(out) == 3
    assert out[0].review_id == "serp-1"
    assert out[0].rating == 5
    assert out[0].text == "Great coffee"
    assert out[1].text == "Loved it"
    assert out[2].author == "Carol"
    assert out[0].source == "serpapi:Best Cafe"

    assert calls[0]["params"]["engine"] == "google_maps"
    assert calls[1]["params"]["engine"] == "google_maps_reviews"
    assert calls[2]["params"]["next_page_token"] == "tok-2"


@pytest.mark.asyncio
async def test_serpapi_no_data_id_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [{"local_results": [{"title": "No data id"}]}]
    calls: list[dict[str, Any]] = []

    def _client_factory(timeout: float) -> _FakeAsyncClient:
        return _FakeAsyncClient(responses=responses, calls=calls, timeout=timeout)

    monkeypatch.setattr("adapters.serpapi_reviews.httpx.AsyncClient", _client_factory)
    source = SerpApiReviewSource(api_key="k", reviews_limit=10)
    out = await source.fetch("unknown")
    assert out == []
    assert len(calls) == 1
