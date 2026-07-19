from __future__ import annotations

import httpx

from models.review import Review

SERPAPI_DEFAULT_REVIEWS_LIMIT = 200


class SerpApiReviewSource:
    def __init__(self, api_key: str, reviews_limit: int) -> None:
        self.api_key = api_key
        self.reviews_limit = max(1, reviews_limit)

    async def fetch(self, source_id: str) -> list[Review]:
        query = source_id.strip()
        if not query:
            raise ValueError("source_id is required for serpapi")

        async with httpx.AsyncClient(timeout=20.0) as client:
            search_resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google_maps",
                    "q": query,
                    "api_key": self.api_key,
                },
            )
            search_resp.raise_for_status()
            search_payload = search_resp.json()

            data_id = self._extract_data_id(search_payload)
            if not data_id:
                return []
            place_name = self._extract_place_name(search_payload, query)

            out: list[Review] = []
            next_page_token: str | None = None
            page_index = 0
            while len(out) < self.reviews_limit:
                params: dict[str, str] = {
                    "engine": "google_maps_reviews",
                    "data_id": data_id,
                    "api_key": self.api_key,
                }
                if next_page_token:
                    params["next_page_token"] = next_page_token

                reviews_resp = await client.get("https://serpapi.com/search", params=params)
                reviews_resp.raise_for_status()
                reviews_payload = reviews_resp.json()
                reviews = reviews_payload.get("reviews", [])

                for item in reviews:
                    page_index += 1
                    rating = int(round(float(item.get("rating", 1) or 1)))
                    snippet = str(item.get("snippet") or item.get("extracted_snippet") or "")
                    user = item.get("user", {})
                    author = str(user.get("name", "")) if isinstance(user, dict) else ""
                    out.append(
                        Review(
                            review_id=f"serp-{page_index}",
                            rating=max(1, min(5, rating)),
                            text=snippet,
                            author=author,
                            source=f"serpapi:{place_name}",
                        )
                    )
                    if len(out) >= self.reviews_limit:
                        break

                pagination = reviews_payload.get("serpapi_pagination", {})
                next_page_token = pagination.get("next_page_token") if isinstance(pagination, dict) else None
                if not next_page_token:
                    break

            return out

    def _extract_data_id(self, payload: dict) -> str:
        place_results = payload.get("place_results", {})
        if isinstance(place_results, dict):
            data_id = str(place_results.get("data_id", "")).strip()
            if data_id:
                return data_id

        local_results = payload.get("local_results", [])
        if isinstance(local_results, list) and local_results:
            first = local_results[0]
            if isinstance(first, dict):
                return str(first.get("data_id", "")).strip()
        return ""

    def _extract_place_name(self, payload: dict, default: str) -> str:
        place_results = payload.get("place_results", {})
        if isinstance(place_results, dict):
            title = str(place_results.get("title", "")).strip()
            if title:
                return title
        local_results = payload.get("local_results", [])
        if isinstance(local_results, list) and local_results:
            first = local_results[0]
            if isinstance(first, dict):
                title = str(first.get("title", "")).strip()
                if title:
                    return title
        return default
