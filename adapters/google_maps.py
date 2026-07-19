from __future__ import annotations

import httpx

from models.review import Review

GOOGLE_REVIEW_LIMIT = 5


class GoogleMapsReviewSource:
    """Places API (New) source. Google returns at most 5 reviews per place."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.source_limit = GOOGLE_REVIEW_LIMIT

    async def fetch(self, source_id: str) -> list[Review]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
        query = source_id.strip()
        if not query:
            raise ValueError("source_id is required for google_maps")

        headers = {
            "X-Goog-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={**headers, "X-Goog-FieldMask": "places.id,places.displayName"},
                json={"textQuery": query},
            )
            search.raise_for_status()
            places = search.json().get("places", [])
            if not places:
                return []

            place_resource = str(places[0].get("id", "")).strip()
            if not place_resource:
                return []
            if not place_resource.startswith("places/"):
                place_resource = f"places/{place_resource}"

            details = await client.get(
                f"https://places.googleapis.com/v1/{place_resource}",
                headers={
                    **headers,
                    "X-Goog-FieldMask": "id,displayName,rating,userRatingCount,reviews",
                },
            )
            details.raise_for_status()
            details_data = details.json()
            reviews = details_data.get("reviews", []) or []
            display = details_data.get("displayName", {})
            place_name = display.get("text", query) if isinstance(display, dict) else query

        out: list[Review] = []
        for idx, item in enumerate(reviews[:GOOGLE_REVIEW_LIMIT], start=1):
            rating = int(item.get("rating", 1) or 1)
            text_obj = item.get("text", {})
            if isinstance(text_obj, dict):
                text = str(text_obj.get("text", ""))
            else:
                text = str(text_obj or "")
            author_obj = item.get("authorAttribution", {})
            author = str(author_obj.get("displayName", "")) if isinstance(author_obj, dict) else ""
            out.append(
                Review(
                    review_id=f"gmap-{idx}",
                    rating=max(1, min(5, rating)),
                    text=text,
                    author=author,
                    source=f"google_maps:{place_name}",
                )
            )
        return out
