from __future__ import annotations

import json
from pathlib import Path

from adapters.base import PlaceMeta, ReviewSort
from models.review import Review


class FixtureReviewSource:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    async def fetch(
        self,
        source_id: str,
        *,
        reviews_limit: int | None = None,
        sort: ReviewSort = "newest",
    ) -> tuple[list[Review], PlaceMeta]:
        del source_id, sort
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        out: list[Review] = []
        for idx, row in enumerate(payload):
            item = dict(row)
            item.setdefault("review_id", f"fixture-{idx + 1}")
            item.setdefault("source", "fixture")
            out.append(Review.model_validate(item))
        if reviews_limit:
            out = out[: max(1, reviews_limit)]
        return out, PlaceMeta(place_name="fixture", official_rating=None, official_review_count=len(out))
