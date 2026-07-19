from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Protocol

from models.review import Review

ReviewSort = Literal["most_relevant", "newest", "highest_rating", "lowest_rating"]


@dataclass(slots=True)
class PlaceMeta:
    place_name: str
    official_rating: float | None = None
    official_review_count: int | None = None


class ReviewSource(Protocol):
    async def fetch(
        self,
        source_id: str,
        *,
        reviews_limit: int | None = None,
        sort: ReviewSort = "newest",
    ) -> tuple[list[Review], PlaceMeta]:
        raise NotImplementedError
