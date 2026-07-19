from __future__ import annotations

from typing import Protocol

from models.review import Review


class ReviewSource(Protocol):
    async def fetch(self, source_id: str) -> list[Review]:
        raise NotImplementedError
