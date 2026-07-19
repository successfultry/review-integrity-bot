from __future__ import annotations

import json
from pathlib import Path

from models.review import Review


class FixtureReviewSource:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    async def fetch(self, source_id: str) -> list[Review]:
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        out: list[Review] = []
        for idx, row in enumerate(payload):
            item = dict(row)
            item.setdefault("review_id", f"fixture-{idx + 1}")
            item.setdefault("source", "fixture")
            out.append(Review.model_validate(item))
        return out
