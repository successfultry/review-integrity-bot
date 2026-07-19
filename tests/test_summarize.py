from __future__ import annotations

import pytest

from core.config import Settings
from models.review import Review
from services.summarize import ReviewSummarizer


class DummySettings(Settings):
    def __init__(self) -> None:
        super().__init__()
        self.use_llm = False
        self.openai_api_key = ""


@pytest.mark.asyncio
async def test_summarize_skips_without_llm() -> None:
    settings = DummySettings()
    summarizer = ReviewSummarizer(settings)
    summary_ru, pros_ru, cons_ru, usage = await summarizer.summarize(
        trace_id="t",
        place_name="Place",
        reviews=[Review(review_id="1", rating=5, text="great", author="a", source="x")],
    )
    assert summary_ru is None
    assert pros_ru == []
    assert cons_ru == []
    assert usage.total_tokens == 0
