from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        self.classifier_confidence_threshold = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.70"))
        self.valid_review_weight = float(os.getenv("VALID_REVIEW_WEIGHT", "1.0"))
        self.low_effort_review_weight = float(os.getenv("LOW_EFFORT_REVIEW_WEIGHT", "0.5"))
        self.confidence_weight_cap = float(os.getenv("CONFIDENCE_WEIGHT_CAP", "0.90"))
        self.bayes_prior_strength = float(os.getenv("BAYES_PRIOR_STRENGTH", "5.0"))
        self.bayes_prior_mean = float(os.getenv("BAYES_PRIOR_MEAN", "3.5"))
        self.use_llm = os.getenv("USE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        self.serpapi_api_key = os.getenv("SERPAPI_KEY", "").strip()
        self.serpapi_reviews_limit = int(os.getenv("SERPAPI_REVIEWS_LIMIT", "200"))
        self.default_source = os.getenv("DEFAULT_SOURCE", "google_maps").strip()
        self.input_price_per_1m = float(os.getenv("INPUT_PRICE_PER_1M", "0.15"))
        self.output_price_per_1m = float(os.getenv("OUTPUT_PRICE_PER_1M", "0.60"))
        self.classify_max_retries = int(os.getenv("CLASSIFY_MAX_RETRIES", "3"))
        self.classify_max_concurrency = int(os.getenv("CLASSIFY_MAX_CONCURRENCY", "5"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
