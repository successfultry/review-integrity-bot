from __future__ import annotations

from adapters.base import PlaceMeta
from adapters.google_maps import GOOGLE_REVIEW_LIMIT, GoogleMapsReviewSource
from adapters.serpapi_reviews import SERPAPI_DEFAULT_REVIEWS_LIMIT, SerpApiReviewSource
from core.config import Settings
from core.logging import get_logger
from models.review import AnalysisResult, AnalyzeRequest, AnalyzedReview
from services.classify import ReviewClassifier
from services.errors import SourceError
from services.score import score_reviews

LOG = get_logger(__name__)


class ReviewAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classifier = ReviewClassifier(settings)
        self.google_source = GoogleMapsReviewSource(settings.google_maps_api_key)
        self.serpapi_source = SerpApiReviewSource(
            api_key=settings.serpapi_api_key,
            reviews_limit=settings.serpapi_reviews_limit or SERPAPI_DEFAULT_REVIEWS_LIMIT,
        )

    async def analyze(self, request: AnalyzeRequest, trace_id: str) -> AnalysisResult:
        source_id = request.source_id.strip()
        if not source_id:
            raise SourceError("source_id is required")

        source = request.source
        requested_limit = request.reviews_limit
        requested_sort = request.sort
        source_limit: int | None = None
        warning: str | None = None
        if source == "serpapi":
            if not self.settings.serpapi_api_key:
                raise SourceError("SERPAPI_KEY is not set")
            source_limit = requested_limit or self.settings.serpapi_reviews_limit
        else:
            source_limit = min(GOOGLE_REVIEW_LIMIT, requested_limit) if requested_limit else GOOGLE_REVIEW_LIMIT

        try:
            if source == "serpapi":
                reviews, meta = await self.serpapi_source.fetch(
                    source_id,
                    reviews_limit=source_limit,
                    sort=requested_sort,
                )
            else:
                reviews, meta = await self.google_source.fetch(
                    source_id,
                    reviews_limit=source_limit,
                    sort=requested_sort,
                )
        except SourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                f"{source}_failed",
                extra={"extra_payload": {"trace_id": trace_id, "source": source, "error": str(exc)}},
            )
            raise SourceError(f"{source} fetch failed: {exc}") from exc

        if not isinstance(meta, PlaceMeta):
            meta = PlaceMeta(place_name=source_id)
        if not reviews:
            raise SourceError(f"no reviews found for source={source!r} source_id={source_id!r}")

        classifications, usage = await self.classifier.classify_reviews(reviews, trace_id=trace_id)
        score = score_reviews(
            ratings=[r.rating for r in reviews],
            labels=[c.label for c in classifications],
            confidences=[c.confidence for c in classifications],
            valid_weight=self.settings.valid_review_weight,
            low_effort_weight=self.settings.low_effort_review_weight,
            confidence_cap=self.settings.confidence_weight_cap,
            bayes_prior_strength=self.settings.bayes_prior_strength,
            bayes_prior_mean=self.settings.bayes_prior_mean,
        )

        analyzed_reviews = [
            AnalyzedReview(
                review_id=r.review_id,
                rating=r.rating,
                text=r.text,
                author=r.author,
                label=c.label,
                reason=c.reason,
                confidence=c.confidence,
                method=c.method,
            )
            for r, c in zip(reviews, classifications)
        ]

        sample_size = len(reviews)
        if source == "google_maps" and sample_size >= GOOGLE_REVIEW_LIMIT:
            warning = (
                f"Google Places returns at most {GOOGLE_REVIEW_LIMIT} reviews; "
                "true_rating is Bayesian-shrunk for small samples."
            )

        true_rating = score["true_rating"]
        delta = score["delta"]
        return AnalysisResult(
            source=source,
            source_id=source_id,
            naive_rating=float(score["naive_rating"]),
            true_rating=float(true_rating) if true_rating is not None else None,
            delta=float(delta) if delta is not None else None,
            total=int(score["total"]),
            sample_size=sample_size,
            official_rating=meta.official_rating,
            official_review_count=meta.official_review_count,
            source_limit=source_limit,
            warning=warning,
            excluded_count=int(score["excluded_count"]),
            excluded_by_class=dict(score["excluded_by_class"]),
            per_class_counts=dict(score["per_class_counts"]),
            usage=usage,
            reviews=analyzed_reviews,
        )
