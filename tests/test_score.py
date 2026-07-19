from models.review import ReviewClass
from services.score import score_reviews


def test_score_reviews_weighting_exclusions_and_bayes() -> None:
    ratings = [5, 5, 2, 1]
    labels = [ReviewClass.valid, ReviewClass.speculative, ReviewClass.low_effort, ReviewClass.spam_offtopic]
    confidences = [1.0, 0.9, 1.0, 0.9]
    out = score_reviews(
        ratings=ratings,
        labels=labels,
        confidences=confidences,
        valid_weight=1.0,
        low_effort_weight=0.5,
        confidence_cap=0.9,
        bayes_prior_strength=0.0,
        bayes_prior_mean=3.5,
    )

    # valid: 5 * 1.0 * 0.9 = 4.5, den 0.9
    # low_effort: 2 * 0.5 * 0.9 = 0.9, den 0.45
    # true = (4.5 + 0.9) / (0.9 + 0.45) = 5.4 / 1.35 = 4.0
    assert out["naive_rating"] == 3.25
    assert out["true_rating"] == 4.0
    assert out["delta"] == 0.75
    assert out["excluded_count"] == 2
    assert out["per_class_counts"]["valid"] == 1
    assert out["per_class_counts"]["speculative"] == 1


def test_score_none_when_no_included_reviews() -> None:
    out = score_reviews(
        ratings=[5, 1],
        labels=[ReviewClass.spam_offtopic, ReviewClass.speculative],
        confidences=[0.9, 0.9],
        valid_weight=1.0,
        low_effort_weight=0.5,
        confidence_cap=0.9,
        bayes_prior_strength=5.0,
        bayes_prior_mean=3.5,
    )
    assert out["true_rating"] is None
    assert out["delta"] is None
    assert out["excluded_count"] == 2


def test_score_bayesian_shrinkage() -> None:
    out = score_reviews(
        ratings=[5],
        labels=[ReviewClass.valid],
        confidences=[1.0],
        valid_weight=1.0,
        low_effort_weight=0.5,
        confidence_cap=0.9,
        bayes_prior_strength=5.0,
        bayes_prior_mean=3.5,
    )
    # (5*0.9 + 5*3.5) / (0.9 + 5) = (4.5 + 17.5) / 5.9 ≈ 3.729
    assert out["true_rating"] == 3.729
