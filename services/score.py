from __future__ import annotations

from models.review import ReviewClass

_INCLUDED = {ReviewClass.valid, ReviewClass.low_effort}


def score_reviews(
    ratings: list[int],
    labels: list[ReviewClass],
    confidences: list[float],
    valid_weight: float,
    low_effort_weight: float,
    confidence_cap: float,
    bayes_prior_strength: float,
    bayes_prior_mean: float,
) -> dict[str, float | int | None | dict[str, int]]:
    total = len(ratings)
    naive = round(sum(ratings) / total, 3) if total else 0.0

    per_class: dict[str, int] = {k.value: 0 for k in ReviewClass}
    excluded_by_class: dict[str, int] = {}
    weighted_sum = 0.0
    weighted_den = 0.0
    excluded_count = 0
    cap = max(0.0, min(1.0, confidence_cap))

    for rating, label, confidence in zip(ratings, labels, confidences):
        per_class[label.value] += 1
        if label not in _INCLUDED:
            excluded_count += 1
            excluded_by_class[label.value] = excluded_by_class.get(label.value, 0) + 1
            continue

        base = valid_weight if label == ReviewClass.valid else low_effort_weight
        weight = base * min(max(confidence, 0.0), cap)
        weighted_sum += rating * weight
        weighted_den += weight

    if weighted_den <= 0:
        true_rating: float | None = None
        delta: float | None = None
    else:
        c = max(0.0, bayes_prior_strength)
        m = bayes_prior_mean
        true_rating = round((weighted_sum + c * m) / (weighted_den + c), 3)
        delta = round(true_rating - naive, 3)

    return {
        "naive_rating": naive,
        "true_rating": true_rating,
        "delta": delta,
        "total": total,
        "excluded_count": excluded_count,
        "excluded_by_class": excluded_by_class,
        "per_class_counts": per_class,
    }
