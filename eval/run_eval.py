from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.config import get_settings
from models.review import Review, ReviewClass
from services.classify import ReviewClassifier


def _build_confusion(predicted: list[ReviewClass], expected: list[ReviewClass]) -> dict[str, dict[str, int]]:
    classes = [c.value for c in ReviewClass]
    matrix = {e: {p: 0 for p in classes} for e in classes}
    for exp, pred in zip(expected, predicted):
        matrix[exp.value][pred.value] += 1
    return matrix


def _precision_recall(expected: list[ReviewClass], predicted: list[ReviewClass], target: ReviewClass) -> tuple[float, float]:
    tp = sum(1 for e, p in zip(expected, predicted) if e == target and p == target)
    fp = sum(1 for e, p in zip(expected, predicted) if e != target and p == target)
    fn = sum(1 for e, p in zip(expected, predicted) if e == target and p != target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def main() -> None:
    settings = get_settings()
    path = Path(__file__).resolve().parent / "golden.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    reviews = [
        Review(
            review_id=f"gold-{i + 1}",
            rating=int(r["rating"]),
            text=str(r["text"]),
            author="eval",
            source="golden",
        )
        for i, r in enumerate(rows)
    ]
    expected = [ReviewClass(str(r["expected_label"])) for r in rows]

    mode = "llm" if settings.use_llm and settings.openai_api_key else "heuristic"
    print(f"mode: {mode} (USE_LLM={settings.use_llm})")

    classifier = ReviewClassifier(settings)
    predicted, _usage = await classifier.classify_reviews(reviews, trace_id="eval-trace")
    pred_labels = [p.label for p in predicted]

    correct = sum(1 for e, p in zip(expected, pred_labels) if e == p)
    accuracy = correct / len(expected) if expected else 0.0
    print(f"accuracy: {accuracy:.3f} ({correct}/{len(expected)})")

    f1s: list[float] = []
    print("\nper-class precision/recall/f1:")
    for klass in ReviewClass:
        precision, recall = _precision_recall(expected, pred_labels, klass)
        f1 = _f1(precision, recall)
        if any(e == klass for e in expected) or any(p == klass for p in pred_labels):
            f1s.append(f1)
        print(f"- {klass.value}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"\nmacro-F1 (present classes): {macro_f1:.3f}")

    matrix = _build_confusion(pred_labels, expected)
    print("\nconfusion matrix (expected -> predicted):")
    for exp, cols in matrix.items():
        print(f"{exp}: {cols}")


if __name__ == "__main__":
    asyncio.run(main())
