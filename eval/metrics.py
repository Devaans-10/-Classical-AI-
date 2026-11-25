"""Evaluation metrics for the Document QA system.

AI technique: Measurement / experimental evaluation (no learning).

Why: We quantify how well the symbolic QA system reproduces the gold set.
Metrics are computed over deterministic answers (no model uncertainty), so
precision/recall/accuracy are exact counts. We also report abstention rate and
an ablation (with rules vs. without rules) to show the contribution of the
inference engine -- required by the assignment.
"""

from __future__ import annotations

from typing import Any


def normalize(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def answers_match(actual: str, expected: str) -> bool:
    """True when the actual answer supports the expected gold answer."""
    a = normalize(actual)
    e = normalize(expected)
    if a == e or e == "not in document":
        return a == e
    # Allow prefix/substring containment for elaborated answers (e.g. condonation).
    return e in a or a.startswith(e + " ") or a.startswith(e)


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy, precision, recall, abstention, per-category breakdown.

    Each result dict: {question, expected, answer, status, category, needs_inference, answerable, correct, response_time_ms}
    """
    total = len(results)
    answered = [r for r in results if r["status"] == "answered"]
    not_answered = [r for r in results if r["status"] != "answered"]
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    correct_answered = [r for r in answered if r.get("correct")]
    wrong_answered = [r for r in answered if not r.get("correct")]

    # Accuracy over answerable questions (the ones the KB *should* support).
    acc_denom = max(len(answerable), 1)
    accuracy = len(correct_answered) / acc_denom

    # Precision over questions the system attempted to answer.
    prec_denom = max(len(answered), 1)
    precision = len(correct_answered) / prec_denom

    # Recall = fraction of answerable questions answered correctly.
    recall = len(correct_answered) / acc_denom

    # F1 = harmonic mean of precision and recall.
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    abstention_rate = len(not_answered) / max(total, 1)

    # Abstention among answerable (failures to answer supported questions).
    answerable_not_answered = [r for r in not_answered if r["answerable"]]

    avg_response_time = (
        sum(r.get("response_time_ms", 0) for r in results) / max(total, 1)
    )

    # Per-category breakdown (answerable only).
    categories: dict[str, dict[str, int]] = {}
    for r in answerable:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "correct": 0, "answered": 0})
        categories[cat]["total"] += 1
        if r["status"] == "answered":
            categories[cat]["answered"] += 1
        if r.get("correct"):
            categories[cat]["correct"] += 1

    cat_summary = {}
    for cat, c in categories.items():
        cat_summary[cat] = {
            "total": c["total"],
            "answered": c["answered"],
            "correct": c["correct"],
            "accuracy": (c["correct"] / c["total"]) if c["total"] else 0.0,
        }

    # Inference subset.
    inference = [r for r in answerable if r["needs_inference"]]
    inf_correct = [r for r in inference if r.get("correct")]
    inference_summary = {
        "total": len(inference),
        "correct": len(inf_correct),
        "accuracy": (len(inf_correct) / len(inference)) if inference else 0.0,
    }

    # Confusion table (answerable).
    confusion = {
        "true_positive": len(correct_answered),
        "false_positive": len(wrong_answered),
        "false_negative": len(answerable_not_answered),
        "true_negative": len(
            [r for r in unanswerable if r["status"] != "answered"]
        ),
    }

    return {
        "total": total,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answered": len(answered),
        "not_answered": len(not_answered),
        "correct": len(correct_answered),
        "wrong": len(wrong_answered),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstention_rate": abstention_rate,
        "avg_response_time_ms": avg_response_time,
        "categories": cat_summary,
        "inference": inference_summary,
        "confusion": confusion,
    }
