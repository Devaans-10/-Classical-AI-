"""Ablation-based evaluation entry point used by the FastAPI backend.

AI technique: Experimental evaluation with controlled ablation.

Why: `run_ablation` runs the gold suite twice -- once with the rule engine
enabled and once with rules disabled -- and reports the delta. This isolates
the contribution of the symbolic inference rules (forward/backward chaining)
to accuracy, especially on inference questions. It is the backend contract for
the evaluation dashboard's ablation cards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kb.parser import parse_file
from kb.rules import load_rules_from_json
from kb.store import TripleStore
from qa.answerer import Answerer
from eval.metrics import compute_metrics, answers_match


def _build_store(doc_path: Path, with_rules: bool, rules_path: Path) -> TripleStore:
    facts = parse_file(str(doc_path))
    store = TripleStore()
    store.add_facts(facts)
    if with_rules and rules_path.exists():
        store.add_rules(load_rules_from_json(str(rules_path)))
    return store


def _run_side(doc_path: Path, rules_path: Path, test_set: list[dict], with_rules: bool) -> dict[str, Any]:
    store = _build_store(doc_path, with_rules, rules_path)
    answerer = Answerer(store, rules_path=str(rules_path) if with_rules else None)

    results: list[dict[str, Any]] = []
    for entry in test_set:
        t0 = time.perf_counter()
        ans = answerer.answer(entry["question"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        expected = entry["expected"]
        if ans.status == "answered":
            correct = answers_match(ans.answer, expected)
        else:
            correct = (not entry["answerable"]) and ans.status == "not_in_document"
        results.append({
            "id": entry["id"],
            "category": entry["category"],
            "question": entry["question"],
            "expected": expected,
            "actual": ans.answer,
            "status": ans.status,
            "correct": bool(correct),
            "needs_inference": bool(entry["needs_inference"]),
            "answerable": bool(entry["answerable"]),
            "response_time_ms": round(elapsed_ms, 2),
        })

    metrics = compute_metrics(results)
    return {
        "total": metrics["total"],
        "answerable": metrics["answerable"],
        "answered": metrics["answered"],
        "correct": metrics["correct"],
        "wrong": metrics["wrong"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "abstention_rate": metrics["abstention_rate"],
        "avg_response_time_ms": metrics["avg_response_time_ms"],
        "inference_accuracy": metrics["inference"]["accuracy"],
        "category_breakdown": metrics["categories"],
        "results": results,
        "with_rules": with_rules,
    }


def run_ablation(doc_path: Path | str, rules_path: Path | str, test_set_path: Path | str) -> dict[str, Any]:
    """Run the gold QA suite with and without rules; return the comparison report.

    Report shape expected by the frontend:
      { with_rules: {...}, without_rules: {...}, delta: {...} }
    """
    doc_path = Path(doc_path)
    rules_path = Path(rules_path)
    test_set_path = Path(test_set_path)

    with open(test_set_path, encoding="utf-8") as fh:
        test_set = json.load(fh)["questions"]

    with_rules = _run_side(doc_path, rules_path, test_set, with_rules=True)
    without_rules = _run_side(doc_path, rules_path, test_set, with_rules=False)

    delta = {
        "accuracy_gain": with_rules["accuracy"] - without_rules["accuracy"],
        "precision_gain": with_rules["precision"] - without_rules["precision"],
        "recall_gain": with_rules["recall"] - without_rules["recall"],
        "inference_accuracy_with_rules": with_rules["inference_accuracy"],
        "inference_accuracy_without_rules": without_rules["inference_accuracy"],
        "inference_accuracy_gain": with_rules["inference_accuracy"]
        - without_rules["inference_accuracy"],
        "abstention_rate_with_rules": with_rules["abstention_rate"],
        "abstention_rate_without_rules": without_rules["abstention_rate"],
    }

    return {"with_rules": with_rules, "without_rules": without_rules, "delta": delta}
