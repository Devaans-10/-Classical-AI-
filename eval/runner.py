"""Evaluation runner: executes the gold suite and reports metrics.

AI technique: Experimental evaluation over a symbolic QA system.

Why: The runner drives the answerer through every gold question, records the
produced answer + inference trace + response time, and compares against the
expected gold answer. It also runs an ablation (KB with rules vs. KB without
rules) to isolate the contribution of the rule-based inference engine -- direct
evidence for the CO-4 (evaluate performance) outcome.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from kb.parser import parse_file
from kb.store import TripleStore
from kb.rules import load_rules_from_json
from qa.answerer import Answerer
from eval.metrics import compute_metrics, answers_match

HERE = Path(__file__).resolve().parent
DOC_PATH = HERE.parent / "data" / "handbook.md"
RULES_PATH = HERE.parent / "data" / "rules.json"
GOLD_PATH = HERE / "gold_set.json"


def build_store(with_rules: bool = True) -> TripleStore:
    facts = parse_file(str(DOC_PATH))
    store = TripleStore()
    store.add_facts(facts)
    if with_rules:
        store.add_rules(load_rules_from_json(str(RULES_PATH)))
    return store


def run_suite(with_rules: bool = True) -> dict[str, Any]:
    """Run the gold suite once. Returns per-question results + metrics."""
    with open(GOLD_PATH, encoding="utf-8") as fh:
        gold = json.load(fh)["questions"]

    store = build_store(with_rules=with_rules)
    answerer = Answerer(store, rules_path=str(RULES_PATH) if with_rules else None)

    results: list[dict[str, Any]] = []
    for entry in gold:
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
            "question": entry["question"],
            "expected": expected,
            "answer": ans.answer,
            "status": ans.status,
            "confidence": ans.confidence,
            "category": entry["category"],
            "needs_inference": entry["needs_inference"],
            "answerable": entry["answerable"],
            "section": entry["section"],
            "correct": bool(correct),
            "response_time_ms": round(elapsed_ms, 2),
            "trace": [t.to_dict() for t in ans.trace],
            "citations": ans.citations,
            "template": ans.template,
        })

    metrics = compute_metrics(results)
    return {"results": results, "metrics": metrics, "with_rules": with_rules}


def export_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "id", "question", "expected", "answer", "status", "confidence",
            "category", "needs_inference", "answerable", "correct",
            "response_time_ms", "template",
        ])
        for r in results:
            w.writerow([
                r["id"], r["question"], r["expected"], r["answer"], r["status"],
                r["confidence"], r["category"], r["needs_inference"],
                r["answerable"], r["correct"], r["response_time_ms"],
                r["template"],
            ])
