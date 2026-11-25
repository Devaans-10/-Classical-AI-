"""Automated test suite for AID201A Classical AI Document QA System.

Tests:
  - Document parsing and structural triple extraction (>= 40 facts)
  - Multi-format parser support (Markdown, TXT, DOCX, PDF fallback)
  - Horn-clause inference engine (forward and backward chaining)
  - Deterministic Question Answering (direct, boolean, reverse, inference, abstention)
  - Reasoning trace and provenance citation generation
  - Evaluation suite and ablation metrics (35 gold test cases)
  - FastAPI backend endpoints
"""

import json
from pathlib import Path
import pytest

from kb.parser import parse_file, parse_text, Section
from kb.store import TripleStore
from kb.rules import load_rules_from_json, InferenceEngine
from kb.types import Triple, Fact, Provenance
from qa.answerer import Answerer
from eval.evaluator import run_ablation
from app.backend.main import (
    health, kb_stats, kb_facts, kb_rules, kb_sections,
    answer_question, QuestionRequest, run_evaluation
)

WORKSPACE = Path(__file__).resolve().parent.parent
HANDBOOK_PATH = WORKSPACE / "data" / "handbook.md"
RULES_PATH = WORKSPACE / "data" / "rules.json"
TEST_SET_PATH = WORKSPACE / "eval" / "gold_test_set.json"


# --------------------------------------------------------------------------- #
# 1. Document Parsing & Structure-Driven Fact Extraction
# --------------------------------------------------------------------------- #
def test_handbook_fact_count():
    facts = parse_file(HANDBOOK_PATH)
    # Assignment constraint: document with at least 40 extractable facts
    assert len(facts) >= 40, f"Expected at least 40 facts, got {len(facts)}"


def test_markdown_table_parsing():
    md = """# Section
| Component | Weight |
| Midterm | 30% |
| Final | 70% |
"""
    facts = parse_text(md)
    assert any(f.triple.subject == "final" and f.triple.predicate == "weight" and f.triple.object == "70%" for f in facts)


def test_clause_extraction():
    clause_text = """# Attendance
Clause 2.1.1: A student with attendance greater than or equal to 75 percent is eligible to appear for the end-semester examination.
Clause 2.1.2: A student with attendance below 60 percent is debarred from the examination.
"""
    facts = parse_text(clause_text)
    assert any(f.triple.predicate == "eligible_for_exam_if_attendance_ge" and f.triple.object == "75" for f in facts)
    assert any(f.triple.predicate == "debarred_if_attendance_lt" and f.triple.object == "60" for f in facts)


# --------------------------------------------------------------------------- #
# 2. Rule Base & Inference Engine
# --------------------------------------------------------------------------- #
def test_rules_loaded():
    rules = load_rules_from_json(RULES_PATH)
    # Assignment constraint: at least 5 inference rules
    assert len(rules) >= 5, f"Expected at least 5 rules, got {len(rules)}"
    rule_names = {r.name for r in rules}
    assert "exam_eligible_by_attendance" in rule_names
    assert "exam_debarred_low_attendance" in rule_names
    assert "condonation_fee_applicable" in rule_names
    assert "pass_by_marks" in rule_names
    assert "distinction_by_marks" in rule_names


def test_backward_chaining_eligibility():
    facts = parse_file(HANDBOOK_PATH)
    store = TripleStore()
    store.add_facts(facts)
    store.add_rules(load_rules_from_json(RULES_PATH))
    # Inject working fact
    store.add_fact(Fact(Triple("candidate", "attendance", "82"), Provenance("test")))

    engine = InferenceEngine(store)
    derived = engine.backward_chain(Triple("candidate", "exam_eligible", "yes"))
    assert derived is not None
    assert derived.rule.name == "exam_eligible_by_attendance"


def test_backward_chaining_debarred():
    facts = parse_file(HANDBOOK_PATH)
    store = TripleStore()
    store.add_facts(facts)
    store.add_rules(load_rules_from_json(RULES_PATH))
    # Inject working fact
    store.add_fact(Fact(Triple("candidate", "attendance", "50"), Provenance("test")))

    engine = InferenceEngine(store)
    derived = engine.backward_chain(Triple("candidate", "exam_debarred", "yes"))
    assert derived is not None
    assert derived.rule.name == "exam_debarred_low_attendance"


# --------------------------------------------------------------------------- #
# 3. Question Answering Templates & Reasoning Trace
# --------------------------------------------------------------------------- #
@pytest.fixture
def qa_engine():
    facts = parse_file(HANDBOOK_PATH)
    store = TripleStore()
    store.add_facts(facts)
    store.add_rules(load_rules_from_json(RULES_PATH))
    return Answerer(store)


def test_qa_direct_lookup(qa_engine):
    ans = qa_engine.answer("What is the course code?")
    assert ans.status == "answered"
    assert ans.answer == "AID201A"
    assert len(ans.citations) >= 1
    assert len(ans.trace) >= 1


def test_qa_boolean(qa_engine):
    ans = qa_engine.answer("Is the calculator allowed?")
    assert ans.status == "answered"
    assert ans.answer == "Yes"


def test_qa_reverse_lookup(qa_engine):
    ans = qa_engine.answer("Which component has weight 70%?")
    assert ans.status == "answered"
    assert "End-Semester Exam" in ans.answer


def test_qa_inference_eligible(qa_engine):
    ans = qa_engine.answer("Is a student with 80% attendance eligible?")
    assert ans.status == "answered"
    assert ans.answer == "Yes"
    # Verify trace steps include lookup -> rule fired -> derived fact -> answer
    trace_types = [t.type for t in ans.trace]
    assert "lookup" in trace_types
    assert "rule" in trace_types
    assert "derived" in trace_types
    assert "answer" in trace_types


def test_qa_inference_debarred(qa_engine):
    ans = qa_engine.answer("Is a student with 58% attendance debarred?")
    assert ans.status == "answered"
    assert ans.answer == "Yes"
    assert any("exam_debarred_low_attendance" in t.description for t in ans.trace)


def test_qa_inference_condonation(qa_engine):
    ans = qa_engine.answer("Is a student with 70% attendance eligible?")
    assert ans.status == "answered"
    assert "condonation fee of 500 INR" in ans.answer


def test_qa_inference_distinction(qa_engine):
    ans = qa_engine.answer("Is a student scoring 85 marks a distinction?")
    assert ans.status == "answered"
    assert ans.answer == "Yes"
    assert any("distinction_by_marks" in t.description for t in ans.trace)


def test_qa_abstention(qa_engine):
    # Hard constraint: return "Not in document" when KB cannot support answer. Never guess.
    ans = qa_engine.answer("What is the weather tomorrow?")
    assert ans.status == "not_in_document"
    assert ans.answer == "Not in document"
    assert ans.confidence == 0.0


# --------------------------------------------------------------------------- #
# 4. Evaluation Suite & Ablation Study
# --------------------------------------------------------------------------- #
def test_gold_test_set_requirements():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    test_cases = data.get("questions") or data.get("test_set", [])

    # Requirement: gold test set of at least 30 Q&A pairs
    assert len(test_cases) >= 30, f"Expected >= 30 test cases, got {len(test_cases)}"

    # Requirement: at least 10 requiring inference
    inference_cases = [c for c in test_cases if c.get("needs_inference") or c.get("requires_inference")]
    assert len(inference_cases) >= 10, f"Expected >= 10 inference cases, got {len(inference_cases)}"

    # Requirement: at least 5 unanswerable
    unanswerable_cases = [c for c in test_cases if (not c.get("answerable", True)) or (not c.get("is_answerable", True))]
    assert len(unanswerable_cases) >= 5, f"Expected >= 5 unanswerable cases, got {len(unanswerable_cases)}"


def test_eval_metrics_and_ablation():
    report = run_ablation(HANDBOOK_PATH, RULES_PATH, TEST_SET_PATH)

    with_rules = report["with_rules"]
    without_rules = report["without_rules"]
    delta = report["delta"]

    # Check metrics existence
    assert "accuracy" in with_rules
    assert "precision" in with_rules
    assert "recall" in with_rules
    assert "abstention_rate" in with_rules
    assert "avg_response_time_ms" in with_rules

    # Rules must significantly boost inference accuracy
    assert with_rules["accuracy"] >= without_rules["accuracy"]
    assert delta["inference_accuracy_with_rules"] > delta["inference_accuracy_without_rules"]
    assert delta["accuracy_gain"] > 0


# --------------------------------------------------------------------------- #
# 5. FastAPI Endpoints
# --------------------------------------------------------------------------- #
def test_api_endpoints():
    h = health()
    assert h["status"] == "healthy"

    stats = kb_stats()
    assert stats["facts"] >= 40
    assert stats["rules"] >= 5

    facts_res = kb_facts("")
    assert len(facts_res["facts"]) >= 40

    rules_res = kb_rules()
    assert len(rules_res["rules"]) >= 5

    sections_res = kb_sections()
    assert len(sections_res["sections"]) >= 5

    qa_res = answer_question(QuestionRequest(question="What is the pass mark?"))
    assert qa_res["status"] == "answered"
    assert qa_res["answer"] == "40%"

    eval_res = run_evaluation()
    assert eval_res["with_rules"]["accuracy"] == 1.0
