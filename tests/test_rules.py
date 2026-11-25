"""Tests for the rules engine (forward and backward chaining).

AI technique under test: Rule-based inference (Horn clauses, forward chaining +
backward chaining) over the triple KB. Numeric comparisons (cmp:ge/cmp:lt) are
supported.
"""

from kb.parser import parse_text
from kb.store import TripleStore
from kb.rules import InferenceEngine, load_rules_from_json, Rule
from kb.types import Fact, Provenance, Triple


def _store_with_rules():
    facts = parse_text(
        "Attendance Policy\n"
        "Minimum attendance required: 75%\n"
        "Clause 2.1.1: A student with attendance greater than or equal to 75 percent is eligible to appear for the end-semester examination.\n"
        "Clause 2.1.2: attendance below 60 percent is debarred.\n"
    )
    store = TripleStore()
    store.add_facts(facts)
    store.add_rules(load_rules_from_json("data/rules.json"))
    return store


def test_forward_chain_derives_eligibility():
    store = _store_with_rules()
    store.add_fact(Fact(Triple("alice", "attendance", "80"), Provenance("q")))
    eng = InferenceEngine(store)
    derived = eng.forward_chain()
    assert any(
        d.triple == Triple("alice", "exam_eligible", "yes")
        and d.rule.name == "exam_eligible_by_attendance"
        for d in derived
    )


def test_backward_chain_proves_debarred():
    store = _store_with_rules()
    store.add_fact(Fact(Triple("bob", "attendance", "50"), Provenance("q")))
    eng = InferenceEngine(store)
    d = eng.backward_chain(Triple("bob", "exam_debarred", "yes"))
    assert d is not None
    assert d.rule is not None
    assert d.rule.name == "exam_debarred_low_attendance"


def test_backward_chain_fails_when_unsupported():
    store = _store_with_rules()
    # No attendance fact for carol, and 50 is not < 60 anyway; nothing proves eligible.
    eng = InferenceEngine(store)
    d = eng.backward_chain(Triple("carol", "exam_eligible", "yes"))
    assert d is None


def test_rule_variables_and_enabled():
    rules = load_rules_from_json("data/rules.json")
    assert len(rules) >= 5
    assert all(r.enabled for r in rules)
    names = {r.name for r in rules}
    assert {"exam_eligible_by_attendance", "pass_by_marks", "distinction_by_marks"} <= names
