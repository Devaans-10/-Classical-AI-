"""Tests for the question-answering layer (at least 10 QA cases).

AI technique under test: Template-based question parsing + forward/backward
chaining + trace generation. Every answer is backed by a fact or a rule; the
system abstains ("Not in document") when the KB cannot support a question.
"""

from kb.parser import parse_file
from kb.store import TripleStore
from kb.rules import load_rules_from_json
from qa.answerer import Answerer


def _answerer():
    facts = parse_file("data/handbook.md")
    store = TripleStore()
    store.add_facts(facts)
    store.add_rules(load_rules_from_json("data/rules.json"))
    return Answerer(store, rules_path="data/rules.json")


CASES = [
    ("What is the course code?", "AID201A", True),
    ("What is the pass mark?", "40%", True),
    ("Is the calculator allowed?", "Yes", True),
    ("Is the mobile phone allowed?", "No", True),
    ("Is a student with 80% attendance eligible?", "Yes", True),
    ("Is a student with 58% attendance debarred?", "Yes", True),
    ("Is a student scoring 85 marks a distinction?", "Yes", True),
    ("Is a student scoring 35 marks a pass?", "No", True),
    ("Is a student scoring 60 marks a first division?", "Yes", True),
    ("What is the hostel curfew time?", "21:00", True),
    ("What is the weather tomorrow?", "Not in document", False),
    ("What is the instructor's salary?", "Not in document", False),
]


def test_qa_cases():
    a = _answerer()
    for question, expected, answerable in CASES:
        r = a.answer(question)
        if answerable:
            assert r.status == "answered", f"expected answered for {question!r}, got {r.status}"
            assert expected.lower() in r.answer.lower(), (
                f"expected {expected!r} in {r.answer!r} for {question!r}"
            )
        else:
            assert r.status == "not_in_document", f"expected abstention for {question!r}"
            assert r.confidence == 0.0


def test_every_answered_has_trace_and_citation():
    a = _answerer()
    r = a.answer("Is a student with 80% attendance eligible?")
    assert r.status == "answered"
    assert len(r.trace) >= 2
    assert any(t.type == "rule" for t in r.trace)
    assert len(r.citations) >= 1
