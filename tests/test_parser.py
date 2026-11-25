"""Tests for the document parser (KB construction).

AI technique under test: Knowledge Representation via structure-driven extraction.
The parser turns document structure (headings, tables, key:value lines, numbered
clauses) into triples with provenance -- no NLP is involved.
"""

from kb.parser import parse_text, parse_file


def test_extracts_key_value_facts():
    text = "Course code: AID201A\nCredits: 4\n"
    facts = parse_text(text)
    kv = [f for f in facts if f.triple.predicate == "course code"]
    assert any(f.triple.object == "AID201A" for f in kv)


def test_extracts_table_facts():
    text = "| Component | Weight | Marks |\n|---|---|---|\n| Exam | 70% | 70 |\n"
    facts = parse_text(text)
    exam = [f for f in facts if f.triple.subject == "exam" and f.triple.predicate == "weight"]
    assert any(f.triple.object == "70%" for f in exam)


def test_extracts_clause_facts():
    text = "Clause 2.1.1: A student with attendance greater than or equal to 75 percent is eligible to appear for the end-semester examination.\n"
    facts = parse_text(text)
    elig = [f for f in facts if f.triple.predicate == "eligible_for_exam_if_attendance_ge"]
    assert any(f.triple.object == "75" for f in elig)
    # Provenance carries the clause id.
    assert any(f.provenance.clause == "Clause 2.1.1" for f in facts)


def test_facts_have_provenance():
    facts = parse_text("Section 1\nCourse code: AID201A\n")
    assert all(f.provenance.section for f in facts)
    assert all(0.0 <= f.confidence <= 1.0 for f in facts)


def test_at_least_40_facts_from_handbook():
    facts = parse_file("data/handbook.md")
    assert len(facts) >= 40, f"expected >=40 facts, got {len(facts)}"
