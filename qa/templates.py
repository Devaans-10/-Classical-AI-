"""Question-template matcher.

AI technique: Deterministic template matching (no NLP, no LLM).

Why: Questions are mapped to KB query patterns by matching against a small,
authored set of regex templates. This is deterministic, explainable, and never
hallucinates. Each template captures the question intent (lookup, boolean,
comparison, count) and binds the entities/values that become a triple query or a
rule goal.

Template -> query pattern map:
  * "What is X's Y?"             -> lookup triple (X, Y, ?)
  * "What is the Y of X?"        -> lookup triple (X, Y, ?)
  * "Is X Y?"                    -> boolean lookup (X, Y, yes)
  * "Which X has Y Z?"           -> reverse lookup subjects with (predicate=Y, object=Z)
  * "When/How much/How many ...?" -> value lookup (subject, predicate, ?)
  * Inference: eligibility/pass/distinction from a supplied number -> backward chaining
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# A template: (name, compiled regex, kind)
# kind in {lookup, boolean, reverse, when_howmuch, inference_elig, inference_marks}
_TEMPLATE_LIST: list[tuple[str, re.Pattern, str]] = [
    # 1. Inference: Attendance & Eligibility / Debarment
    ("eligibility_inf", re.compile(
        r"^(?:is|can)\s+(?:a|the|an)?\s*(?P<who>student|person|candidate)?"
        r"\s*(?:with|scoring|getting|having)?\s*(?P<val>\d+(?:\.\d+)?)\s*%?\s*attendance\s*"
        r"(?:be\s+)?(?P<target>eligible|allowed|debarred)\b.*?\??$", re.IGNORECASE),
     "inference_elig"),

    # 2. Inference: Marks & Pass / Distinction / First Division
    ("pass_inf", re.compile(
        r"^(?:is|can)\s+(?:a|the|an)?\s*(?P<who>student|person|candidate)?"
        r"\s*(?:scoring|with|getting|having)\s*(?P<val>\d+(?:\.\d+)?)\s*(?:marks|%)?\s*"
        r"(?:a\s+)?(?:be\s+)?(?:a\s+)?(?P<target>pass|distinction|first division|fail)\b.*?\??$", re.IGNORECASE),
     "inference_marks"),

    # 3. Reverse lookup: "Which component has weight 70%?" / "Which X has Y Z?"
    ("reverse_has", re.compile(
        r"^(?:which|what)\s+(?P<rest>.+?)\s+has\s+(?P<predicate>[a-zA-Z0-9_\s]+?)\s+(?:equal to\s+|of\s+)?(?P<value>[a-zA-Z0-9_%-]+)\??$",
        re.IGNORECASE),
     "reverse"),

    # 4. "What is X's Y?"
    ("what_is_possessive", re.compile(
        r"^(?:what|where|when)\s+is\s+(?:the\s+)?(?P<subject>.+?)['’]\s*s\s+(?P<predicate>.+?)\??$", re.IGNORECASE),
     "lookup"),

    # 5. "What is the Y of X?"
    ("what_is_of", re.compile(
        r"^(?:what|where|when)\s+is\s+the\s+(?P<predicate>.+?)\s+of\s+(?:the\s+)?(?P<subject>.+?)\??$", re.IGNORECASE),
     "lookup"),

    # 6. "How many credits is the course?" / "How much is X?"
    ("how_many_credits", re.compile(
        r"^how\s+many\s+(?P<predicate>credits)\s+(?:is|in|for|has)\s+(?:the\s+)?(?P<subject>.+?)\??$", re.IGNORECASE),
     "lookup"),
    ("how_much_is", re.compile(
        r"^how\s+(?:much|many)\s+(?:is\s+)?(?:the\s+)?(?P<predicate>.+?)\s+(?:is|for|of)\s+(?:the\s+)?(?P<subject>.+?)\??$",
        re.IGNORECASE),
     "when_howmuch"),

    # 7. "When is X?"
    ("when_is", re.compile(
        r"^when\s+is\s+(?:the\s+)?(?P<predicate>.+?)\??$", re.IGNORECASE),
     "lookup"),

    # 8. "Where is X?" / "Who is X?"
    ("where_who_is", re.compile(
        r"^(?:where|who)\s+is\s+(?:the\s+)?(?P<predicate>.+?)\??$", re.IGNORECASE),
     "lookup"),

    # 9. Boolean queries: "Is the calculator allowed?" / "Is reference books issuable?"
    ("is_boolean", re.compile(
        r"^is\s+(?:the\s+)?(?P<subject>.+?)\s+(?P<predicate>allowed|permitted|issuable|closed|eligible|open)\??$", re.IGNORECASE),
     "boolean"),
    ("is_general_boolean", re.compile(
        r"^is\s+(?:the\s+)?(?P<subject>.+?)\s+(?P<predicate>.+?)\??$", re.IGNORECASE),
     "boolean"),

    # 10. Direct lookup: "What is the pass mark?" / "What is the course code?"
    ("what_is_direct", re.compile(
        r"^what\s+is\s+(?:the\s+)?(?P<predicate>.+?)\??$", re.IGNORECASE),
     "lookup"),
]


@dataclass
class QuestionPlan:
    """Result of matching a question to a template."""

    template: str
    kind: str
    groups: dict[str, str]
    raw: str


def match_question(question: str) -> Optional[QuestionPlan]:
    """Match `question` against the authored templates (first match wins).

    Returns None when no template matches -- the caller should then answer
    "Not in document" (the KB cannot support the question).
    """
    q = question.strip()
    for name, pattern, kind in _TEMPLATE_LIST:
        m = pattern.match(q)
        if m:
            return QuestionPlan(template=name, kind=kind, groups=m.groupdict(), raw=q)
    return None


def template_names() -> list[str]:
    return [t[0] for t in _TEMPLATE_LIST]
