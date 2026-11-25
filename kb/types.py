"""Knowledge base schema definitions.

AI technique: Knowledge Representation (KR).

Why KR: The system must answer questions using *only* facts in the document,
never by language understanding or a learned model. We therefore commit to an
explicit, symbolic representation: every extractable fact is a triple
(subject, predicate, object) annotated with provenance (where it came from in
the document). Derived knowledge is produced by rules over these triples.
This makes the reasoning traceable, auditable, and free of hallucination -- a
requirement of the assignment.

Schema:
  Triple  = (subject, predicate, object)            # atomic unit of knowledge
  Fact    = Triple + provenance + confidence         # a stored triple
  Rule    = head + body(conditions) + name + provenance   # Horn-style clause
  KB      = collection of facts + rules, indexed for search

Provenance is a dict with keys: section (str), clause (str|None), page (int|None).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _norm_subject(value: str) -> str:
    """Canonical subject normalization: lowercase, strip articles and leading
    digits/dots so '2. attendance policy' and 'the attendance policy' both map
    to 'attendance policy'. Used for subject indexing and lookup only (never for
    object values, which must be preserved verbatim)."""
    t = _clean_text(value).lower()
    t = re.sub(r"^[\d\.\s]+", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t.strip()


def _norm_predicate(value: str) -> str:
    """Canonical predicate normalization: lowercase, collapse whitespace,
    convert underscores to spaces, and strip leading articles."""
    t = _clean_text(value).lower().replace("_", " ")
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t.strip()


@dataclass(frozen=True)
class Provenance:
    """Where a fact or rule originated in the source document.

    section: heading under which the fact was found (e.g. "2. Attendance Policy").
    clause:  numbered clause id when available (e.g. "2.1.3"), else None.
    page:    page number when the source is paginated (PDF/DOCX), else None.
    """

    section: str
    clause: str | None = None
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "clause": self.clause, "page": self.page}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        return cls(section=str(d.get("section", "")),
                   clause=d.get("clause"),
                   page=d.get("page"))

    def short(self) -> str:
        parts = [self.section]
        if self.clause:
            parts.append(self.clause)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " / ".join(parts)


@dataclass(frozen=True)
class Triple:
    """Atomic unit of knowledge: (subject, predicate, object).

    Entities are normalized to lowercase, whitespace-collapsed strings so that
    lookups are deterministic and case-insensitive. Values are kept as-is so
    units and formatting are preserved for citation.
    """

    subject: str
    predicate: str
    object: str

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Triple":
        return cls(subject=str(d["subject"]), predicate=str(d["predicate"]),
                   object=str(d["object"]))

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


@dataclass
class Fact:
    """A stored triple with provenance and confidence.

    confidence: deterministic score in [0, 1]. Facts extracted from structure
    (tables, key: value lines, numbered clauses) receive confidence 1.0.
    """

    triple: Triple
    provenance: Provenance
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"triple": self.triple.to_dict(),
                "provenance": self.provenance.to_dict(),
                "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        return cls(triple=Triple.from_dict(d["triple"]),
                   provenance=Provenance.from_dict(d["provenance"]),
                   confidence=float(d.get("confidence", 1.0)))

    def __str__(self) -> str:
        return f"{self.triple} [conf={self.confidence}]"


@dataclass
class Rule:
    """A Horn-style inference rule: head :- body.

    head:     the consequent Triple (may contain variables prefixed with '?').
    body:     list of condition Triples (may contain variables).
    name:     human-readable rule name.
    provenance: where the rule is authored (kept in the editable rules file).

    Variables are matched against KB facts during forward/backward chaining by
    binding ?var to the object/subject position of a fact triple.
    """

    name: str
    head: Triple
    body: list[Triple]
    provenance: Provenance | None = None
    enabled: bool = True

    def variables(self) -> set[str]:
        """Collect all variable names (tokens starting with '?') in head+body."""
        vars_: set[str] = set()
        for t in [self.head, *self.body]:
            for part in (t.subject, t.predicate, t.object):
                if isinstance(part, str) and part.startswith("?"):
                    vars_.add(part)
        return vars_

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "head": self.head.to_dict(),
                "body": [t.to_dict() for t in self.body],
                "provenance": (self.provenance.to_dict() if self.provenance else None),
                "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rule":
        return cls(name=str(d["name"]), head=Triple.from_dict(d["head"]),
                   body=[Triple.from_dict(t) for t in d.get("body", [])],
                   provenance=(Provenance.from_dict(d["provenance"]) if d.get("provenance") else None),
                   enabled=bool(d.get("enabled", True)))

    def __str__(self) -> str:
        body = " ^ ".join(str(t) for t in self.body)
        return f"{self.name}: {body} => {self.head}"


@dataclass
class DerivedFact:
    """A fact derived by rule firing during inference.

    triple:        the derived Triple.
    rule:          the Rule that produced it.
    supporting:    list of Fact instances that fired the rule.
    depth:         chaining depth (0 = base fact, N = derived N steps away).
    """

    triple: Triple
    rule: Rule | None
    supporting: list[Fact]
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"triple": self.triple.to_dict(),
                "rule": (self.rule.name if self.rule else None),
                "supporting": [str(f.triple) for f in self.supporting],
                "depth": self.depth}
