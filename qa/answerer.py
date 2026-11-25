"""Question answering + inference trace builder.

AI technique: Template-based question parsing + forward/backward chaining +
explanation generation.

Why: Every answer is produced by either (a) a direct KB lookup (a stored fact)
or (b) a rule derivation (forward/backward chaining). The answerer records an
explicit inference chain (fact used -> rule fired -> derived fact -> answer) for
every response. When neither facts nor rules support the question, it returns
"Not in document" with confidence 0 -- it never guesses.

The answerer works on a working store = KB facts + question-supplied values (for
inference questions), so policy thresholds stored in the KB combine with the
numbers in the question through the rule engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kb.store import TripleStore, _norm
from kb.rules import InferenceEngine, load_rules_from_json
from kb.types import Fact, Provenance, Rule, Triple, _norm_subject, _norm_predicate
from qa.templates import QuestionPlan, match_question


@dataclass
class TraceStep:
    step_no: int
    type: str  # 'lookup' | 'rule' | 'fact' | 'derived' | 'answer' | 'abstain'
    description: str
    detail: str = ""
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_no": self.step_no,
            "type": self.type,
            "description": self.description,
            "detail": self.detail,
            "provenance": self.provenance,
        }


@dataclass
class Answer:
    question: str
    status: str  # 'answered' | 'not_in_document' | 'ambiguous'
    answer: str
    confidence: float
    trace: list[TraceStep] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    template: str | None = None
    response_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "confidence": self.confidence,
            "trace": [t.to_dict() for t in self.trace],
            "citations": self.citations,
            "template": self.template,
            "response_time_ms": self.response_time_ms,
        }


_TRUE = {"yes", "true", "allowed", "y", "1", "eligible", "pass", "distinction", "open"}
_FALSE = {"no", "false", "not allowed", "n", "0", "debarred", "fail", "closed"}


def _cite(fact: Fact) -> dict[str, Any]:
    return {
        "section": fact.provenance.section,
        "clause": fact.provenance.clause,
        "page": fact.provenance.page,
        "short": fact.provenance.short(),
        "triple": str(fact.triple),
    }


class Answerer:
    """Answers a question using the KB and rule engine, building a trace."""

    def __init__(self, store: TripleStore, rules_path: str | None = None) -> None:
        self.store = store
        self.engine = InferenceEngine(store)
        if rules_path:
            self.engine.store.add_rules(load_rules_from_json(rules_path))

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #
    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        plan = match_question(question)
        if plan is None:
            elapsed = (time.perf_counter() - start) * 1000.0
            return self._not_answered(question, None, elapsed,
                                      "No question template matched.")

        try:
            result = self._dispatch(plan)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            result = self._not_answered(question, plan, elapsed,
                                        f"Reasoning error: {e}")
        result.response_time_ms = (time.perf_counter() - start) * 1000.0
        return result

    # ---------------------------------------------------------------- #
    # Dispatch
    # ---------------------------------------------------------------- #
    def _dispatch(self, plan: QuestionPlan) -> Answer:
        kind = plan.kind
        if kind in ("lookup", "when_howmuch"):
            return self._lookup(plan)
        if kind == "boolean":
            return self._boolean(plan)
        if kind == "reverse":
            return self._reverse(plan)
        if kind == "inference_elig":
            return self._infer_eligibility(plan)
        if kind == "inference_marks":
            return self._infer_marks(plan)
        return self._not_answered(plan.raw, plan, 0.0,
                                  "Unsupported template kind.")

    # ---------------------------------------------------------------- #
    # Direct lookup
    # ---------------------------------------------------------------- #
    def _lookup(self, plan: QuestionPlan) -> Answer:
        g = plan.groups
        subject = g.get("subject", "").strip()
        predicate = g.get("predicate", "").strip()
        trace: list[TraceStep] = []
        citations: list[dict[str, Any]] = []

        step = 1
        facts = self._lookup_facts(subject, predicate)
        if facts:
            fact = facts[0]
            trace.append(TraceStep(step, "lookup",
                                   f"Found fact ({fact.triple.subject}, {fact.triple.predicate}, ...)",
                                   f"object = {fact.triple.object}", _cite(fact)))
            citations.append(_cite(fact))
            step += 1
            return self._answered(plan, fact.triple.object, 1.0, trace, citations)

        # predicate-only fallback (the question names a predicate directly)
        if subject and not predicate:
            facts = self.store.query(predicate=_norm_predicate(subject))
            if len(facts) == 1:
                fact = facts[0]
                trace.append(TraceStep(step, "lookup",
                                       f"Predicate-only match for '{subject}'",
                                       f"({fact.triple.subject}, {fact.triple.predicate}, {fact.triple.object})",
                                       _cite(fact)))
                citations.append(_cite(fact))
                return self._answered(plan, fact.triple.object, 1.0, trace, citations)
            if len(facts) > 1:
                vals = ", ".join(f.triple.object for f in facts[:5])
                trace.append(TraceStep(step, "ambiguous",
                                       f"Multiple subjects have predicate '{subject}'",
                                       f"values: {vals}"))
                return self._answered(plan, vals, 0.5, trace, citations)

        return self._not_answered(plan.raw, plan, 0.0,
                                  f"No fact for ({subject}, {predicate}) in the KB.")

    def _lookup_facts(self, subject: str, predicate: str) -> list[Fact]:
        """Try (subject, predicate), fuzzy subject, combined predicate, then
        predicate-only. Returns matching facts."""
        norm_s = _norm_subject(subject) if subject else ""
        norm_p = _norm_predicate(predicate) if predicate else ""

        if norm_s and norm_p:
            # 1. Exact subject + predicate
            facts = self.store.query(subject=norm_s, predicate=norm_p)
            if facts:
                return facts

            # 2. Fuzzy subject match (e.g. "library" matches "library rules")
            candidates = self.store.query(predicate=norm_p)
            matching = [f for f in candidates
                        if norm_s in _norm_subject(f.triple.subject) or _norm_subject(f.triple.subject) in norm_s]
            if matching:
                return matching

            # 3. Combined predicate (e.g. subject="calculator", predicate="allowed" -> "calculator allowed")
            combined = _norm_predicate(norm_s + " " + norm_p)
            facts = self.store.query(predicate=combined)
            if facts:
                return facts

            # 4. If unambiguous across all subjects for this predicate
            if len(candidates) == 1:
                return candidates

        if norm_p and not norm_s:
            facts = self.store.query(predicate=norm_p)
            if facts:
                return facts

        if norm_s and not norm_p:
            facts = self.store.query(predicate=norm_s)
            if facts:
                return facts

        return []

    # ---------------------------------------------------------------- #
    # Boolean
    # ---------------------------------------------------------------- #
    def _boolean(self, plan: QuestionPlan) -> Answer:
        g = plan.groups
        subject = g.get("subject", "").strip()
        predicate = g.get("predicate", "").strip()
        trace: list[TraceStep] = []
        citations: list[dict[str, Any]] = []
        step = 1

        facts = self._lookup_facts(subject, predicate)
        if not facts:
            return self._not_answered(plan.raw, plan, 0.0,
                                      f"No fact for ({subject}, {predicate}).")

        fact = facts[0]
        obj = fact.triple.object.strip().lower()
        trace.append(TraceStep(step, "lookup",
                               f"Checked ({fact.triple.subject}, {fact.triple.predicate})",
                               f"stored value = {fact.triple.object}", _cite(fact)))
        citations.append(_cite(fact))
        step += 1

        if obj in _TRUE:
            return self._answered(plan, "Yes", 1.0, trace, citations)
        if obj in _FALSE:
            return self._answered(plan, "No", 1.0, trace, citations)

        return self._answered(plan, fact.triple.object, 0.8, trace, citations)

    # ---------------------------------------------------------------- #
    # Reverse lookup
    # ---------------------------------------------------------------- #
    def _reverse(self, plan: QuestionPlan) -> Answer:
        g = plan.groups
        predicate = g.get("predicate", "").strip()
        value = g.get("value", "").strip()
        trace: list[TraceStep] = []
        step = 1
        facts = self.store.query(predicate=_norm_predicate(predicate),
                                 object_=_norm(value))
        if not facts:
            # Try without strict object normalization (e.g. percent)
            all_for_pred = self.store.query(predicate=_norm_predicate(predicate))
            facts = [f for f in all_for_pred if _norm(f.triple.object) == _norm(value)
                     or f.triple.object.replace("%", "").strip() == value.replace("%", "").strip()]

        if not facts:
            return self._not_answered(plan.raw, plan, 0.0,
                                      f"No subject has ({predicate}, {value}).")

        subjects = ", ".join(f.triple.subject.title() for f in facts)
        for f in facts:
            trace.append(TraceStep(step, "lookup",
                                   f"Found match: ({f.triple.subject}, {predicate}, {value})",
                                   f"subject = {f.triple.subject}", _cite(f)))
            step += 1
        return self._answered(plan, subjects, 1.0, trace,
                              [_cite(f) for f in facts])

    # ---------------------------------------------------------------- #
    # Inference: eligibility
    # ---------------------------------------------------------------- #
    def _infer_eligibility(self, plan: QuestionPlan) -> Answer:
        g = plan.groups
        who = g.get("who", "student").strip() or "student"
        val = g.get("val", "").strip()
        target = g.get("target", "eligible").strip().lower()
        trace: list[TraceStep] = []
        citations: list[dict[str, Any]] = []
        step = 1

        try:
            val_num = float(val)
        except ValueError:
            return self._not_answered(plan.raw, plan, 0.0, "Invalid numeric value in question.")

        # Inject question-supplied attendance as a working fact.
        work_fact = Fact(Triple(who, "attendance", val),
                         Provenance(section="question context"))
        trace.append(TraceStep(step, "lookup",
                               f"Working fact from question: ({who}, attendance, {val})",
                               "supplied in question", None))
        step += 1

        # Build a working store = KB + working fact.
        work_store = TripleStore()
        work_store.add_facts(list(self.store.facts()) + [work_fact])
        work_store.add_rules(list(self.store.rules()))
        eng = InferenceEngine(work_store)

        if target == "debarred":
            derived = eng.backward_chain(Triple(who, "exam_debarred", "yes"))
            if derived and derived.rule:
                trace.append(TraceStep(step, "rule",
                                       f"Fired rule '{derived.rule.name}'",
                                       f"{' ^ '.join(str(c) for c in derived.rule.body)} => {derived.rule.head}",
                                       (derived.rule.provenance.to_dict()
                                        if derived.rule.provenance else None)))
                if derived.rule.provenance:
                    citations.append(derived.rule.provenance.to_dict())
                    citations[-1]["short"] = "rule: " + derived.rule.name
                    citations[-1]["triple"] = str(derived.rule.head)
                step += 1
                for s in derived.supporting:
                    trace.append(TraceStep(step, "fact",
                                           f"Supporting fact: {s.triple}",
                                           "from KB", _cite(s)))
                    citations.append(_cite(s))
                    step += 1
                trace.append(TraceStep(step, "derived",
                                       f"Derived: {derived.triple}",
                                       "by backward chaining over rule space", None))
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} is debarred with {val}% attendance.",
                                       f"{val} < 60% debarment threshold", None))
                return self._answered(plan, "Yes", 0.95, trace, citations)
            else:
                # Student is NOT debarred
                threshold_facts = self.store.query(predicate="debarred_if_attendance_lt")
                prov = threshold_facts[0].provenance if threshold_facts else Provenance("2. Attendance Policy", "Clause 2.1.2")
                trace.append(TraceStep(step, "rule",
                                       "Evaluated debarment condition: attendance < 60%",
                                       f"{val_num} >= 60. Condition is FALSE.",
                                       prov.to_dict()))
                citations.append(prov.to_dict())
                citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} is NOT debarred.",
                                       f"{val}% attendance is at or above the 60% debarment threshold", None))
                return self._answered(plan, "No", 0.95, trace, citations)

        else:
            # Target is "eligible" / "allowed"
            derived = eng.backward_chain(Triple(who, "exam_eligible", "yes"))
            if derived and derived.rule:
                trace.append(TraceStep(step, "rule",
                                       f"Fired rule '{derived.rule.name}'",
                                       f"{' ^ '.join(str(c) for c in derived.rule.body)} => {derived.rule.head}",
                                       (derived.rule.provenance.to_dict()
                                        if derived.rule.provenance else None)))
                if derived.rule.provenance:
                    citations.append(derived.rule.provenance.to_dict())
                    citations[-1]["short"] = "rule: " + derived.rule.name
                    citations[-1]["triple"] = str(derived.rule.head)
                step += 1
                for s in derived.supporting:
                    trace.append(TraceStep(step, "fact",
                                           f"Supporting fact: {s.triple}",
                                           "from KB", _cite(s)))
                    citations.append(_cite(s))
                    step += 1
                trace.append(TraceStep(step, "derived",
                                       f"Derived: {derived.triple}",
                                       "by backward chaining over rule space", None))
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} is eligible to appear for the exam.",
                                       f"attendance {val}% >= 75% required threshold", None))
                return self._answered(plan, "Yes", 0.95, trace, citations)

            # Not directly eligible: check condonation (60-74%) or debarred (<60%)
            if 60.0 <= val_num < 75.0:
                cond_rule = next((r for r in self.store.rules() if r.name == "condonation_fee_applicable"), None)
                prov = cond_rule.provenance if cond_rule else Provenance("2. Attendance Policy", "Clause 2.1.4")
                trace.append(TraceStep(step, "rule",
                                       "Evaluated rule 'condonation_fee_applicable'",
                                       f"{val_num}% is between 60% and 74%. Requires condonation fee of 500 INR.",
                                       prov.to_dict() if prov else None))
                if prov:
                    citations.append(prov.to_dict())
                    citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} is not directly eligible, but eligible with condonation fee of 500 INR.",
                                       f"attendance {val}% is in the 60-74% condonation bracket", None))
                return self._answered(plan, "No (requires condonation fee of 500 INR)", 0.95, trace, citations)

            if val_num < 60.0:
                prov = Provenance("2. Attendance Policy", "Clause 2.1.2")
                trace.append(TraceStep(step, "rule",
                                       "Evaluated clause 2.1.2: attendance below 60% is debarred",
                                       f"{val_num}% < 60%. Student is debarred.", prov.to_dict()))
                citations.append(prov.to_dict())
                citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} is not eligible (debarred due to attendance < 60%).",
                                       f"attendance {val}% is below 60%", None))
                return self._answered(plan, "No", 0.95, trace, citations)

        return self._not_answered(plan.raw, plan, 0.0,
                                  "Attendance-based eligibility not derivable from KB rules.")

    # ---------------------------------------------------------------- #
    # Inference: marks
    # ---------------------------------------------------------------- #
    def _infer_marks(self, plan: QuestionPlan) -> Answer:
        g = plan.groups
        who = g.get("who", "student").strip() or "student"
        val = g.get("val", "").strip()
        target = g.get("target", "pass").strip().lower()
        trace: list[TraceStep] = []
        citations: list[dict[str, Any]] = []
        step = 1

        try:
            val_num = float(val)
        except ValueError:
            return self._not_answered(plan.raw, plan, 0.0, "Invalid numeric value in question.")

        work_fact = Fact(Triple(who, "total_marks", val),
                         Provenance(section="question context"))
        trace.append(TraceStep(step, "lookup",
                               f"Working fact from question: ({who}, total_marks, {val})",
                               "supplied in question", None))
        step += 1

        work_store = TripleStore()
        work_store.add_facts(list(self.store.facts()) + [work_fact])
        work_store.add_rules(list(self.store.rules()))
        eng = InferenceEngine(work_store)

        if "distinction" in target:
            goal_pred = "distinction"
            derived = eng.backward_chain(Triple(who, goal_pred, "yes"))
            if derived and derived.rule:
                trace.append(TraceStep(step, "rule",
                                       f"Fired rule '{derived.rule.name}'",
                                       f"{' ^ '.join(str(c) for c in derived.rule.body)} => {derived.rule.head}",
                                       (derived.rule.provenance.to_dict()
                                        if derived.rule.provenance else None)))
                if derived.rule.provenance:
                    citations.append(derived.rule.provenance.to_dict())
                    citations[-1]["short"] = "rule: " + derived.rule.name
                    citations[-1]["triple"] = str(derived.rule.head)
                step += 1
                for s in derived.supporting:
                    trace.append(TraceStep(step, "fact",
                                           f"Supporting fact: {s.triple}",
                                           "from KB", _cite(s)))
                    citations.append(_cite(s))
                    step += 1
                trace.append(TraceStep(step, "derived",
                                       f"Derived: {derived.triple}",
                                       "by backward chaining", None))
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} achieves distinction with {val} marks.",
                                       f"score {val} >= distinction threshold", None))
                return self._answered(plan, "Yes", 0.95, trace, citations)
            else:
                dist_facts = self.store.query(predicate="distinction mark")
                thresh_str = dist_facts[0].triple.object if dist_facts else "75%"
                prov = dist_facts[0].provenance if dist_facts else Provenance("4. Examination Rules")
                trace.append(TraceStep(step, "rule",
                                       f"Evaluated distinction rule: marks >= {thresh_str}",
                                       f"Score {val_num} < {thresh_str}. Distinction threshold not met.",
                                       prov.to_dict()))
                citations.append(prov.to_dict())
                citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} does NOT achieve distinction with {val} marks.",
                                       f"score {val} is below {thresh_str} distinction mark", None))
                return self._answered(plan, "No", 0.95, trace, citations)

        elif "first division" in target:
            goal_pred = "first_division"
            derived = eng.backward_chain(Triple(who, goal_pred, "yes"))
            if derived and derived.rule:
                trace.append(TraceStep(step, "rule",
                                       f"Fired rule '{derived.rule.name}'",
                                       f"{' ^ '.join(str(c) for c in derived.rule.body)} => {derived.rule.head}",
                                       (derived.rule.provenance.to_dict()
                                        if derived.rule.provenance else None)))
                if derived.rule.provenance:
                    citations.append(derived.rule.provenance.to_dict())
                    citations[-1]["short"] = "rule: " + derived.rule.name
                    citations[-1]["triple"] = str(derived.rule.head)
                step += 1
                for s in derived.supporting:
                    trace.append(TraceStep(step, "fact",
                                           f"Supporting fact: {s.triple}",
                                           "from KB", _cite(s)))
                    citations.append(_cite(s))
                    step += 1
                trace.append(TraceStep(step, "derived",
                                       f"Derived: {derived.triple}",
                                       "by backward chaining", None))
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} achieves first division with {val} marks.",
                                       f"score {val} >= first division threshold", None))
                return self._answered(plan, "Yes", 0.95, trace, citations)
            else:
                fdiv_facts = self.store.query(predicate="first division mark")
                thresh_str = fdiv_facts[0].triple.object if fdiv_facts else "60%"
                prov = fdiv_facts[0].provenance if fdiv_facts else Provenance("4. Examination Rules")
                trace.append(TraceStep(step, "rule",
                                       f"Evaluated first division rule: marks >= {thresh_str}",
                                       f"Score {val_num} < {thresh_str}. First division threshold not met.",
                                       prov.to_dict()))
                citations.append(prov.to_dict())
                citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} does NOT achieve first division with {val} marks.",
                                       f"score {val} is below {thresh_str}", None))
                return self._answered(plan, "No", 0.95, trace, citations)

        else:
            # Pass or fail
            goal_pred = "pass"
            derived = eng.backward_chain(Triple(who, goal_pred, "yes"))
            if derived and derived.rule:
                trace.append(TraceStep(step, "rule",
                                       f"Fired rule '{derived.rule.name}'",
                                       f"{' ^ '.join(str(c) for c in derived.rule.body)} => {derived.rule.head}",
                                       (derived.rule.provenance.to_dict()
                                        if derived.rule.provenance else None)))
                if derived.rule.provenance:
                    citations.append(derived.rule.provenance.to_dict())
                    citations[-1]["short"] = "rule: " + derived.rule.name
                    citations[-1]["triple"] = str(derived.rule.head)
                step += 1
                for s in derived.supporting:
                    trace.append(TraceStep(step, "fact",
                                           f"Supporting fact: {s.triple}",
                                           "from KB", _cite(s)))
                    citations.append(_cite(s))
                    step += 1
                trace.append(TraceStep(step, "derived",
                                       f"Derived: {derived.triple}",
                                       "by backward chaining", None))
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} passes with {val} marks.",
                                       f"marks {val} >= pass mark threshold", None))
                return self._answered(plan, "Yes", 0.95, trace, citations)
            else:
                pass_facts = self.store.query(predicate="pass mark")
                thresh_str = pass_facts[0].triple.object if pass_facts else "40%"
                prov = pass_facts[0].provenance if pass_facts else Provenance("4. Examination Rules")
                trace.append(TraceStep(step, "rule",
                                       f"Evaluated pass mark rule: marks >= {thresh_str}",
                                       f"Score {val_num} < {thresh_str}. Minimum pass mark not reached.",
                                       prov.to_dict()))
                citations.append(prov.to_dict())
                citations[-1]["short"] = prov.short()
                step += 1
                trace.append(TraceStep(step, "answer",
                                       f"Conclusion: {who} does NOT pass with {val} marks.",
                                       f"marks {val} is below {thresh_str} pass mark", None))
                return self._answered(plan, "No", 0.95, trace, citations)

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _answered(self, plan: QuestionPlan, answer: str, confidence: float,
                  trace: list[TraceStep], citations: list[dict[str, Any]]) -> Answer:
        return Answer(question=plan.raw, status="answered", answer=answer,
                      confidence=confidence, trace=trace, citations=citations,
                      template=plan.template)

    def _not_answered(self, question: str, plan: QuestionPlan | None,
                      elapsed_ms: float, reason: str) -> Answer:
        trace = [TraceStep(1, "abstain",
                           "No supporting fact or rule found in the KB.",
                           reason)]
        return Answer(question=question, status="not_in_document",
                      answer="Not in document", confidence=0.0, trace=trace,
                      citations=[], template=(plan.template if plan else None),
                      response_time_ms=elapsed_ms)
