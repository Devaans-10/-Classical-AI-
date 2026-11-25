"""Rules engine: forward and backward chaining over the triple KB.

AI technique: Rule-based inference (forward chaining + backward chaining).

Why: Some answers are not stored directly in the document and must be derived.
We use Horn-style rules (head :- body) over triples. Forward chaining derives
all consequences of the KB (data-driven, good for exploration). Backward
chaining proves a specific query goal (goal-driven, used to answer a question
and to build the inference trace). Both produce an explicit inference chain
(fact used -> rule fired -> derived fact -> answer) which is the required
reasoning trace. Never guesses: if neither facts nor rules support the answer,
the answerer returns "Not in document".

Variable naming: any token starting with '?' in a rule is a variable. Comparison
conditions are encoded as Triples whose predicate starts with 'cmp:' and whose
subject/object are variables or numbers, e.g. (?attendance, cmp:ge, 75) means
?attendance >= 75. Numeric comparison operators: ge, le, gt, lt, eq.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kb.store import TripleStore, _norm
from kb.types import DerivedFact, Fact, Provenance, Rule, Triple

_CMP_RE = re.compile(r"^cmp:(ge|le|gt|lt|eq)$")
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def _is_var(term: str) -> bool:
    return isinstance(term, str) and term.startswith("?")


def _is_num(term: Any) -> bool:
    if isinstance(term, (int, float)):
        return True
    if _is_var(term):
        return False
    if isinstance(term, str):
        cleaned = re.sub(r"[^\d.-]", "", term.strip())
        return bool(cleaned and re.match(r"^-?\d+(?:\.\d+)?$", cleaned))
    return False


def _to_num(term: Any) -> float | None:
    if isinstance(term, (int, float)):
        return float(term)
    if isinstance(term, str):
        cleaned = re.sub(r"[^\d.-]", "", term.strip())
        if cleaned and re.match(r"^-?\d+(?:\.\d+)?$", cleaned):
            return float(cleaned)
    return None


def _eval_cmp(op: str, left: Any, right: Any) -> bool:
    ln = _to_num(left)
    rn = _to_num(right)
    if ln is None or rn is None:
        return False
    if op == "ge":
        return ln >= rn
    if op == "le":
        return ln <= rn
    if op == "gt":
        return ln > rn
    if op == "lt":
        return ln < rn
    if op == "eq":
        return ln == rn
    return False


def _match_pattern(pattern: Triple, fact: Triple, bindings: dict[str, Any]) -> bool:
    """Try to match a pattern triple (with ?vars) against a concrete fact triple.

    Updates `bindings` in place on success. Returns False on conflict.
    Comparison predicates (cmp:*) are evaluated numerically rather than matched.
    """
    pairs = [(pattern.subject, fact.subject), (pattern.predicate, fact.predicate),
             (pattern.object, fact.object)]
    for pterm, fterm in pairs:
        if _is_var(pterm):
            if pterm in bindings:
                if _norm(str(bindings[pterm])) != _norm(str(fterm)):
                    return False
            else:
                bindings[pterm] = fterm
        elif _is_var(fterm):
            # object side variable in fact won't happen (facts are concrete), but be safe.
            if fterm in bindings:
                if _norm(str(bindings[fterm])) != _norm(str(pterm)):
                    return False
            else:
                bindings[fterm] = pterm
        else:
            if _norm(str(pterm)) != _norm(str(fterm)):
                return False
    return True


def _instantiate(triple: Triple, bindings: dict[str, Any]) -> Triple:
    """Replace ?vars in a triple with their bound values."""
    def repl(term: Any) -> Any:
        if _is_var(term):
            return bindings.get(term, term)
        return term
    return Triple(subject=repl(triple.subject), predicate=repl(triple.predicate),
                  object=repl(triple.object))


def _condition_satisfied(condition: Triple, facts: list[Fact],
                         bindings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return list of extended bindings satisfying one condition against facts.

    For comparison conditions (cmp:*), evaluate against current bindings.
    For ordinary triples, match against every fact and collect bindings.
    """
    m = _CMP_RE.match(condition.predicate)
    if m:
        op = m.group(1)
        left = condition.subject
        right = condition.object
        # Resolve bound vars.
        l = bindings.get(left, left) if _is_var(left) else left
        r = bindings.get(right, right) if _is_var(right) else right
        if _eval_cmp(op, l, r):
            return [dict(bindings)]
        return []

    results: list[dict[str, Any]] = []
    for f in facts:
        b = dict(bindings)
        if _match_pattern(condition, f.triple, b):
            results.append(b)
    return results


class InferenceEngine:
    """Forward- and backward-chaining inference over a TripleStore."""

    def __init__(self, store: TripleStore) -> None:
        self.store = store

    # ---------------------------------------------------------------- #
    # Forward chaining
    # ---------------------------------------------------------------- #
    def forward_chain(self, max_iterations: int = 100) -> list[DerivedFact]:
        """Derive all consequences by repeatedly firing enabled rules.

        Returns the list of derived facts in the order they were derived, each
        with the rule fired and supporting facts. Stops when a pass adds no new
        facts or max_iterations is reached.
        """
        facts: list[Fact] = list(self.store.facts())
        rules = [r for r in self.store.rules() if r.enabled]
        derived: list[DerivedFact] = []
        derived_keys: set[tuple[str, str, str]] = {(f.triple.subject, f.triple.predicate,
                                                    f.triple.object) for f in facts}

        for _ in range(max_iterations):
            added_any = False
            for rule in rules:
                # Try all combinations of bindings satisfying the body.
                bindings_list: list[dict[str, Any]] = [{}]
                for condition in rule.body:
                    new_bindings: list[dict[str, Any]] = []
                    for b in bindings_list:
                        new_bindings.extend(_condition_satisfied(condition, facts, b))
                    bindings_list = new_bindings
                    if not bindings_list:
                        break
                if not bindings_list:
                    continue
                for b in bindings_list:
                    head = _instantiate(rule.head, b)
                    key = (head.subject, head.predicate, head.object)
                    if key in derived_keys:
                        continue
                    supporting = [f for f in facts
                                  if _matches_fact_pattern_support(head, f, b, rule)]
                    df = DerivedFact(triple=head, rule=rule,
                                     supporting=supporting,
                                     depth=(max((s.depth if hasattr(s, 'depth') else 0)
                                                for s in supporting) + 1) if supporting else 0)
                    derived.append(df)
                    facts.append(Fact(head, rule.provenance or Provenance(section="rules"),
                                      confidence=1.0))
                    derived_keys.add(key)
                    added_any = True
            if not added_any:
                break
        return derived

    # ---------------------------------------------------------------- #
    # Backward chaining
    # ---------------------------------------------------------------- #
    def backward_chain(self, goal: Triple, max_depth: int = 10) -> DerivedFact | None:
        """Attempt to prove `goal` by backward chaining.

        Returns a DerivedFact with the full derivation (rule fired, supporting
        facts) if provable, else None. The DerivedFact.triple is the goal (with
        variables instantiated). Supporting facts are the base facts used at the
        leaves of the derivation.
        """
        result = self._prove(goal, {}, max_depth, [])
        return result

    def _prove(self, goal: Triple, bindings: dict[str, Any], depth: int,
               path: list[Rule]) -> DerivedFact | None:
        if depth <= 0:
            return None
        goal_inst = _instantiate(goal, bindings)

        # 1. Try to match a base fact directly.
        for f in self.store.query(subject=goal_inst.subject,
                                  predicate=goal_inst.predicate,
                                  object_=goal_inst.object):
            return DerivedFact(triple=goal_inst, rule=None, supporting=[f], depth=0)

        # 2. Try rules whose head unifies with the goal.
        for rule in self.store.rules():
            if not rule.enabled:
                continue
            # Unify rule head with goal_inst to get initial bindings.
            head_b: dict[str, Any] = dict(bindings)
            if not _unify(rule.head, goal_inst, head_b):
                continue
            # Prove each body condition under accumulated bindings.
            body_bindings_list: list[dict[str, Any]] = [head_b]
            proven_supporting: list[list[Fact]] = []
            ok = True
            for condition in rule.body:
                new_list: list[dict[str, Any]] = []
                for b in body_bindings_list:
                    # For comparison conditions, _condition_satisfied returns bindings.
                    # For triple conditions, gather supporting facts too.
                    if _CMP_RE.match(condition.predicate):
                        res = _condition_satisfied(condition, self.store.facts(), b)
                        new_list.extend(res)
                    else:
                        # Need one fact per binding; collect facts that satisfy.
                        fact_options: list[tuple[dict[str, Any], list[Fact]]] = []
                        for f in self.store.facts():
                            bb = dict(b)
                            if _match_pattern(condition, f.triple, bb):
                                fact_options.append((bb, [f]))
                        for bb, supp in fact_options:
                            new_list.append(bb)
                            proven_supporting.append(supp)
                    body_bindings_list = new_list
                    if not body_bindings_list:
                        ok = False
                        break
                if not ok:
                    break
            if not body_bindings_list:
                continue
            # Take the first successful binding set; supporting facts = union.
            final_b = body_bindings_list[0]
            supporting: list[Fact] = []
            # Re-collect supporting facts for final_b across body triple conditions.
            for condition in rule.body:
                if _CMP_RE.match(condition.predicate):
                    continue
                found = False
                for f in self.store.facts():
                    bb = dict(final_b)
                    if _match_pattern(condition, f.triple, bb):
                        # ensure consistency with final_b
                        if all(_norm(str(bb.get(k, v))) == _norm(str(v)) for k, v in final_b.items()):
                            supporting.append(f)
                            found = True
                            break
            derived_triple = _instantiate(rule.head, final_b)
            return DerivedFact(triple=derived_triple, rule=rule,
                               supporting=supporting, depth=1)
        return None


def _unify(pattern: Triple, concrete: Triple, bindings: dict[str, Any]) -> bool:
    """Unify a pattern triple (may contain ?vars) with a concrete triple."""
    pairs = [(pattern.subject, concrete.subject), (pattern.predicate, concrete.predicate),
             (pattern.object, concrete.object)]
    for pterm, cterm in pairs:
        if _is_var(pterm):
            if pterm in bindings:
                if _norm(str(bindings[pterm])) != _norm(str(cterm)):
                    return False
            else:
                bindings[pterm] = cterm
        else:
            if _norm(str(pterm)) != _norm(str(cterm)):
                return False
    return True


def load_rules_from_json(path: str | Path) -> list[Rule]:
    """Load rules from the editable JSON rule base into Rule objects.

    The JSON file is hand-authored and editable without changing code. Each rule
    is a Horn clause: head :- body. 'cmp:*' predicates are numeric comparisons;
    variables start with '?'.
    """
    import json
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rules: list[Rule] = []
    for rd in data.get("rules", []):
        prov = None
        if rd.get("provenance"):
            prov = Provenance(section=str(rd["provenance"].get("section", "")),
                              clause=rd["provenance"].get("clause"),
                              page=rd["provenance"].get("page"))
        rules.append(Rule(name=str(rd["name"]),
                          head=Triple(subject=str(rd["head"]["subject"]),
                                      predicate=str(rd["head"]["predicate"]),
                                      object=str(rd["head"]["object"])),
                          body=[Triple(subject=str(c["subject"]),
                                       predicate=str(c["predicate"]),
                                       object=str(c["object"])) for c in rd.get("body", [])],
                          provenance=prov,
                          enabled=bool(rd.get("enabled", True))))
    return rules


def _matches_fact_pattern_support(head: Triple, fact: Fact, bindings: dict[str, Any],
                                  rule: Rule) -> bool:
    """Heuristic: mark a fact as supporting if it was used to bind the body."""
    # Called during forward chaining; approximate support by body match.
    for condition in rule.body:
        if _CMP_RE.match(condition.predicate):
            continue
        b = dict(bindings)
        if _match_pattern(condition, fact.triple, b):
            return True
    return False
