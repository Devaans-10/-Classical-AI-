"""Triple store: the Knowledge Base (KB).

AI technique: Knowledge Representation + graph search (BFS/DFS).

Why: Facts are stored as triples in an indexed store so that any question can be
reduced to a graph query. The KB is queried directly (forward lookup) and
traversed (BFS/DFS over subject-predicate-object edges) to find connected
entities and multi-hop derivations before rules are fired. Indexes make lookup
O(1) per predicate/subject/object, keeping responses fast and deterministic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterator

from kb.types import Fact, Provenance, Rule, Triple


from kb.types import Fact, Provenance, Rule, Triple, _norm_subject, _norm_predicate


def _norm(value: str) -> str:
    """Normalize object values (lowercase, whitespace-collapsed, preserved
    verbatim otherwise -- objects are not stemmed or article-stripped)."""
    return " ".join(str(value).strip().split()).lower()


class TripleStore:
    """In-memory triple store with three indexes (subject, predicate, object).

    The store holds base facts (extracted from the document) and derived facts
    (produced by the rules engine). It supports:
      * exact / wildcard triple queries
      * graph traversal (BFS and DFS) over the entity-relation graph
      * rule registration for the inference engine
    """

    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._rules: list[Rule] = []
        # Indexes map a normalized component -> set of fact indices.
        self._by_subject: dict[str, set[int]] = defaultdict(set)
        self._by_predicate: dict[str, set[int]] = defaultdict(set)
        self._by_object: dict[str, set[int]] = defaultdict(set)
        self._fact_keys: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------ #
    # Fact management
    # ------------------------------------------------------------------ #
    def add_fact(self, fact: Fact) -> bool:
        """Add a fact if its (subject, predicate, object) key is new.

        Returns True when the fact was inserted, False when it was a duplicate.
        Duplicate prevention keeps the KB compact and inference deterministic.
        """
        key = fact.triple.key()
        norm_key = (_norm_subject(fact.triple.subject), _norm_predicate(fact.triple.predicate),
                    _norm(fact.triple.object))
        if norm_key in self._fact_keys:
            return False
        self._fact_keys.add(norm_key)
        idx = len(self._facts)
        self._facts.append(fact)
        self._by_subject[norm_key[0]].add(idx)
        self._by_predicate[norm_key[1]].add(idx)
        self._by_object[norm_key[2]].add(idx)
        return True

    def add_facts(self, facts: list[Fact]) -> int:
        added = 0
        for f in facts:
            added += int(self.add_fact(f))
        return added

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def add_rules(self, rules: list[Rule]) -> None:
        for r in rules:
            self.add_rule(r)

    def facts(self) -> list[Fact]:
        return list(self._facts)

    def rules(self) -> list[Rule]:
        return list(self._rules)

    def fact_count(self) -> int:
        return len(self._facts)

    def rule_count(self) -> int:
        return len(self._rules)

    def section_count(self) -> int:
        return len({f.provenance.section for f in self._facts})

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def _candidate_indices(self, subject: str | None, predicate: str | None,
                           object_: str | None) -> set[int]:
        """Intersect indexes for the non-None query components."""
        pools = []
        if subject is not None:
            pools.append(self._by_subject.get(_norm_subject(subject), set()))
        if predicate is not None:
            pools.append(self._by_predicate.get(_norm_predicate(predicate), set()))
        if object_ is not None:
            pools.append(self._by_object.get(_norm(object_), set()))
        if not pools:
            return set(range(len(self._facts)))
        result = set(pools[0])
        for pool in pools[1:]:
            result &= pool
        return result

    def query(self, subject: str | None = None, predicate: str | None = None,
              object_: str | None = None) -> list[Fact]:
        """Return facts matching the given (subject, predicate, object) pattern.

        Any component may be None (wildcard). Matching is case-insensitive and
        whitespace-normalized.
        """
        indices = self._candidate_indices(subject, predicate, object_)
        return [self._facts[i] for i in sorted(indices)]

    def get(self, subject: str, predicate: str,
            object_: str | None = None) -> list[Fact]:
        """Convenience: facts for a specific subject-predicate pair."""
        return self.query(subject=subject, predicate=predicate, object_=object_)

    def values(self, subject: str, predicate: str) -> list[str]:
        """Return the distinct object values for (subject, predicate)."""
        return sorted({f.triple.object for f in self.get(subject, predicate)})

    def predicates_for(self, subject: str) -> list[str]:
        """Return distinct predicates outgoing from a subject."""
        return sorted({f.triple.predicate for f in self.query(subject=subject)})

    def subjects_with(self, predicate: str, object_: str) -> list[str]:
        """Return distinct subjects having (predicate, object)."""
        return sorted({f.triple.subject for f in self.query(predicate=predicate,
                                                           object_=object_)})

    # ------------------------------------------------------------------ #
    # Graph search over the KB (BFS / DFS)
    # ------------------------------------------------------------------ #
    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        """Outgoing (predicate, object) edges from an entity."""
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for f in self.query(subject=entity):
            edge = (f.triple.predicate, f.triple.object, f.provenance.short())
            if edge not in seen:
                seen.add(edge)
                out.append(edge)
        return out

    def incoming(self, entity: str) -> list[tuple[str, str, str]]:
        """Incoming (subject, predicate) edges to an entity."""
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for f in self.query(object_=entity):
            edge = (f.triple.subject, f.triple.predicate, f.provenance.short())
            if edge not in seen:
                seen.add(edge)
                out.append(edge)
        return out

    def bfs(self, start: str, max_depth: int = 3) -> list[dict[str, Any]]:
        """Breadth-first traversal from `start` up to max_depth.

        Returns a list of visited nodes with the edge used to reach them and
        depth. Used to explore multi-hop connections between entities mentioned
        in a question before applying rules.
        """
        start_norm = _norm(start)
        visited: dict[str, dict[str, Any]] = {start_norm: {
            "entity": start, "depth": 0, "via": None, "provenance": None}}
        queue: deque[tuple[str, int]] = deque([(start_norm, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for predicate, obj, prov in self.neighbors(current):
                nnorm = _norm(obj)
                if nnorm not in visited:
                    visited[nnorm] = {"entity": obj, "depth": depth + 1,
                                      "via": predicate, "provenance": prov}
                    queue.append((nnorm, depth + 1))
        return list(visited.values())

    def dfs(self, start: str, max_depth: int = 3) -> list[dict[str, Any]]:
        """Depth-first traversal from `start` up to max_depth (mirrors BFS)."""
        start_norm = _norm(start)
        visited: dict[str, dict[str, Any]] = {}
        stack: list[tuple[str, int]] = [(start_norm, 0)]
        while stack:
            current, depth = stack.pop()
            if current in visited or depth > max_depth:
                continue
            # Find how we reached `current` (re-query for provenance on re-entry).
            entry = visited.get(current)
            visited[current] = {"entity": start if depth == 0 else current,
                                "depth": depth, "via": entry.get("via") if entry else None,
                                "provenance": entry.get("provenance") if entry else None}
            for predicate, obj, prov in self.neighbors(current):
                nnorm = _norm(obj)
                if nnorm not in visited:
                    stack.append((nnorm, depth + 1))
        return list(visited.values())

    def summary(self) -> dict[str, Any]:
        predicates = sorted({f.triple.predicate for f in self._facts})
        sections = sorted({f.provenance.section for f in self._facts})
        return {"facts": self.fact_count(), "rules": self.rule_count(),
                "sections": len(sections), "predicates": len(predicates),
                "predicate_list": predicates, "section_list": sections}

    def to_dict(self) -> dict[str, Any]:
        return {"facts": [f.to_dict() for f in self._facts],
                "rules": [r.to_dict() for r in self._rules]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TripleStore":
        store = cls()
        for fd in d.get("facts", []):
            store.add_fact(Fact.from_dict(fd))
        for rd in d.get("rules", []):
            store.add_rule(Rule.from_dict(rd))
        return store

    def __len__(self) -> int:
        return self.fact_count()

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts)
