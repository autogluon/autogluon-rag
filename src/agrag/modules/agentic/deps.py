"""Lightweight ``#n`` dependency validation for the sequential multi-hop path.

MuSiQue-style plans thread earlier hops' answers into later hops via ``#n``
back-references (``#1`` = the answer of hop 1). This module parses those references
per hop and validates them structurally, without introducing typed nodes or a full
DAG: it answers two questions the executor needs cheaply --

* which earlier hops does hop ``i`` depend on (its prerequisite indices), and
* does hop ``i`` contain an invalid reference (self/forward/out-of-range), which
  can never resolve and should be surfaced as ``dependency_unresolved``?

Because a valid ``#n`` always points at an earlier hop (``n < i``), the natural
subquery order is already a valid topological order -- the executor does not need
to reorder hops. Dependency info is used only to (a) flag invalid references and
(b) recognize when a hop's prerequisite grounded to UNKNOWN so the hop can be
tagged rather than silently threading a non-answer.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set

# Same numeric back-reference shape the executor substitutes (``#1``, ``#12``).
_REF_PATTERN = re.compile(r"#(\d+)")


@dataclass
class HopDeps:
    """Parsed dependencies for a single hop.

    Attributes:
    ----------
    index : int
        0-based position of this hop in ``subqueries``.
    prereqs : Set[int]
        0-based indices of the earlier hops this hop validly references.
    invalid_refs : List[int]
        The raw 1-based ``#n`` values that are structurally impossible for this
        hop: ``n < 1``, ``n`` pointing at this hop itself, or ``n`` at/after this
        hop (a forward reference), or ``n`` beyond the number of hops. Such a hop
        can never fully resolve its query and is flagged ``dependency_unresolved``.
    """

    index: int
    prereqs: Set[int] = field(default_factory=set)
    invalid_refs: List[int] = field(default_factory=list)


def parse_hop_dependencies(subqueries: List[str]) -> List[HopDeps]:
    """Parse and validate ``#n`` references for each hop in ``subqueries``.

    Returns one ``HopDeps`` per subquery, in order. A reference ``#n`` on hop at
    0-based position ``i`` (1-based hop number ``i + 1``) is valid iff
    ``1 <= n <= i`` (it points at a strictly earlier hop that exists). Valid
    references become 0-based prerequisites (``n - 1``); everything else is
    recorded in ``invalid_refs``.
    """
    total = len(subqueries)
    result: List[HopDeps] = []
    for i, subquery in enumerate(subqueries):
        deps = HopDeps(index=i)
        for raw in _REF_PATTERN.findall(subquery or ""):
            n = int(raw)
            # Valid: 1-based reference to a strictly earlier, existing hop.
            if 1 <= n <= i and n <= total:
                deps.prereqs.add(n - 1)
            else:
                deps.invalid_refs.append(n)
        result.append(deps)
    return result
