"""Shared helpers for the sequential multi-hop path.

These small utilities are used by the executor, synthesizer, and verifier so the
three components agree on (a) what counts as an unresolved hop answer, (b) how the
resolved hop chain is rendered into a prompt block, and (c) how evidence is ranked
for the bounded prompt window. Keeping them in one place prevents the drift that
previously let an ``UNKNOWN`` hop leak into the synthesis/verifier prompts while
being filtered out of anchor selection.
"""

import re
from typing import List, Optional


_UNKNOWN_SENTINEL = "UNKNOWN"
# Punctuation stripped off a token before comparing it to the UNKNOWN sentinel.
_TOKEN_STRIP = ".,:;!?\"'()[]"

# Content-word tokenizer for the entity-grounding gate: alphanumeric runs, lowered.
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Function words that carry no entity signal, so their presence/absence in the
# evidence says nothing about whether the answer is grounded. Kept small and
# generic (articles, prepositions, conjunctions, copulas) to avoid over-filtering
# real answer tokens.
_ENTITY_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for", "from",
        "by", "with", "as", "is", "was", "are", "were", "be", "it", "its", "this",
        "that", "these", "those", "he", "she", "they", "his", "her", "their",
    }
)


def hop_answer_ungrounded(answer, evidence_texts) -> bool:
    """True when a hop answer shares no content token with the hop's evidence.

    A deterministic, reject-only grounding gate for named/textual hop answers,
    mirroring the verifier's ungrounded-number gate. It fires only when every
    content token of the answer is absent from the concatenated evidence -- the
    parametric-leak signature (the model named an entity the evidence never
    mentions). Sharing even one content token keeps the answer (paraphrase- and
    partial-match-tolerant), which is what keeps this from re-introducing the
    over-strict extraction that once poisoned ~1/3 of hops.

    Returns False (never gate) when: the answer is empty/UNKNOWN (the UNKNOWN path
    already owns it); the answer has no content tokens after stopword/number
    removal (a bare number/stopword -- the numeric gate owns that); or there is no
    evidence text at all (a retrieval fault, handled elsewhere -- not a leak).
    """
    text = (answer or "").strip()
    if not text or is_unknown_hop_answer(text):
        return False
    tokens = [
        t for t in _CONTENT_TOKEN_RE.findall(text.lower())
        if len(t) > 1 and not t.isdigit() and t not in _ENTITY_STOPWORDS
    ]
    if not tokens:
        return False
    haystack = " ".join((t or "") for t in evidence_texts).lower()
    if not haystack.strip():
        return False
    return not any(t in haystack for t in tokens)


def is_unknown_hop_answer(answer) -> bool:
    """True when a hop answer is missing/unresolved (empty or literally UNKNOWN).

    Two shapes count as unresolved:

    * the answer is empty, or its first token is ``UNKNOWN`` (case-insensitive) --
      the model answered with the plain sentinel (``"unknown."``); and
    * the answer is verbose prose that trails off into the all-caps ``UNKNOWN``
      sentinel anywhere in it (e.g. ``"A Don is from Laos. ... UNKNOWN"``). The
      grounding instruction emits the sentinel in all-caps, so this embedded check
      is case-sensitive: a title like ``"The Unknown Soldier"`` (title-cased
      ``Unknown``) is a real answer and stays resolved. This stops a half-answered
      hop from being threaded verbatim into the next hop's ``#n`` reference (the
      query-poisoning bug where prose containing ``UNKNOWN`` became a search query).
    """
    text = (answer or "").strip()
    if not text:
        return True
    if text.upper().split()[0].strip(_TOKEN_STRIP) == _UNKNOWN_SENTINEL:
        return True
    # Embedded all-caps sentinel anywhere -> the hop did not fully resolve.
    return any(tok.strip(_TOKEN_STRIP) == _UNKNOWN_SENTINEL for tok in text.split())


# Draft prefixes / phrases that signal the model refused or produced a non-answer
# rather than answering (a refusal or an "insufficient evidence" abstention). Such a
# draft must never verify as ``supported`` and marks the point where the direct-answer
# fallback should try one more evidence-grounded synthesis. Shared here so the verifier
# (false-positive guard) and the executor (fallback trigger) agree on the definition.
_NON_ANSWER_CUES = (
    "cannot be answered",
    "can't be answered",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i don't have",
    "i do not have",
    "not enough information",
    "no answer",
    "insufficient information",
    "insufficient evidence",
    "the question cannot",
)


def is_non_answer(draft_answer) -> bool:
    """True when the draft is empty or a refusal/hedge rather than an answer.

    Recognizes three shapes: an empty draft, a bare UNKNOWN sentinel (the grounding
    non-answer), and an explicit refusal/abstention cue at/near the start of the text.
    Kept conservative -- only the leading 80 chars are scanned for a cue -- so a real
    answer that merely mentions one of these phrases later is not misflagged. Shared by
    the verifier (which forces such a draft to ``unsupported``) and the executor's
    direct-answer fallback (which fires only when the accepted draft is a non-answer).
    """
    text = (draft_answer or "").strip()
    if not text:
        return True
    if is_unknown_hop_answer(text):
        return True
    head = text.lower()[:80]
    return any(cue in head for cue in _NON_ANSWER_CUES)


def resolved_hops(hop_answers: Optional[List[dict]]) -> List[dict]:
    """Return only the hops that resolved to a real (non-UNKNOWN, non-empty) answer."""
    if not hop_answers:
        return []
    return [
        h
        for h in hop_answers
        if (h.get("subquery") or h.get("hop_query"))
        and not is_unknown_hop_answer(h.get("answer"))
    ]


def hop_answer_anchors(hop_answers: Optional[List[dict]]) -> List[str]:
    """Lowercased resolved hop answers usable as evidence-selection anchors.

    Skips empty/UNKNOWN answers and anything too short to match meaningfully.
    """
    anchors = []
    for h in resolved_hops(hop_answers):
        cleaned = (h.get("answer") or "").strip()
        if len(cleaned) >= 2:
            anchors.append(cleaned.lower())
    return anchors


def render_hop_chain(hop_answers: Optional[List[dict]], header: str, trailer: str = "") -> str:
    """Render the resolved multi-hop chain as a compact prompt block.

    Only hops that resolved to a real answer are shown (empty/UNKNOWN hops are
    filtered, matching anchor selection). Returns an empty string when nothing
    resolved, leaving the caller's prompt unchanged.
    """
    lines = []
    for h in resolved_hops(hop_answers):
        sub = h.get("subquery") or h.get("hop_query") or ""
        ans = (h.get("answer") or "").strip()
        lines.append(f"- {sub} -> {ans}")
    if not lines:
        return ""
    return f"{header}\n" + "\n".join(lines) + trailer


def render_hop_chain_with_gaps(hop_answers: Optional[List[dict]], header: str, trailer: str = "") -> str:
    """Render every hop, marking unresolved ones explicitly (for the verifier).

    Unlike ``render_hop_chain`` (which hides UNKNOWN/empty hops so a non-answer
    never leaks into the synthesizer prompt), this shows the full chain and labels
    a hop the executor could not ground as ``(NOT RESOLVED from evidence)``. The
    verifier needs to see that a required intermediate fact was never established:
    otherwise it judges the final answer against a curated, self-consistent chain
    and rubber-stamps a value the model filled in from parametric knowledge. Returns
    an empty string when there are no hops, leaving the prompt unchanged.
    """
    if not hop_answers:
        return ""
    lines = []
    for h in hop_answers:
        sub = h.get("subquery") or h.get("hop_query") or ""
        if not sub:
            continue
        ans = (h.get("answer") or "").strip()
        if is_unknown_hop_answer(ans):
            lines.append(f"- {sub} -> (NOT RESOLVED from evidence)")
        else:
            lines.append(f"- {sub} -> {ans}")
    if not lines:
        return ""
    return f"{header}\n" + "\n".join(lines) + trailer


def rank_evidence(evidence_store, anchors=None):
    """Return evidence ordered by (anchor hit, relevance), stable on insertion order.

    Items whose text contains an answer anchor come first, then higher
    reranker/retrieval score, then original order as a stable tiebreak. Shared by
    the synthesizer (so it drafts from the strongest evidence) and the verifier (so
    it judges against the same window the answer was drafted from), which previously
    used disjoint selection strategies.
    """
    anchors = anchors or []

    def _relevance(ev):
        score = ev.rerank_score if ev.rerank_score is not None else ev.retrieval_score
        return score if score is not None else float("-inf")

    def _anchor_hit(ev):
        low = (ev.text or "").lower()
        return 1 if any(a in low for a in anchors) else 0

    ordered = sorted(
        enumerate(evidence_store),
        key=lambda pair: (_anchor_hit(pair[1]), _relevance(pair[1]), -pair[0]),
        reverse=True,
    )
    return [ev for _, ev in ordered]
