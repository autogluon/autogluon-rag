"""Adapters for the MuSiQue multi-hop QA benchmark (Trivedi et al., 2021).

MuSiQue (arXiv:2108.00573; HuggingFace mirror ``dgslibisey/MuSiQue``) is a
multi-hop question-answering dataset built by *composing* single-hop questions,
so answering requires 2-4 connected reasoning hops. Unlike MultiHop-RAG (one
shared 609-article corpus), MuSiQue ships a **self-contained corpus per
question**: each row carries its own ~20 ``paragraphs`` (a small number of
supporting paragraphs plus distractors), a single gold ``answer`` with
``answer_aliases``, and a ``question_decomposition`` describing the hops.

Row schema (per the HuggingFace dataset viewer):

* ``id``                    -- e.g. ``"2hop__482757_12019"`` (prefix encodes hops).
* ``question``              -- the composed multi-hop question (str).
* ``answer``                -- the single gold answer (str).
* ``answer_aliases``        -- acceptable answer variants (list[str]).
* ``answerable``            -- whether the answer is present in the paragraphs (bool).
* ``paragraphs``            -- list of ``{idx, title, paragraph_text, is_supporting}``.
* ``question_decomposition``-- list of ``{id, question, answer, paragraph_support_idx}``.

Because the corpus is per-question, the companion
``local_example/benchmark_musique.py`` runner re-indexes each question's own
``paragraphs`` before querying -- the faithful "distractor" setting from the
paper, where the retriever must find the few supporting paragraphs among
distractors. These adapters follow the common dataset-adapter contract (dataset
row -> plain text / query / references / gold facts) so the same evaluation and
retrieval-metric utilities apply unchanged.
"""


def get_musique_query(row):
    """Extract the multi-hop question from a MuSiQue row."""
    return row["question"]


def get_musique_responses(row):
    """Extract the acceptable answer(s) for answer-quality scoring.

    Returns the gold ``answer`` plus any ``answer_aliases`` as a list of
    references (the ``EvaluationModule`` metrics and F1 expect a list per query,
    and crediting aliases avoids penalizing a correct answer phrased differently).
    Rows with no usable answer return an empty list and are skipped by the runner.
    """
    answers = []
    answer = row.get("answer")
    if answer:
        answers.append(answer)
    for alias in row.get("answer_aliases") or []:
        if alias and alias not in answers:
            answers.append(alias)
    return answers


def get_musique_paragraph_docs(row):
    """Build the per-question corpus: one ingest-text string per paragraph.

    Each paragraph becomes ``"{title}\\n\\n{paragraph_text}"`` (prepending the
    title so lexical/semantic retrieval can match on it). The returned list --
    supporting paragraphs plus distractors -- is exactly the knowledge base
    indexed for this one question. Empty paragraphs are skipped.
    """
    docs = []
    for para in row.get("paragraphs") or []:
        if not isinstance(para, dict):
            continue
        title = para.get("title") or ""
        text = para.get("paragraph_text") or ""
        if not text:
            continue
        docs.append(f"{title}\n\n{text}" if title else text)
    return docs


def get_musique_evidence_facts(row):
    """Extract the gold supporting-paragraph texts for retrieval scoring.

    The paragraphs flagged ``is_supporting`` are the ones the answer depends on;
    they are the gold relevant passages used to compute Hit@k / recall@k / MRR /
    evidence-coverage against what the pipeline retrieved. Unanswerable rows may
    have no supporting paragraphs, in which case this returns an empty list and
    the runner excludes the query from retrieval metrics (undefined) while still
    answer-scoring it.
    """
    facts = []
    for para in row.get("paragraphs") or []:
        if isinstance(para, dict) and para.get("is_supporting"):
            text = para.get("paragraph_text") or ""
            if text:
                facts.append(text)
    return facts


def get_musique_supporting_flags(row):
    """Per-paragraph ``is_supporting`` flags, aligned with ``get_musique_paragraph_docs``.

    Returns a list of booleans whose i-th entry is whether the i-th *ingested*
    paragraph (after the same empty-text filtering ``get_musique_paragraph_docs``
    applies) is a gold supporting paragraph. This alignment is what lets a caller
    map a retrieved chunk back to its paragraph and decide whether that paragraph
    was one the answer should be grounded on -- the basis for the paper's official
    **Support F1** (supporting-paragraph identification). Keeping the filtering
    identical to ``get_musique_paragraph_docs`` guarantees paragraph index ``i``
    means the same paragraph in both.
    """
    flags = []
    for para in row.get("paragraphs") or []:
        if not isinstance(para, dict):
            continue
        if not (para.get("paragraph_text") or ""):
            continue
        flags.append(bool(para.get("is_supporting")))
    return flags


def get_musique_answerable(row):
    """Whether the answer is present in the paragraphs (defaults to True if absent)."""
    return bool(row.get("answerable", True))


def get_musique_question_type(row):
    """Derive the hop-count bucket used for the per-type metric breakdown.

    MuSiQue encodes the number of reasoning hops in the ``id`` prefix
    (``"2hop"`` / ``"3hop"`` / ``"4hop"``). The static-vs-agentic gap is expected
    to widen with hop count -- more hops means more sub-questions a single-shot
    retrieve-then-read pipeline cannot cover -- so reporting per hop count (rather
    than a single aggregate) is the analog of MultiHop-RAG's ``question_type``
    breakdown. Falls back to the decomposition length, else ``"unknown"``.
    """
    qid = row.get("id") or ""
    if "hop" in qid:
        prefix = qid.split("__")[0]
        if prefix:
            return prefix
    decomposition = row.get("question_decomposition") or []
    if decomposition:
        return f"{len(decomposition)}hop"
    return "unknown"
