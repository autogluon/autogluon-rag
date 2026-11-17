"""Benchmark standard RAG vs. agentic RAG on the MuSiQue multi-hop QA benchmark.

MuSiQue (Trivedi et al., 2021 -- arXiv:2108.00573; HuggingFace mirror
``dgslibisey/MuSiQue``) builds multi-hop questions by composing single-hop
questions, so answering requires 2-4 connected reasoning hops. Each question
ships its own ~20-paragraph pool -- a few supporting paragraphs plus distractors.
In the paper's "distractor" setting each question is answered against only its own
pool.

This runner does not use the per-question pools in isolation. It merges every
selected question's paragraphs into one deduplicated global corpus, indexed once,
and answers every question against that whole corpus (``build_global_corpus``). A
question's supporting paragraphs must therefore be found among all questions'
paragraphs, not just its own 20 -- a harder, more realistic global-retrieval task,
and deliberately no longer the paper's official distractor benchmark. Identical
Wikipedia paragraphs shared across questions are written only once.

This runner builds a single corpus and index up front (via ``initialize_rag_pipeline``
over the fully populated global corpus dir), so the embedding / reranker /
generator models load once and no per-question re-indexing happens. Both modes
(standard and agentic) run over the same global index through the same
``generate_response(..., mode=...)`` entry point; the agentic workflow itself is
measured as-is. Total corpus size grows with the number of questions, so very
large samples will be slow (see ``build_global_corpus``).

The agentic run here uses the LLM planner + policy (the shared Bedrock generator
decomposes the query into subqueries and chooses the next action among the legal
set), not the deterministic regex planner / rule-based action cascade. These flags
are set in ``main`` on the ``AutoGluonRAG`` instance rather than in the yaml, so
``configs/agent/default.yaml`` (LLM off by default) is untouched.

By default this reads the frozen, self-contained eval set produced by
``build_musique_eval_set.py`` (``--eval-set``, default
``local_example/evaluation_data_musique/musique_eval_set.30.jsonl`` -- a 30-row
stratified slice, 5 rows per hop type, all answerable) so runs are offline and
reproducible. Pass ``--from-hf`` (or delete the frozen file) to load
``dgslibisey/MuSiQue`` from HuggingFace instead, applying the same reproducible
row selection. Either way the loaded rows are merged into the global corpus.

Reuses ``local_example/local_config.yaml`` (Bedrock Cohere Embed English v3
embeddings + Bedrock Claude Sonnet 4.6 generator). Source ``credential.sh`` for
Bedrock access before running. The config's saved-index paths are not written to:
this runner overrides the vector-DB save/load flags in memory so the global index
never touches disk.
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import time

import numpy as np
from datasets import load_dataset

from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.musique.musique import (
    get_musique_answerable,
    get_musique_evidence_facts,
    get_musique_paragraph_docs,
    get_musique_query,
    get_musique_question_type,
    get_musique_responses,
)
from agrag.evaluation.evaluator import EvaluationModule
from agrag.evaluation.llm_judge import AnswerJudge, judge_matches
from agrag.evaluation.retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics_for_query
from agrag.evaluation.utils import (
    calculate_exact_match_score,
    calculate_f1_score,
    f1_metric,
    inclusive_exact_match_metric,
    rouge_geometric_mean,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = "local_example/local_config.yaml"
DATASET = "dgslibisey/MuSiQue"
DEFAULT_SPLIT = "validation"
# Answer-quality metric computed via the evaluator (pure-Python, no model download).
# Matches the MultiHop-RAG / NQ benchmarks; token-F1 is computed separately below.
EM_METRIC = "inclusive_exact_match"
# Fixed seed for reproducible query-row selection (shared across both modes).
DEFAULT_SEED = 1234
# Frozen, self-contained MuSiQue slice produced by build_musique_eval_set.py. Used
# by default so runs are offline and reproducible (no HuggingFace access needed).
# Defaults to the 30-row stratified slice (5 rows per hop type, all answerable),
# which keeps a full run cheap. For a lower-variance measurement, the 90-row set is
# still available via --eval-set (schema is identical).
DEFAULT_EVAL_SET = "local_example/evaluation_data_musique/musique_eval_set.30.jsonl"


def load_rows(eval_set_path, from_hf, split, size, seed, stratify, answerable_only):
    """Return the MuSiQue rows to evaluate, from the frozen file or HuggingFace.

    Frozen path (default): read the self-contained JSONL built by
    ``build_musique_eval_set.py`` -- offline and reproducible, no network. Taken
    whenever ``from_hf`` is False and ``eval_set_path`` exists. HF path
    (``--from-hf``, or when the frozen file is absent): load ``dgslibisey/MuSiQue``
    and apply the same reproducible ``select_query_indices`` the frozen builder
    uses, so both routes evaluate comparable rows.
    """
    if not from_hf and os.path.exists(eval_set_path):
        # The frozen file is evaluated as-is: its rows were already selected (size,
        # seed, split, stratify) when it was built by build_musique_eval_set.py. So
        # those flags cannot re-select here -- warn if the user passed non-defaults
        # so the ignored flags are visible rather than silently dropped. Rebuild the
        # frozen file or pass --from-hf to change the selection. (--answerable-only
        # is still applied downstream in main, so it is not listed here.)
        ignored = []
        if size and size != 30:
            ignored.append(f"--max-eval-size={size}")
        if seed != DEFAULT_SEED:
            ignored.append(f"--seed={seed}")
        if split != DEFAULT_SPLIT:
            ignored.append(f"--split={split}")
        if stratify:
            ignored.append("--stratify")
        if ignored:
            print(
                f"WARNING: {', '.join(ignored)} ignored on the frozen path "
                f"({eval_set_path}); these only take effect with --from-hf or by "
                f"rebuilding the frozen file. Evaluating all frozen rows as-is."
            )
        rows = []
        with open(eval_set_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} frozen MuSiQue rows from {eval_set_path}")
        return rows

    print(f"Loading {DATASET} split '{split}' from HuggingFace ...")
    ds = load_dataset(DATASET, split=split)
    indices = select_query_indices(ds, size, seed=seed, answerable_only=answerable_only, stratify=stratify)
    # Attach the dataset index as _source_index so the HF path records row provenance the
    # same way the frozen builder does; otherwise source_index / selection.source_indices
    # would be all null on --from-hf, defeating traceability.
    rows = [{**ds[i], "_source_index": i} for i in indices]
    print(f"Selected {len(rows)} rows from HuggingFace (seed={seed}, stratify={stratify}).")
    return rows


def _cost_summary(timings):
    n = len(timings)
    total = sum(timings)
    return {
        "num_queries_answered": n,
        "total_latency_s": round(total, 3),
        "avg_latency_s": round(total / n, 3) if n else 0.0,
    }


def _agentic_behavior_summary(agent_runs):
    """Summarize whether the agentic path actually decomposed queries.

    The evidence-coverage advantage over static RAG only materializes when the
    planner issues MULTIPLE retrieval calls / subqueries. If ``retrieval_calls``
    is ~1 across the board, agentic degenerated to a single-shot run and any score
    parity with standard is expected, not a benchmark failure. This makes that
    visible at a glance instead of hidden in the per-query traces.
    """
    n = len(agent_runs)
    if not n:
        return None
    avg_retrieval = sum(r["retrieval_calls"] for r in agent_runs) / n
    avg_subqueries = sum(r["num_subqueries"] for r in agent_runs) / n
    avg_iterations = sum(r["iterations"] for r in agent_runs) / n
    # pct_multi_step counts real decomposition (>1 subquery), not bare re-retrieval.
    # A question that re-retrieved the same query several times without decomposing
    # (num_subqueries<=1) is single-hop reasoning even though retrieval_calls>1 --
    # counting it as multi-step (the old retrieval_calls>1 rule) overstated
    # decomposition (e.g. a 2hop answered at hop 1 with subqueries=[] scored as
    # multi-step). pct_reretrieval keeps the old signal visible separately.
    multi_step = sum(1 for r in agent_runs if r["num_subqueries"] > 1)
    pct_multi_step = round(100.0 * multi_step / n, 1)
    reretrieval = sum(1 for r in agent_runs if r["retrieval_calls"] > 1)
    pct_reretrieval = round(100.0 * reretrieval / n, 1)
    # Describe what actually happened rather than emitting a fixed caveat: a low
    # pct_multi_step means the planner degenerated to single-shot (score parity
    # with standard is then expected); a high one means it genuinely decomposed.
    if pct_multi_step < 20.0:
        note = (
            f"pct_multi_step={pct_multi_step} is low: the planner mostly did NOT "
            "decompose into >1 subquery, so agentic ~= single-shot and score parity "
            f"with standard is expected (pct_reretrieval={pct_reretrieval} counts "
            "re-retrieval without decomposition)."
        )
    else:
        note = (
            f"pct_multi_step={pct_multi_step}: the planner decomposed most queries "
            "into >1 subquery, so agentic is doing real multi-step work."
        )
    return {
        "avg_retrieval_calls": round(avg_retrieval, 2),
        "avg_num_subqueries": round(avg_subqueries, 2),
        "avg_iterations": round(avg_iterations, 2),
        "pct_multi_step": pct_multi_step,
        "pct_reretrieval": pct_reretrieval,
        "note": note,
    }


def _judge_scores_from_verdicts(verdicts):
    """Aggregate tri-state ``llm_judge`` verdicts (True/False/None) into the score fragment.

    Mirrors the judge aggregation in ``_quality_scores``: ``None`` is a FAILED call --
    excluded from the accuracy denominator and counted in ``llm_judge_failures``, never
    credited. Extracted so per-question-type judge accuracy can be derived by slicing the
    verdicts already computed once over all rows, instead of re-invoking the judge per
    bucket (which doubled judge calls and, being non-deterministic, need not agree with
    the overall figure).
    """
    scored = [v for v in verdicts if v is not None]
    failures = len(verdicts) - len(scored)
    return {
        "llm_judge": round(sum(scored) / len(scored), 4) if scored else None,
        "llm_judge_scored": len(scored),
        "llm_judge_failures": failures,
    }


def _quality_scores(evaluator, predictions, references, queries, judge=None, return_item_verdicts=False):
    """Answer-quality metrics for one set of (prediction, references) pairs.

    Exact-match goes through the shared ``EvaluationModule`` (inclusive EM), and
    token-level F1 -- MuSiQue's official answer metric -- is computed directly via
    ``f1_metric``/``calculate_f1_score`` so it is aggregated to a mean (the
    evaluator's callable-metric path would return the raw per-example list). The
    ROUGE geometric mean (ROUGE-1 x ROUGE-2 x ROUGE-L)^(1/3) is added the same way,
    via ``rouge_geometric_mean`` so it is aggregated (mean-then-GM) per bucket.

    When ``judge`` (an ``AnswerJudge``) is supplied, a supplementary model-based
    ``llm_judge`` accuracy is added: the fraction of predictions the judge deems
    correct given the question and gold answer(s). It is reported ALONGSIDE, never
    instead of, EM/F1 -- the judge is non-deterministic and not comparable to
    published exact-match numbers, so the deterministic metrics stay the baseline.
    A judge call that errors or returns an unparseable reply is recorded as a
    failure (``None``), not substituted with the exact-match verdict: it is excluded
    from the ``llm_judge`` denominator and counted in ``llm_judge_failures`` so the
    judge metric reflects only rows the judge actually decided (crediting a failed
    call with EM would silently mix a different metric into the judge score). When
    every judge call fails, ``llm_judge`` is reported as ``None``.

    When ``return_item_verdicts`` is True, returns ``(scores, item_verdicts)`` where
    ``item_verdicts`` holds the index-aligned per-example lists
    ``{"inclusive_em", "strict_em", "llm_judge"}``. ``llm_judge`` entries are
    ``True``/``False`` per judged row, ``None`` for a failed judge call, and all
    ``None`` when no judge is supplied. These let callers persist per-question
    pass/fail so EM-vs-judge disagreement and run-to-run flips can be audited.
    Otherwise returns just ``scores`` (unchanged legacy behavior).
    """
    if not predictions:
        empty = {
            EM_METRIC: 0.0,
            "strict_exact_match": 0.0,
            "f1": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rouge_gm": 0.0,
            "count": 0,
        }
        if judge is not None:
            empty["llm_judge"] = 0.0
        if return_item_verdicts:
            return empty, {"inclusive_em": [], "strict_em": [], "llm_judge": []}
        return empty
    em = evaluator.evaluate_responses(predictions=predictions, references=references, queries=queries)
    # Strict EM: normalized equality (case/punct-insensitive, same normalization as
    # F1/ROUGE), WITHOUT substring containment -- so a gold answer merely buried in a
    # long prediction is not credited. Reported alongside the inclusive EM to expose
    # the substring-inflation gap; treat this + F1/ROUGE as the primary quality signal.
    strict_matches = inclusive_exact_match_metric(
        predictions=predictions,
        references=references,
        ignore_case=True,
        ignore_punctuation=True,
        substring=False,
    )
    strict_em = calculate_exact_match_score(strict_matches)
    f1 = calculate_f1_score(f1_metric(predictions, references))
    rouge = rouge_geometric_mean(predictions, references)
    scores = {
        **em,
        "strict_exact_match": round(strict_em, 4),
        "f1": round(f1, 4),
        **rouge,
        "count": len(predictions),
    }
    # Per-example inclusive-EM verdicts (substring containment) -- the same signal
    # the aggregated EM_METRIC mean is built from. Persisted (when requested) as a
    # per-question verdict alongside the judge; no longer used as a judge fallback.
    inclusive_matches = inclusive_exact_match_metric(
        predictions=predictions, references=references, ignore_case=True, ignore_punctuation=True, substring=True
    )
    verdicts = None
    if judge is not None:
        # A failed judge call is recorded as None and EXCLUDED from the accuracy --
        # never credited the exact-match verdict (which would mix metrics). Report the
        # failure count so a flaky judge is visible rather than silently masked.
        verdicts = judge_matches(judge, predictions, references, queries)
        scores.update(_judge_scores_from_verdicts(verdicts))
    if return_item_verdicts:
        item_verdicts = {
            "inclusive_em": [bool(x) for x in inclusive_matches],
            "strict_em": [bool(x) for x in strict_matches],
            # Preserve tri-state: True/False for judged rows, None for a failed call.
            "llm_judge": [(None if v is None else bool(v)) for v in verdicts]
            if verdicts is not None
            else [None] * len(predictions),
        }
        return scores, item_verdicts
    return scores


def build_global_corpus(rows, work_dir):
    """Merge every selected question's paragraphs into ONE deduplicated corpus dir.

    Instead of the paper's per-question distractor pools, this pools all selected
    questions' paragraphs into a single corpus that is indexed once and queried by
    every question -- so each question's supporting paragraphs must be found among
    all questions' paragraphs (a harder, more realistic global-retrieval setting).

    Paragraphs are deduplicated by exact string identity: the same Wikipedia
    paragraph appearing in multiple questions is written only once. The identity
    key is the exact ``get_musique_paragraph_docs`` string (title-prepended), which
    is also what the index ingests, so a paragraph maps to exactly one global file.

    Each unique paragraph is written once as ``para_{global_i}.txt`` (``global_i``
    is first-seen order). Returns ``doc_to_global`` mapping each paragraph string to
    its global index, so downstream Support-F1 can translate a question's gold
    supporting paragraphs into global corpus indices for Support-F1 scoring.

    The corpus dir is populated fully before the pipeline is initialized, so the
    normal one-time ``initialize_rag_pipeline`` build indexes the whole global
    corpus -- no per-question re-indexing is needed.

    Note on scale: total corpus size grows with the number of questions. The
    vector DB's duplicate removal builds an O(n^2) similarity matrix and BM25 is a
    pure-Python O(corpus) scan per query, so very large samples will be slow; the
    default ~30-question samples (a few hundred paragraphs after dedup) are fine.
    """
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    doc_to_global = {}
    total = 0
    for row in rows:
        for doc in get_musique_paragraph_docs(row):
            total += 1
            if doc in doc_to_global:
                continue
            global_i = len(doc_to_global)
            doc_to_global[doc] = global_i
            with open(os.path.join(work_dir, f"para_{global_i}.txt"), "w", encoding="utf-8") as f:
                f.write(doc + "\n")

    print(
        f"Global corpus: {total} paragraphs across {len(rows)} questions "
        f"-> {len(doc_to_global)} unique written to {work_dir}"
    )
    return doc_to_global


def run_query(agrag, query, mode):
    """Execute ONE comparable run for ``query`` in ``mode`` and time all of it.

    Both modes go through a single ``generate_response(..., return_trace=True)``
    call, so exactly one retrieval + generation happens per standard query. The
    whole comparable operation -- retrieval and generation -- is timed for both.
    (The global corpus is indexed once up front; that build is not attributed to
    either mode.)

    Returns a dict with the answer, the ranked evidence texts the answer was
    conditioned on (best-first), the full serializable trace, per-query latency,
    and agentic decomposition metrics (``None`` for standard).
    """
    resolved = mode or "standard"
    start = time.perf_counter()
    answer, trace = agrag.generate_response(query, mode=resolved, return_trace=True)
    latency = time.perf_counter() - start

    trace = trace if isinstance(trace, dict) else {}
    evidence = trace.get("evidence", []) or []
    # Evidence is stored best-first per retrieval in both modes; keep that order.
    texts = [ev.get("text", "") for ev in evidence]

    agent_metrics = None
    if resolved == "agentic":
        tmetrics = trace.get("metrics", {}) or {}
        agent_metrics = {
            "retrieval_calls": tmetrics.get("retrieval_calls", 0),
            "num_subqueries": len(trace.get("subqueries", []) or []),
            "iterations": tmetrics.get("iterations", 0),
        }

    return {
        "answer": answer,
        "evidence_texts": texts,
        "evidence": evidence,
        "trace": trace,
        "latency": latency,
        "agent_metrics": agent_metrics,
    }


def select_query_indices(queries_ds, max_eval_size, seed, answerable_only=False, stratify=False):
    """Select the query-row indices to evaluate, reproducibly.

    The same indices are used for both modes so the comparison is paired. Only rows
    with a non-empty expected answer are eligible (the metrics need a reference);
    with ``answerable_only`` the unanswerable rows are also dropped. Selection is
    deterministic given ``seed``.

    stratify=False : first ``max_eval_size`` eligible rows in dataset order.
    stratify=True  : reproducible stratified sample by hop-count question type --
        eligible rows are bucketed by type, each bucket shuffled with the fixed
        seed, then round-robined so every hop count is represented proportionally.

    Returns a sorted list of dataset indices.
    """
    eligible = []
    for idx, row in enumerate(queries_ds):
        if not get_musique_responses(row):
            continue
        if answerable_only and not get_musique_answerable(row):
            continue
        eligible.append(idx)

    if not max_eval_size or max_eval_size >= len(eligible):
        return eligible

    if not stratify:
        return eligible[:max_eval_size]

    buckets = {}
    for idx in eligible:
        qtype = get_musique_question_type(queries_ds[idx])
        buckets.setdefault(qtype, []).append(idx)

    rng = random.Random(seed)
    for qtype in sorted(buckets):
        rng.shuffle(buckets[qtype])

    # Round-robin across types (sorted for determinism) until the quota is met.
    ordered_types = sorted(buckets)
    selected = []
    position = 0
    while len(selected) < max_eval_size:
        progressed = False
        for qtype in ordered_types:
            bucket = buckets[qtype]
            if position < len(bucket):
                selected.append(bucket[position])
                progressed = True
                if len(selected) >= max_eval_size:
                    break
        if not progressed:
            break
        position += 1
    return sorted(selected)


def build_evaluator(agrag):
    """An EvaluationModule with the EM metric initialized, ready for evaluate_responses."""
    evaluator = EvaluationModule(rag_instance=agrag)
    evaluator.metrics = [EM_METRIC]
    evaluator.metric_init_params = {}
    # Normalize the aggregated inclusive-EM the same way as the per-example verdict and
    # the other quality metrics (strict_em / f1 / ROUGE): case- and punctuation-
    # insensitive. Without this the evaluator's defaults (ignore_case=False,
    # ignore_punctuation=False; articles-only) make the headline inclusive_exact_match
    # stricter than every metric beside it AND disagree with verdicts["inclusive_em"],
    # which is built with ignore_case/ignore_punctuation=True. evaluate_responses forwards
    # **metric_score_params into inclusive_exact_match_metric, so this takes effect.
    evaluator.metric_score_params = {"ignore_case": True, "ignore_punctuation": True}
    evaluator.metric_instances = evaluator.initialize_metrics([EM_METRIC])
    return evaluator


def _evidence_provenance(evidence):
    """Compact per-chunk provenance for the JSONL row (drops bulky embeddings)."""
    provenance = []
    for rank, ev in enumerate(evidence):
        provenance.append(
            {
                "rank": ev.get("rank", rank),
                "doc_id": ev.get("doc_id"),
                "chunk_id": ev.get("chunk_id"),
                "source": ev.get("source"),
                "retrieval_score": ev.get("retrieval_score"),
                "rerank_score": ev.get("rerank_score"),
            }
        )
    return provenance


def apply_rerank_setting(agrag, use_reranker, reranker_top_k=None):
    """Reconfigure the already-built pipeline's reranker gate WITHOUT re-indexing.

    ``use_reranker`` only gates the retriever's rerank stage; the vector index,
    embeddings, and parent store are untouched. Re-initializing the retriever
    module rebuilds it against the same in-memory ``vector_db_module`` (cheap --
    no embedding or index work), and clearing the cached agentic module forces it
    to rebuild against the new retriever on the next agentic query. When
    ``reranker_top_k`` is given, the Reranker instance is rebuilt too (its truncation
    cutoff is baked in at construction) so the sweep can vary how many chunks
    survive reranking -- otherwise enabling reranking silently reintroduces the
    truncate-to-reranker_top_k step and confounds the comparison with "fewer chunks
    reached the generator".
    """
    agrag.args.use_reranker = use_reranker
    if reranker_top_k is not None:
        agrag.args.reranker_top_k = reranker_top_k
        agrag.initialize_reranker_module()
    agrag.initialize_retriever_module()
    # The retriever was just rebuilt; re-attach the parent store (small-to-big
    # expansion) and drop the cached agentic module so it rebinds to the new retriever.
    agrag._attach_parent_store_to_retriever()
    agrag.agentic_module = None
    print(f"  reranking {'ON' if use_reranker else 'OFF'}" + (f" (top_k={reranker_top_k})" if reranker_top_k else ""))


def rerank_variants(rerank_arg):
    """Map the ``--rerank`` choice to an ordered list of ``(use_reranker, suffix)``.

    ``config`` (default) leaves the pipeline exactly as the yaml configured it --
    a single unsuffixed run, so behavior is unchanged when the flag is omitted.
    ``on``/``off`` force a single setting; ``both`` runs the on and off settings
    back to back so the two can be compared side by side in one results file.
    """
    if rerank_arg == "both":
        return [(True, "_rerank_on"), (False, "_rerank_off")]
    if rerank_arg == "on":
        return [(True, "")]
    if rerank_arg == "off":
        return [(False, "")]
    return [(None, "")]  # "config": no override


def run_mode(agrag, evaluator, rows, mode, jsonl_writer=None, label=None, judge=None, run_index=0):
    """Run one evaluation pass over the pre-selected rows.

    ``rows`` is the shared, reproducible list of MuSiQue rows used for BOTH modes
    (paired comparison). Every question is queried against the same global corpus,
    which was merged and indexed once before this call. ``jsonl_writer`` is an
    optional callable receiving one dict per query, written as a JSONL row.
    ``label`` overrides the ``mode`` field written to the JSONL / printed header
    (used to tag rerank-sweep variants, e.g. ``agentic_rerank_on``); ``mode`` itself
    still drives ``generate_response``. ``judge`` (an ``AnswerJudge``), when given,
    adds the supplementary ``llm_judge`` accuracy to every quality bucket.
    ``run_index`` tags each written JSONL row so repeated runs (``--runs N``) are
    distinguishable in the single predictions file. Returns overall + per-hop-type
    metrics plus the per-query latencies.

    Rows are retained and written AFTER scoring (not streamed inside the loop) so
    the per-question deterministic-EM and LLM-judge verdicts -- computed once the
    full prediction list exists -- can be attached to each row.
    """
    label = label or (mode or "standard")
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG on MuSiQue  (n={len(rows)})")
    print("=" * 72)

    # Bucket by hop-count question type so we can score each type separately.
    buckets = {}
    all_preds, all_refs, all_queries = [], [], []
    all_retrieval = []  # per-query retrieval metrics, rows with gold facts only
    agent_runs = []  # per-query agentic decomposition signals (agentic mode only)
    latencies = []
    written_rows = []  # per-query row dicts, retained so verdicts can be attached
    for idx, row in enumerate(rows):
        expected = get_musique_responses(row)
        if not expected:
            continue
        query = get_musique_query(row)
        qtype = get_musique_question_type(row)
        gold_facts = get_musique_evidence_facts(row)

        # No per-question re-indexing: every question is queried against the same
        # global corpus that was merged and indexed once before this pass.
        run = run_query(agrag, query, mode)
        latencies.append(run["latency"])
        if run["agent_metrics"] is not None:
            agent_runs.append(run["agent_metrics"])

        b = buckets.setdefault(qtype, {"preds": [], "refs": [], "queries": [], "retrieval": [], "indices": []})
        b["preds"].append(run["answer"])
        b["refs"].append(expected)
        b["queries"].append(query)
        all_preds.append(run["answer"])
        all_refs.append(expected)
        all_queries.append(query)
        # Record this row's position in the overall (all_preds) lists so per-type judge
        # accuracy can be sliced from the overall item_verdicts computed once below,
        # rather than re-invoking the judge per bucket.
        b["indices"].append(len(all_preds) - 1)

        # Retrieval scoring needs gold facts; unanswerable rows may have none, so
        # they are excluded from retrieval metrics (undefined) but still answer-scored.
        rmetrics = {}
        if gold_facts:
            rmetrics = retrieval_metrics_for_query(run["evidence_texts"], gold_facts)
            b["retrieval"].append(rmetrics)
            all_retrieval.append(rmetrics)

        # Retain the row; verdicts are attached and it is written after scoring.
        written_rows.append(
            {
                "mode": label,
                "run_index": run_index,
                "row_index": idx,
                "source_index": row.get("_source_index"),
                "question_type": qtype,
                "answerable": get_musique_answerable(row),
                "query": query,
                "references": expected,
                "prediction": run["answer"],
                "evidence_texts": run["evidence_texts"],
                "evidence_provenance": _evidence_provenance(run["evidence"]),
                "retrieval_metrics": rmetrics,
                "latency_s": round(run["latency"], 4),
                "agent_metrics": run["agent_metrics"],
                "trace": run["trace"],
            }
        )

    overall, item_verdicts = _quality_scores(
        evaluator, all_preds, all_refs, all_queries, judge=judge, return_item_verdicts=True
    )
    # item_verdicts lists are index-aligned with all_preds, hence with written_rows
    # (both skip the same answer-less rows). Attach the per-question pass/fail so
    # EM-vs-judge disagreement and run-to-run flips can be audited from the JSONL.
    for i, wrow in enumerate(written_rows):
        wrow["verdicts"] = {
            "inclusive_em": item_verdicts["inclusive_em"][i],
            "strict_em": item_verdicts["strict_em"][i],
            "llm_judge": item_verdicts["llm_judge"][i],
        }
        if jsonl_writer is not None:
            jsonl_writer(wrow)

    per_type = {}
    for qtype, b in sorted(buckets.items()):
        # Deterministic quality metrics per bucket (cheap, no model). The judge is not
        # re-run here: its per-type accuracy is sliced from the overall item_verdicts
        # (judged once over all rows), keeping per-type consistent with overall and
        # halving judge cost.
        quality = _quality_scores(evaluator, b["preds"], b["refs"], b["queries"], judge=None)
        if judge is not None:
            quality.update(_judge_scores_from_verdicts([item_verdicts["llm_judge"][i] for i in b["indices"]]))
        per_type[qtype] = {
            "count": len(b["preds"]),
            "quality": quality,
            "retrieval": aggregate_retrieval_metrics(b["retrieval"]),
        }

    result = {
        "quality_overall": overall,
        "retrieval_overall": aggregate_retrieval_metrics(all_retrieval),
        "quality_by_question_type": per_type,
        "cost": _cost_summary(latencies),
    }
    behavior = _agentic_behavior_summary(agent_runs)
    if behavior is not None:
        result["agentic_behavior"] = behavior
        print(f"\nAgentic decomposition: {json.dumps(behavior)}")
    return result


def _aggregate_runs(run_dicts):
    """Aggregate a label's per-run metrics dicts into a mean +/- std shape.

    ``run_dicts`` is the list of ``run_mode`` results for one label across repeated
    runs (same eval set, same settings, different Bedrock samples). The result has
    the identical nested shape, but every numeric leaf becomes
    ``{"mean", "std", "n_runs", "runs": [raw per-run values]}`` so run-to-run noise
    is explicit and CIs can be recomputed. Nested dicts (quality_overall,
    quality_by_question_type[*], retrieval, cost, agentic_behavior) recurse;
    non-numeric leaves (e.g. the behavior ``note`` string) take the first run's
    value. Booleans are treated as non-numeric so they are not averaged.
    """
    first = run_dicts[0]
    out = {}
    for key, sample in first.items():
        values = [d[key] for d in run_dicts if key in d]
        if isinstance(sample, dict):
            out[key] = _aggregate_runs([v for v in values if isinstance(v, dict)])
        elif isinstance(sample, bool) or not isinstance(sample, (int, float)):
            out[key] = sample  # strings / bools / None: keep the first run's value
        else:
            nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            out[key] = {
                "mean": round(float(np.mean(nums)), 4),
                "std": round(float(np.std(nums)), 4),
                "n_runs": len(nums),
                "runs": [round(float(v), 4) for v in nums],
            }
    return out


def configure_agent_overrides(agrag):
    """Apply the MuSiQue decoding + agentic-workflow overrides to ``agrag`` in memory.

    Sets temperature=0 for deterministic decoding and drives the agentic path with
    the LLM planner + policy, iterative (sequential-hop) planning, multi-hop
    recovery, and the hop/cost budgets and deep-hop knobs. Every setting is applied
    on the ``AutoGluonRAG`` instance rather than in the yaml, so
    ``local_config.yaml`` and ``configs/agent/default.yaml`` stay untouched. Several
    knobs are env-gated so the identical run yields both arms of a paired A/B.

    Must be called before ``initialize_rag_pipeline()`` because temperature=0 has to
    be set before the generator module is built. Extracted here (behavior-preserving)
    so ``main`` and other entry points (e.g. ``benchmark_musique_1000.py``) share one
    source of truth for the workflow configuration and cannot drift.
    """
    # Deterministic decoding: pin temperature=0 so the whole shared-generator
    # pipeline (planner, LLM policy, synthesizer, verifier, executor hypothesize/
    # replan, LLM tools) AND the LLM judge decode greedily. Removes run-to-run
    # synthesis-phrasing nondeterminism that otherwise masks the recovery signal
    # under a brittle substring EM metric. Near-deterministic on Bedrock/Claude
    # (not bitwise). Must be set before initialize_rag_pipeline() below, since the
    # generator module is built during init. Overridden here (not in the yaml) to
    # keep local_config.yaml untouched, matching the other overrides in this block.
    platform_args = dict(agrag.args.generator_model_platform_args or {})
    bedrock_generate_params = dict(platform_args.get("bedrock_generate_params", {}))
    bedrock_generate_params["temperature"] = 0
    platform_args["bedrock_generate_params"] = bedrock_generate_params
    agrag.args.generator_model_platform_args = platform_args
    # Drive the agentic path with the LLM planner + policy (via the shared Bedrock
    # generator) instead of the deterministic regex planner / rule-based action
    # cascade. The agentic module is built lazily on the first mode="agentic"
    # query and reads these flags then, so setting them here (not in the yaml)
    # takes effect while leaving configs/agent/default.yaml untouched.
    agrag.args.agent_use_llm_planner = True
    agrag.args.agent_use_llm_policy = True
    # Sequential-hop execution: resolve subqueries in order, threading each hop's
    # resolved answer into the next hop's retrieval query. Targets the dominant
    # decomposition-failure bucket (late hops retrieved blind). Set here (not in the
    # yaml) so configs/agent/default.yaml stays at its opt-out default.
    agrag.args.agent_use_iterative_planner = True
    # Plan depth: the default cap of 4 subqueries truncates genuine 4-hop MuSiQue
    # questions (the convergent "4hop2" bucket decomposes into 5-7 subqueries), so
    # the answer-bearing tail hops were dropped before retrieval ever ran. Raise the
    # first-pass ceiling to 6 (replan already goes to max_subqueries+2 = 8). Env-
    # overridable; set here (not in the yaml) to keep local_config.yaml untouched,
    # matching the other overrides in this block.
    agrag.args.agent_max_subqueries = int(os.environ.get("MUSIQUE_MAX_SUBQUERIES", "6"))
    # Multi-hop recovery (sequential path). Hop recovery: on an UNKNOWN hop,
    # hypothesize a candidate and use it only as an extra search query, then
    # re-extract (never fabricates the answer). Replan recovery: on a failed final
    # verification, re-decompose the question and re-run the hops -- targets the
    # dominant wrong/under-granular-decomposition failure bucket. Set here (not in
    # the yaml) so configs/agent/default.yaml stays at its opt-out default.
    # Env-gated so the identical script produces both arms of a paired A/B: run
    # with MUSIQUE_RECOVERY=0 for the recovery-off baseline and MUSIQUE_RECOVERY=1
    # (or unset) for the recovery-on arm, on the same eval set at temp=0. Only the
    # agentic path reads these; the standard path is unaffected.
    recovery_on = os.environ.get("MUSIQUE_RECOVERY", "1") != "0"
    agrag.args.agent_use_hop_recovery = recovery_on
    agrag.args.agent_use_replan_recovery = recovery_on
    agrag.args.agent_max_recovery_attempts = 2
    # Global hop budget: cap total hop retrievals across the initial plan + all
    # replans so a replan-heavy question cannot blow up to ~18 retrievals (observed
    # in a 4hop2 run). Bounds worst-case cost without touching the common path.
    agrag.args.agent_max_total_hops = int(os.environ.get("MUSIQUE_MAX_TOTAL_HOPS", "12"))
    # Real cost budgets (wall-clock + call caps) so a pathological question cannot
    # run unbounded even within the hop budget: a stuck rewrite/verify loop is
    # bounded by LLM-call count, and a slow backend by wall-clock. On breach the run
    # returns its best-effort answer tagged ANSWERED_UNVERIFIED (reason
    # ``budget_exhausted``), preserving the never-refuse contract. Env-overridable;
    # 0 (or unset default 0) disables a given cap so the baseline is unchanged unless
    # explicitly opted in. Set here (not in the yaml) to keep local_config.yaml and
    # configs/agent/default.yaml untouched, matching the other overrides above.
    def _opt_budget(env_name):
        raw = os.environ.get(env_name, "0")
        try:
            val = float(raw)
        except ValueError:
            return None
        return val if val > 0 else None

    agrag.args.agent_max_wall_clock_s = _opt_budget("MUSIQUE_MAX_WALL_CLOCK_S")
    agrag.args.agent_max_llm_calls = _opt_budget("MUSIQUE_MAX_LLM_CALLS")
    agrag.args.agent_max_retrieval_calls = _opt_budget("MUSIQUE_MAX_RETRIEVAL_CALLS")
    # Deep-hop robustness knobs (all env-gated so the identical script yields both
    # A/B arms on the same eval set at temp=0; each defaults to off -> baseline).
    #   MUSIQUE_DEEP_TOPK=N   : deep hops (those depending on an earlier hop, or at
    #                           index >= threshold) retrieve N chunks instead of the
    #                           base per-query top_k, so a resolved bridge entity has
    #                           more candidates in the pooled corpus. 0/unset -> off.
    #   MUSIQUE_ENTITY_GATE=1 : reject a hop answer that shares no content token with
    #                           that hop's evidence (a parametric leak) and route it
    #                           through recovery/UNKNOWN. Paired with hop recovery
    #                           (already on unless MUSIQUE_RECOVERY=0) so a gated hop
    #                           can re-retrieve rather than only drop to UNKNOWN.
    #   MUSIQUE_DROP_PREFIX=1 : drop the single-word answer prefix on the final
    #                           synthesis so multi-hop answers keep qualifiers
    #                           ("75% of the world's teak", not "teak").
    #   MUSIQUE_DIRECT_ANSWER_FALLBACK=1 : when the final draft is still a non-answer/
    #                           abstention after all recovery, re-synthesize once with the
    #                           hop chain dropped so a single broken hop no longer forces
    #                           "INSUFFICIENT EVIDENCE", answering directly from evidence.
    #                           Recovers false abstentions; fires only on already-wrong
    #                           drafts so it cannot regress exact-match/judge.
    _deep_topk = int(os.environ.get("MUSIQUE_DEEP_TOPK", "0"))
    agrag.args.agent_deep_hop_top_k = _deep_topk if _deep_topk > 0 else None
    agrag.args.agent_use_entity_grounding = os.environ.get("MUSIQUE_ENTITY_GATE", "0") == "1"
    agrag.args.agent_final_answer_drop_prefix = os.environ.get("MUSIQUE_DROP_PREFIX", "0") == "1"
    agrag.args.agent_use_direct_answer_fallback = os.environ.get("MUSIQUE_DIRECT_ANSWER_FALLBACK", "0") == "1"
    # Round N+1 synthesis/verification robustness knobs (targets the ~94%-of-genuine-
    # failures synthesis+verification bucket; retrieval is effectively solved). All
    # env-gated so the identical script yields both A/B arms on the same eval set.
    #   MUSIQUE_VERIFIER_MODEL=<id> : give the in-loop verifier its own model (default
    #                           us.anthropic.claude-opus-4-8) instead of reusing the
    #                           sonnet synthesizer, breaking the synthesizer<->verifier
    #                           error correlation behind the ~114 silent-wrong accepts.
    #                           Empty/unset here defaults to opus; set "" to disable
    #                           (reuse the generator) for the verifier-off baseline arm.
    #   MUSIQUE_DUAL_SYNTHESIS=1 : always compute a chain-free candidate alongside the
    #                           chain draft and arbitrate/reconcile them (supersedes the
    #                           abstention-only DIRECT_ANSWER_FALLBACK); targets the
    #                           false-abstention + broken-hop + final-selection buckets.
    #                           Default ON (1) for this round; set 0 for the baseline arm.
    #   MUSIQUE_SELF_CONSISTENCY=K : draft the final answer K times at a sampling temp
    #                           and keep the majority (K=1 -> off). Reduces one-off
    #                           misreads. Default 1/off until (verifier+dual) validated.
    _verifier_model = os.environ.get("MUSIQUE_VERIFIER_MODEL", "us.anthropic.claude-opus-4-8")
    agrag.args.agent_verifier_model = _verifier_model or None
    agrag.args.agent_dual_synthesis = os.environ.get("MUSIQUE_DUAL_SYNTHESIS", "1") == "1"
    agrag.args.agent_self_consistency = int(os.environ.get("MUSIQUE_SELF_CONSISTENCY", "1"))


def main():
    # Run relative to the repo root so config/data paths resolve regardless of the
    # caller's cwd. Done here (not at import time) so importing this module -- e.g.
    # build_musique_eval_set.py reuses select_query_indices -- has no side effects.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        default=DEFAULT_EVAL_SET,
        help=f"Frozen MuSiQue JSONL to evaluate offline (default: {DEFAULT_EVAL_SET}).",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Ignore the frozen file and load from HuggingFace (needs network).",
    )
    parser.add_argument(
        "--max-eval-size", type=int, default=30, help="Sample size when loading from HuggingFace (default: 30)."
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help=f"MuSiQue split when --from-hf (default: {DEFAULT_SPLIT})."
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Evaluate only answerable questions (drops rows where the answer is absent from the paragraphs).",
    )
    parser.add_argument(
        "--evaluation-dir", default="local_example/evaluation_data_musique", help="Where results are written."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Repeat the WHOLE evaluation N times and report mean +/- std per metric "
            "(default: 1 = single run, unchanged output shape). Bedrock/Claude at "
            "temperature=0 is greedy-in-expectation but NOT deterministic, so single "
            "runs cannot separate a real change from run-to-run noise; N>=3 quantifies "
            "that noise. Each run's predictions are written to the JSONL tagged with "
            "run_index. NOTE: cost scales linearly with --runs."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Seed for reproducible query selection (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratified sampling by hop-count question type (default: first-N eligible rows in dataset order).",
    )
    parser.add_argument(
        "--rerank",
        choices=["config", "on", "off", "both"],
        default="config",
        help=(
            "Cross-encoder reranking sweep. 'config' (default): use whatever the yaml "
            "sets (unchanged behavior). 'on'/'off': force one setting. 'both': run "
            "reranking on AND off back to back and tag results (_rerank_on/_rerank_off) "
            "for side-by-side comparison. No pipeline-code change; only the retriever's "
            "use_reranker gate is toggled (no re-indexing)."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=(
            "When reranking is enabled, override reranker_top_k (the truncation cutoff). "
            "Vary this so the on/off comparison is not confounded by fewer chunks reaching "
            "the generator. Defaults to the yaml value."
        ),
    )
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add a supplementary model-based 'llm_judge' accuracy (does the prediction "
            "correctly answer the question given the gold answer(s)?) alongside EM/F1. "
            "Reuses the configured generator; ~1 extra call per (mode x question). "
            "Non-deterministic and NOT comparable to published EM/F1 -- reported in "
            "addition to, never instead of, them. Use --no-llm-judge to disable."
        ),
    )
    args = parser.parse_args()

    # Load the same rows for both modes (paired comparison). Frozen JSONL by
    # default (offline); HuggingFace only with --from-hf or if the file is absent.
    # This condition mirrors load_rows' own branch so the recorded selection
    # metadata below matches how the rows were actually chosen.
    used_frozen = not args.from_hf and os.path.exists(args.eval_set)
    rows = load_rows(
        args.eval_set, args.from_hf, args.split, args.max_eval_size, args.seed, args.stratify, args.answerable_only
    )
    # Keep only rows with a usable gold answer (metrics need a reference); with
    # --answerable-only also drop unanswerable rows (the frozen path may include them).
    rows = [r for r in rows if get_musique_responses(r)]
    if args.answerable_only:
        rows = [r for r in rows if get_musique_answerable(r)]
    if not rows:
        raise SystemExit("No eligible questions with a gold answer; nothing to evaluate.")
    print(f"Evaluating {len(rows)} MuSiQue questions; same rows used for both modes.")

    # Merge every selected question's paragraphs into one deduplicated global
    # corpus dir, populated fully before the pipeline is built so the one-time
    # initialize_rag_pipeline indexes the whole corpus (no per-question reindex).
    corpus_dir = os.path.join(tempfile.gettempdir(), "musique_global_corpus")
    build_global_corpus(rows, corpus_dir)

    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    # Never load or persist a shared index: the global corpus is indexed fresh in
    # memory. Overriding here (not in the yaml) keeps local_config.yaml untouched.
    # NB: the effective property is ``use_existing_vector_db_index`` (it reads the
    # ``vector_db.use_existing_vector_db`` config key); assigning the bare
    # ``use_existing_vector_db`` attribute is a silent no-op that only happened to
    # work because the yaml default is already false. Use the real property so this
    # override actually forces a fresh index -- important now that a mismatched
    # on-disk index is a hard load error, not a blind reuse.
    agrag.args.use_existing_vector_db_index = False
    agrag.args.save_vector_db_index = False
    # Deterministic decoding + agentic-workflow configuration (temp=0, LLM planner/
    # policy, iterative planning, recovery, hop/cost budgets, deep-hop knobs). Applied
    # via the shared helper so this runner and benchmark_musique_1000.py stay in lock
    # step. Must run before initialize_rag_pipeline() (temp=0 is read when the
    # generator module is built).
    configure_agent_overrides(agrag)
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)
    # Supplementary model-based correctness metric (reuses the pipeline generator).
    # Reported alongside EM/F1, not as a replacement -- see --llm-judge help.
    judge = AnswerJudge(agrag.generator_module) if args.llm_judge else None

    os.makedirs(args.evaluation_dir, exist_ok=True)
    jsonl_path = os.path.join(args.evaluation_dir, "benchmark_predictions.jsonl")

    # Reranking sweep: one pass per (use_reranker) variant. "config" (default) is a
    # single pass that leaves the yaml setting alone, so omitting --rerank keeps the
    # original two-bucket (standard/agentic) behavior byte-for-byte.
    variants = rerank_variants(args.rerank)
    n_runs = max(1, args.runs)
    # Collect each label's per-run metrics dict so they can be aggregated to
    # mean +/- std after all runs complete. Predictions from every run are written
    # to the single JSONL, each row tagged with run_index (see run_mode).
    per_run = {}
    with open(jsonl_path, "w") as jf:
        def jsonl_writer(row):
            jf.write(json.dumps(row, default=str) + "\n")

        for run_idx in range(n_runs):
            if n_runs > 1:
                print(f"\n########## RUN {run_idx + 1}/{n_runs} ##########")
            for use_reranker, suffix in variants:
                if use_reranker is not None:
                    print(f"\n### Rerank variant: use_reranker={use_reranker} ###")
                    apply_rerank_setting(agrag, use_reranker, args.rerank_top_k)
                for mode, base in ((None, "standard"), ("agentic", "agentic")):
                    label = base + suffix
                    result = run_mode(
                        agrag, evaluator, rows, mode=mode, jsonl_writer=jsonl_writer,
                        label=label, judge=judge, run_index=run_idx,
                    )
                    per_run.setdefault(label, []).append(result)
    print(f"\nSaved per-query predictions to {jsonl_path}")

    # Single run: keep the legacy flat shape byte-for-byte. Multiple runs: replace
    # each metric leaf with {mean, std, n_runs, runs:[...]} so run-to-run noise is
    # explicit and confidence intervals can be recomputed from the raw values.
    if n_runs == 1:
        results = {label: runs[0] for label, runs in per_run.items()}
    else:
        results = {label: _aggregate_runs(runs) for label, runs in per_run.items()}
        results["_run_meta"] = {"n_runs": n_runs, "note": "quality/retrieval metrics are {mean, std, n_runs, runs}"}

    # On the frozen path, size/seed/split/stratify did not drive selection (the
    # rows were selected when the frozen file was built), so record them as null to
    # avoid implying they applied. --answerable-only IS applied on both paths above,
    # so it is reported as-is.
    results["selection"] = {
        "dataset": DATASET,
        "selection_source": "frozen" if used_frozen else "huggingface",
        "eval_set": args.eval_set if used_frozen else None,
        "from_hf": args.from_hf,
        "split": None if used_frozen else args.split,
        "seed": None if used_frozen else args.seed,
        "answerable_only": args.answerable_only,
        "stratify": None if used_frozen else args.stratify,
        "num_selected": len(rows),
        "source_indices": [r.get("_source_index") for r in rows],
        "rerank_sweep": args.rerank,
        "rerank_top_k_override": args.rerank_top_k,
    }

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, shared global corpus + settings)")
    print("=" * 72)
    print(json.dumps({k: v for k, v in results.items() if k != "selection"}, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
