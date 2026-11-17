"""Parallel, resumable 1,000-question MuSiQue benchmark (standard vs. agentic RAG).

``benchmark_musique.py`` runs standard vs. agentic RAG on a 30-row frozen slice,
sequentially. At n=30 the run-to-run noise of temperature=0 Bedrock decoding is
comparable to the signal, so conclusions are shaky. This script scales the same
workflow to a fixed, reproducible 1,000-question set and runs queries in parallel
so 1,000 questions are tractable, surviving interruption via per-query incremental
saves + resume.

The RAG workflow, agent logic, and evaluation metrics are not reimplemented here.
This module is orchestration around the existing entry points:

* answering  -> ``benchmark_musique.run_query`` -> ``AutoGluonRAG.generate_response``
* corpus     -> ``benchmark_musique.build_global_corpus`` (one merged global corpus)
* agent cfg  -> ``benchmark_musique.configure_agent_overrides`` (temp=0, LLM planner/
                policy, iterative planning, recovery, hop/cost budgets, deep-hop knobs)
* metrics    -> ``agrag.evaluation.utils`` (inclusive/strict EM, token-F1, ROUGE-GM),
                ``retrieval_metrics`` (Hit/recall/MRR/coverage), ``AnswerJudge``
* evaluator  -> ``benchmark_musique._quality_scores`` / ``build_evaluator``

What this adds on top:

1. Proportional-hop 1,000-question selection frozen to disk (the built-in
   ``--stratify`` gives equal per-hop counts, which is not the same ratio as the
   full dev set, so a proportional sampler is added).
2. ``--workers N`` process-level parallelism (each worker owns its own
   ``AutoGluonRAG`` -- the agentic ``AgentExecutor`` keeps per-run state on a single
   shared instance and is not re-entrant, so processes, not threads).
3. Bedrock throttling handled by the retry added to the shared Bedrock clients
   (``agrag.modules.bedrock_retry``).
4. Per-query atomic save + resume (only missing query IDs are re-run; never
   duplicated).
5. A per-run folder with a full-provenance ``manifest.json``.

Source ``credential.sh`` for Bedrock before running. The one-time index build
embeds the whole merged corpus (Bedrock Cohere) and is cached on disk keyed by the
frozen-set hash; workers then load that index (no per-worker re-embedding).

Usage
-----
    # one-time (needs HuggingFace network access): select + freeze the 1,000 set
    python local_example/benchmark_musique_1000.py --rebuild-set

    # run 1,000 questions (defaults to 20 workers), run index 0
    python local_example/benchmark_musique_1000.py --run-index 0

    # interrupted? re-run the identical command; only missing queries are executed
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make ``agrag`` (under src/) and ``benchmark_musique`` (this dir) importable whether
# the script is launched by path or as a module, in the parent and in spawned workers.
for _p in (os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "local_example")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the existing runner's workflow verbatim -- no logic copied.
import benchmark_musique as bm  # noqa: E402
from benchmark_musique import (  # noqa: E402
    CONFIG,
    DATASET,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    _cost_summary,
    _evidence_provenance,
    _quality_scores,
    build_evaluator,
    build_global_corpus,
    configure_agent_overrides,
    run_query,
)
from build_musique_eval_set import _freeze_row  # noqa: E402

from agrag.agrag import AutoGluonRAG  # noqa: E402
from agrag.constants import LOGGER_NAME  # noqa: E402
from agrag.evaluation.datasets.musique.musique import (  # noqa: E402
    get_musique_answerable,
    get_musique_evidence_facts,
    get_musique_query,
    get_musique_question_type,
    get_musique_responses,
)
from agrag.evaluation.llm_judge import AnswerJudge  # noqa: E402
from agrag.evaluation.retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics_for_query  # noqa: E402
from agrag.evaluation.utils import f1_metric, inclusive_exact_match_metric  # noqa: E402
from agrag.modules.generator.generator import GeneratorModule  # noqa: E402

logger = logging.getLogger(LOGGER_NAME)

EVAL_DIR = "local_example/evaluation_data_musique"
FROZEN_SET = os.path.join(EVAL_DIR, "musique_eval_set.1000.jsonl")
FROZEN_IDS = os.path.join(EVAL_DIR, "musique_eval_set.1000.ids.json")
# Cache root for the one-time merged corpus + saved vector index (keyed by set hash).
INDEX_CACHE_ROOT = os.path.join(EVAL_DIR, "musique1000_index")
RUNS_ROOT = os.path.join(EVAL_DIR, "musique1000_runs")
DEFAULT_NUM_QUESTIONS = 1000
MODES = (("standard", "standard"), ("agentic", "agentic"))


# --------------------------------------------------------------------------------
# Hashing helpers (first use of hashlib in this repo).
# --------------------------------------------------------------------------------
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_texts(texts):
    """Order-independent hash over a set of corpus doc strings."""
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _safe_id(qid):
    """Filesystem-safe per-query result filename stem."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(qid))


# --------------------------------------------------------------------------------
# 1,000-question selection: proportional to the full dev-set hop ratio.
# --------------------------------------------------------------------------------
def _allocate_proportional(n, bucket_sizes):
    """Largest-remainder allocation of ``n`` across buckets proportional to size.

    Guarantees the allocations sum to exactly ``min(n, total)`` and never exceed a
    bucket's available rows; any shortfall from clamping is redistributed to buckets
    that still have room.
    """
    total = sum(bucket_sizes.values())
    n = min(n, total)
    raw = {k: n * v / total for k, v in bucket_sizes.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(alloc.values())
    # Hand out the leftover to the largest fractional parts (deterministic tie-break).
    order = sorted(bucket_sizes, key=lambda k: (raw[k] - alloc[k], bucket_sizes[k], k), reverse=True)
    i = 0
    while remaining > 0 and order:
        k = order[i % len(order)]
        if alloc[k] < bucket_sizes[k]:
            alloc[k] += 1
            remaining -= 1
        i += 1
        if i > 10 * len(order) and remaining > 0:
            break  # all buckets full (n == total); nothing more to give
    return alloc


def select_proportional_hop_sample(ds, n, seed, answerable_only):
    """Reproducibly select ``n`` row indices preserving the full-set 2/3/4-hop ratio.

    Eligibility reuses the same adapters as the runner (a usable gold answer, and
    answerable when requested). Rows are bucketed by ``get_musique_question_type``;
    ``n`` is split across buckets proportional to each bucket's share of the full
    dev set (not round-robin/equal like the built-in ``--stratify``). Each bucket is
    shuffled with the fixed ``seed`` and its quota taken. Returns (sorted indices,
    info) where ``info`` records the full-set counts and the sampled counts.
    """
    buckets = {}
    for idx in range(len(ds)):
        row = ds[idx]
        if not get_musique_responses(row):
            continue
        if answerable_only and not get_musique_answerable(row):
            continue
        buckets.setdefault(get_musique_question_type(row), []).append(idx)

    full_counts = {k: len(v) for k, v in buckets.items()}
    alloc = _allocate_proportional(n, full_counts)

    rng = random.Random(seed)
    selected = []
    for qtype in sorted(buckets):
        pool = list(buckets[qtype])
        rng.shuffle(pool)
        selected.extend(pool[: alloc[qtype]])

    info = {
        "full_devset_counts": dict(sorted(full_counts.items())),
        "full_devset_total_eligible": sum(full_counts.values()),
        "sampled_counts": {k: alloc[k] for k in sorted(alloc)},
        "sampled_total": len(selected),
    }
    return sorted(selected), info


def build_frozen_set(num_questions, seed, split, answerable_only):
    """Select + freeze the fixed 1,000-question set and its ID list to disk (needs HF)."""
    from datasets import load_dataset

    print(f"Loading {DATASET} split '{split}' from HuggingFace to (re)build the frozen set ...")
    ds = load_dataset(DATASET, split=split)
    indices, info = select_proportional_hop_sample(ds, num_questions, seed, answerable_only)
    if not indices:
        raise SystemExit("No eligible rows selected; nothing to freeze.")

    os.makedirs(os.path.dirname(os.path.abspath(FROZEN_SET)), exist_ok=True)
    ids = []
    with open(FROZEN_SET, "w", encoding="utf-8") as f:
        for idx in indices:
            row = ds[idx]
            frozen = _freeze_row(row)
            frozen["_source_index"] = idx
            f.write(json.dumps(frozen, default=str) + "\n")
            ids.append(row["id"])
    with open(FROZEN_IDS, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "split": split, "count": len(ids), "ids": ids, "selection": info}, f, indent=2)

    print(f"\nFroze {len(ids)} MuSiQue rows -> {FROZEN_SET}")
    print(f"  full dev-set hop counts : {info['full_devset_counts']}  (total {info['full_devset_total_eligible']})")
    print(f"  sampled hop counts      : {info['sampled_counts']}  (total {info['sampled_total']})")
    ratio = {k: round(v / info["full_devset_total_eligible"], 3) for k, v in info["full_devset_counts"].items()}
    sratio = {k: round(v / info["sampled_total"], 3) for k, v in info["sampled_counts"].items()}
    print(f"  full ratio {ratio}  vs  sampled ratio {sratio}")
    return info


def load_frozen_rows():
    if not os.path.exists(FROZEN_SET):
        raise SystemExit(
            f"Frozen set {FROZEN_SET} not found. Build it once with:\n"
            f"    python local_example/benchmark_musique_1000.py --rebuild-set"
        )
    rows = []
    with open(FROZEN_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------------
# One-time corpus + index build (cached on disk; workers load, never rebuild).
# --------------------------------------------------------------------------------
def _cache_paths(dataset_hash):
    cache_dir = os.path.join(INDEX_CACHE_ROOT, dataset_hash[:16])
    return {
        "cache_dir": cache_dir,
        "corpus_dir": os.path.join(cache_dir, "corpus"),
        "index_path": os.path.join(cache_dir, "vector.index"),
        "metadata_path": os.path.join(cache_dir, "metadata.pkl"),
    }


def build_or_load_index(rows, dataset_hash):
    """Merge the global corpus and build+save the vector index once (cached).

    Returns the cache paths plus the corpus hash. If a saved index already exists for
    this dataset hash, the expensive embed-and-index step is skipped and the existing
    index is reused; the corpus dir is always (re)materialized (cheap file writes) so
    workers have the raw paragraphs on disk that the loaded index refers to.
    """
    paths = _cache_paths(dataset_hash)
    os.makedirs(paths["cache_dir"], exist_ok=True)

    # Always (re)materialize the merged global corpus dir; capture the corpus hash.
    doc_to_global = build_global_corpus(rows, paths["corpus_dir"])
    corpus_hash = _sha256_texts(doc_to_global.keys())

    index_exists = os.path.isfile(paths["index_path"]) and os.path.isfile(paths["metadata_path"])
    if index_exists:
        print(f"Reusing cached vector index at {paths['index_path']}")
        return paths, corpus_hash

    print("Building the merged-corpus vector index once (embeds the whole corpus via Bedrock) ...")
    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=paths["corpus_dir"])
    configure_agent_overrides(agrag)
    agrag.args.use_existing_vector_db_index = False
    agrag.args.save_vector_db_index = True
    agrag.args.vector_db_index_save_path = paths["index_path"]
    agrag.args.metadata_index_save_path = paths["metadata_path"]
    # ``build_global_corpus`` already exact-string deduplicates paragraphs, so the
    # vector DB's cosine-similarity dedup is redundant here -- and at merged-corpus
    # scale (~15-20k paras) its O(n^2) similarity matrix is a multi-GB memory/compute
    # spike. A threshold of 1.0 turns that pass off (see remove_duplicates early-out).
    agrag.args.vector_db_sim_threshold = 1.0
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()
    print(f"Saved vector index -> {paths['index_path']}")
    return paths, corpus_hash


# --------------------------------------------------------------------------------
# Worker: own AutoGluonRAG per process, loads the cached index, answers both modes.
# --------------------------------------------------------------------------------
_WORKER = {}


def _build_judge_generator(agrag, judge_model):
    """Standalone GeneratorModule for the LLM judge, isolated from the pipeline
    generator so the judge model (Opus 4.8) can differ from the Sonnet generator.

    Reuses the generator's platform + bedrock params (region, temperature=0) so the
    judge decodes deterministically in the same region. Opus 4.8 rejects the
    ``temperature`` param; ``BedrockGenerator._handle_unsupported_param`` drops it and
    retries, so no special-casing is needed here.
    """
    return GeneratorModule(
        model_name=judge_model,
        model_platform=agrag.args.generator_model_platform,
        platform_args=dict(agrag.args.generator_model_platform_args or {}),
    )


def _init_worker(corpus_dir, index_path, metadata_path, use_judge, judge_model, results_dir, log_path):
    """ProcessPoolExecutor initializer: build ONE pipeline per worker (loads index)."""
    os.chdir(REPO_ROOT)
    # Route this worker's logs (incl. Bedrock throttling-retry warnings) to run.log.
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [pid %(process)d] %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    configure_agent_overrides(agrag)
    agrag.args.use_existing_vector_db_index = True  # load the pre-built index; no re-embedding
    agrag.args.save_vector_db_index = False
    agrag.args.vector_db_index_load_path = index_path
    agrag.args.metadata_index_load_path = metadata_path
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    _WORKER["agrag"] = agrag
    _WORKER["judge"] = AnswerJudge(_build_judge_generator(agrag, judge_model)) if use_judge else None
    _WORKER["results_dir"] = results_dir
    _WORKER["run_index"] = None  # set per-task via payload


def _score_one(prediction, references):
    """Per-query deterministic verdicts, reusing the exact metric primitives."""
    incl = inclusive_exact_match_metric(
        predictions=[prediction], references=[references], ignore_case=True, ignore_punctuation=True, substring=True
    )
    strict = inclusive_exact_match_metric(
        predictions=[prediction], references=[references], ignore_case=True, ignore_punctuation=True, substring=False
    )
    f1 = f1_metric([prediction], [references])
    return {
        "inclusive_em": bool(incl[0]),
        "strict_em": bool(strict[0]),
        "token_f1": round(float(f1[0]), 4),
    }


def _process_query(payload):
    """Answer one MuSiQue row in BOTH modes, score it, and save it atomically.

    Runs in a worker process. Returns a small ``(qid, ok, error)`` tuple; the full
    (bulky) per-query record is written straight to disk here to avoid shipping the
    traces back through IPC. The result file is written only after BOTH modes finish,
    via a temp file + ``os.replace`` -- so a crash mid-query never leaves a partial
    file that resume would mistake for done.
    """
    agrag = _WORKER["agrag"]
    judge = _WORKER["judge"]
    results_dir = _WORKER["results_dir"]
    row, run_index = payload
    qid = row["id"]
    try:
        query = get_musique_query(row)
        references = get_musique_responses(row)
        qtype = get_musique_question_type(row)
        answerable = get_musique_answerable(row)
        gold_facts = get_musique_evidence_facts(row)

        record = {
            "id": qid,
            "run_index": run_index,
            "source_index": row.get("_source_index"),
            "question_type": qtype,
            "answerable": answerable,
            "query": query,
            "references": references,
            "has_gold_facts": bool(gold_facts),
            "modes": {},
        }
        for label, mode_arg in MODES:
            run = run_query(agrag, query, mode_arg)
            verdicts = _score_one(run["answer"], references)
            if judge is not None:
                try:
                    verdicts["llm_judge"] = bool(judge.judge(query, run["answer"], references))
                except Exception as exc:  # noqa: BLE001 -- a flaky judge must not fail the query
                    logger.warning("LLM judge failed on %s/%s (%s); recording None", qid, label, exc)
                    verdicts["llm_judge"] = None
            else:
                verdicts["llm_judge"] = None
            rmetrics = retrieval_metrics_for_query(run["evidence_texts"], gold_facts) if gold_facts else {}
            record["modes"][label] = {
                "prediction": run["answer"],
                "evidence_texts": run["evidence_texts"],
                "evidence_provenance": _evidence_provenance(run["evidence"]),
                "retrieval_metrics": rmetrics,
                "latency_s": round(run["latency"], 4),
                "agent_metrics": run["agent_metrics"],
                "verdicts": verdicts,
                "trace": run["trace"],
            }

        out_path = os.path.join(results_dir, f"{_safe_id(qid)}.json")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(record, f, default=str)
        os.replace(tmp_path, out_path)  # atomic: file appears complete or not at all
        return qid, True, None
    except Exception as exc:  # noqa: BLE001 -- one bad query must not kill the pool
        logger.exception("Query %s failed", qid)
        return qid, False, repr(exc)


# --------------------------------------------------------------------------------
# Aggregation: reuse _quality_scores; llm_judge folded in from saved verdicts.
# --------------------------------------------------------------------------------
def _inject_llm_judge(scores, judge_verdicts, judge_enabled):
    """Add llm_judge accuracy from saved per-query verdicts, matching judge_matches.

    A failed judge call (None) is excluded from the denominator and counted as a
    failure -- identical to how ``_quality_scores`` reports the judge metric.
    """
    if not judge_enabled:
        return scores
    scored = [v for v in judge_verdicts if v is not None]
    failures = len(judge_verdicts) - len(scored)
    scores["llm_judge"] = round(sum(scored) / len(scored), 4) if scored else None
    scores["llm_judge_scored"] = len(scored)
    scores["llm_judge_failures"] = failures
    return scores


def _score_group(evaluator, records, mode, judge_enabled):
    """Compute the _quality_scores + retrieval bundle for one mode over ``records``."""
    preds, refs, queries, retrieval, judge_verdicts = [], [], [], [], []
    for rec in records:
        m = rec["modes"][mode]
        preds.append(m["prediction"])
        refs.append(rec["references"])
        queries.append(rec["query"])
        judge_verdicts.append(m["verdicts"].get("llm_judge"))
        if m.get("retrieval_metrics"):
            retrieval.append(m["retrieval_metrics"])
    quality = _quality_scores(evaluator, preds, refs, queries, judge=None)
    _inject_llm_judge(quality, judge_verdicts, judge_enabled)
    return {"quality": quality, "retrieval": aggregate_retrieval_metrics(retrieval)}


def aggregate(evaluator, records, judge_enabled):
    """Build the summary dict in the same shape as benchmark_results.json."""
    results = {}
    for label, _ in MODES:
        by_type = {}
        for qtype in sorted({r["question_type"] for r in records}):
            group = [r for r in records if r["question_type"] == qtype]
            scored = _score_group(evaluator, group, label, judge_enabled)
            by_type[qtype] = {"count": len(group), **scored}

        overall = _score_group(evaluator, records, label, judge_enabled)
        latencies = [r["modes"][label]["latency_s"] for r in records]
        mode_result = {
            "quality_overall": overall["quality"],
            "retrieval_overall": overall["retrieval"],
            "quality_by_question_type": by_type,
            "cost": _cost_summary(latencies),
        }
        if label == "agentic":
            agent_runs = [r["modes"][label]["agent_metrics"] for r in records if r["modes"][label]["agent_metrics"]]
            behavior = bm._agentic_behavior_summary(agent_runs)
            if behavior is not None:
                mode_result["agentic_behavior"] = behavior
        results[label] = mode_result
    return results


def load_completed(results_dir):
    """IDs already fully saved (only .json files count; .tmp = interrupted mid-write)."""
    done = {}
    for name in os.listdir(results_dir):
        if name.endswith(".json") and not name.endswith(".tmp"):
            path = os.path.join(results_dir, name)
            try:
                with open(path) as f:
                    rec = json.load(f)
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                logger.warning("Ignoring unreadable result file %s", path)
    return done


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    os.chdir(REPO_ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help=(
            "Parallel worker PROCESSES (default: 20). Agentic mode issues 10-20+ "
            "sequential Bedrock calls per question, so the binding constraint is "
            "Bedrock throttling, not CPU. Watch run.log for "
            "'Bedrock throttling retry' lines -- if they are frequent, lower this "
            "(4 is a safe start; 6-8 if throttling is rare)."
        ),
    )
    parser.add_argument("--run-index", type=int, default=0, help="Run index; each run gets its own folder.")
    parser.add_argument(
        "--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS, help="Frozen set size to build (default: 1000)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Selection seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"MuSiQue split when rebuilding (default: {DEFAULT_SPLIT}).")
    parser.add_argument("--answerable-only", action="store_true", help="Keep only answerable rows when selecting.")
    parser.add_argument(
        "--rebuild-set",
        action="store_true",
        help="(Re)select + freeze the 1,000-question set from HuggingFace, then exit.",
    )
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Supplementary model-based llm_judge accuracy alongside EM/F1 (default: on).",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="us.anthropic.claude-opus-4-8",
        help=(
            "Bedrock model id for the LLM judge, separate from the Sonnet generator "
            "so evaluation is independent of generation (default: Claude Opus 4.8)."
        ),
    )
    args = parser.parse_args()

    if args.rebuild_set:
        build_frozen_set(args.num_questions, args.seed, args.split, args.answerable_only)
        return

    # ---- Load the fixed frozen set ------------------------------------------------
    rows = load_frozen_rows()
    rows = [r for r in rows if get_musique_responses(r)]
    if args.answerable_only:
        rows = [r for r in rows if get_musique_answerable(r)]
    if not rows:
        raise SystemExit("No eligible questions in the frozen set.")
    dataset_hash = _sha256_file(FROZEN_SET)
    print(f"Loaded {len(rows)} frozen MuSiQue questions (dataset_hash={dataset_hash[:16]}).")

    # ---- One-time merged corpus + vector index (cached; workers load it) ---------
    paths, corpus_hash = build_or_load_index(rows, dataset_hash)

    # ---- Per-run folder ----------------------------------------------------------
    run_dir = os.path.join(RUNS_ROOT, f"run_{args.run_index:03d}")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [pid %(process)d] %(levelname)s %(message)s"))
    logging.getLogger(LOGGER_NAME).addHandler(fh)
    logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)

    # ---- Resume: run only the missing query IDs ----------------------------------
    completed = load_completed(results_dir)
    selected_ids = [r["id"] for r in rows]
    todo = [r for r in rows if r["id"] not in completed]
    print(f"Resume: {len(completed)} already done, {len(todo)} remaining of {len(rows)}.")

    # ---- Manifest (full provenance), written before execution --------------------
    manifest = {
        "run_index": args.run_index,
        "workers": args.workers,
        "seed": args.seed,
        "num_selected": len(rows),
        "selected_ids": selected_ids,
        "dataset": DATASET,
        "split": args.split,
        "dataset_hash": dataset_hash,
        "corpus_hash": corpus_hash,
        "config": {
            "config_file": CONFIG,
            "temperature": 0,
            "use_reranker": False,
            "llm_judge": args.llm_judge,
            "llm_judge_model": args.llm_judge_model if args.llm_judge else None,
            "answerable_only": args.answerable_only,
            "agent_overrides_env": {
                "MUSIQUE_RECOVERY": os.environ.get("MUSIQUE_RECOVERY", "1"),
                "MUSIQUE_MAX_SUBQUERIES": os.environ.get("MUSIQUE_MAX_SUBQUERIES", "6"),
                "MUSIQUE_MAX_TOTAL_HOPS": os.environ.get("MUSIQUE_MAX_TOTAL_HOPS", "12"),
                "MUSIQUE_MAX_WALL_CLOCK_S": os.environ.get("MUSIQUE_MAX_WALL_CLOCK_S", "0"),
                "MUSIQUE_MAX_LLM_CALLS": os.environ.get("MUSIQUE_MAX_LLM_CALLS", "0"),
                "MUSIQUE_MAX_RETRIEVAL_CALLS": os.environ.get("MUSIQUE_MAX_RETRIEVAL_CALLS", "0"),
                "MUSIQUE_DEEP_TOPK": os.environ.get("MUSIQUE_DEEP_TOPK", "0"),
                "MUSIQUE_ENTITY_GATE": os.environ.get("MUSIQUE_ENTITY_GATE", "0"),
                "MUSIQUE_DROP_PREFIX": os.environ.get("MUSIQUE_DROP_PREFIX", "0"),
                "MUSIQUE_DIRECT_ANSWER_FALLBACK": os.environ.get("MUSIQUE_DIRECT_ANSWER_FALLBACK", "0"),
                "MUSIQUE_VERIFIER_MODEL": os.environ.get("MUSIQUE_VERIFIER_MODEL", "us.anthropic.claude-opus-4-8"),
                "MUSIQUE_DUAL_SYNTHESIS": os.environ.get("MUSIQUE_DUAL_SYNTHESIS", "1"),
                "MUSIQUE_SELF_CONSISTENCY": os.environ.get("MUSIQUE_SELF_CONSISTENCY", "1"),
            },
        },
        "index_cache": paths["cache_dir"],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # ---- Parallel execution ------------------------------------------------------
    if todo:
        # 'spawn' avoids fork-safety issues with boto3/torch threads: each worker
        # builds its own AutoGluonRAG (and Bedrock client) fresh, post-spawn.
        ctx = multiprocessing.get_context("spawn")
        payloads = [(r, args.run_index) for r in todo]
        start = time.perf_counter()
        done_n, fail_n = 0, 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(paths["corpus_dir"], paths["index_path"], paths["metadata_path"], args.llm_judge, args.llm_judge_model, results_dir, log_path),
        ) as pool:
            futures = [pool.submit(_process_query, p) for p in payloads]
            for fut in as_completed(futures):
                qid, ok, err = fut.result()
                done_n += 1
                if not ok:
                    fail_n += 1
                    print(f"  [{done_n}/{len(todo)}] FAILED {qid}: {err}")
                elif done_n % 10 == 0 or done_n == len(todo):
                    elapsed = time.perf_counter() - start
                    rate = done_n / elapsed if elapsed else 0
                    print(f"  [{done_n}/{len(todo)}] done ({rate:.2f} q/s, {fail_n} failed)")
        print(f"Execution finished: {done_n} processed this run ({fail_n} failed).")
    else:
        print("Nothing to run; all queries already completed. Re-aggregating.")

    # ---- Aggregate (idempotent; reads every saved per-query result) --------------
    records = list(load_completed(results_dir).values())
    print(f"Aggregating {len(records)} completed queries ...")
    # A fresh pipeline just for the evaluator's EM metric (pure-Python metric; the
    # generator is only used by _quality_scores if a judge is passed -- it is not).
    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=paths["corpus_dir"])
    configure_agent_overrides(agrag)
    agrag.args.use_existing_vector_db_index = True
    agrag.args.save_vector_db_index = False
    agrag.args.vector_db_index_load_path = paths["index_path"]
    agrag.args.metadata_index_load_path = paths["metadata_path"]
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()
    evaluator = build_evaluator(agrag)

    results = aggregate(evaluator, records, args.llm_judge)
    results["selection"] = {
        "dataset": DATASET,
        "selection_source": "frozen",
        "eval_set": FROZEN_SET,
        "dataset_hash": dataset_hash,
        "corpus_hash": corpus_hash,
        "seed": args.seed,
        "answerable_only": args.answerable_only,
        "num_selected": len(rows),
        "num_aggregated": len(records),
        "run_index": args.run_index,
        "workers": args.workers,
        "llm_judge_model": args.llm_judge_model if args.llm_judge else None,
    }

    print("\n" + "=" * 72)
    print(f"SUMMARY  (standard vs. agentic, n={len(records)})")
    print("=" * 72)
    print(json.dumps({k: v for k, v in results.items() if k != "selection"}, indent=2, default=str))

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")
    print(f"Per-query results in {results_dir}")


if __name__ == "__main__":
    main()
