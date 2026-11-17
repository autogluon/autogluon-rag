"""Freeze a small, reproducible slice of real MuSiQue into a self-contained JSONL.

The MuSiQue benchmark runner (``benchmark_musique.py``) needs MuSiQue rows. Pulling the
full ``dgslibisey/MuSiQue`` dataset from HuggingFace on every run is slow and
requires network access, and the exact rows can drift if the mirror changes. This
script selects a deterministic, hop-count-stratified sample of real MuSiQue rows
once and writes them verbatim to a JSONL file, so downstream evaluation is offline
and reproducible.

Each output row is a real MuSiQue example, carrying its full native schema so the
existing adapters (``agrag.evaluation.datasets.musique.musique``) apply to it
unchanged:

* ``id``                     -- hop-count is encoded in the prefix (``2hop__...``).
* ``question`` / ``answer`` / ``answer_aliases``
* ``answerable``             -- whether the answer is present in the paragraphs.
* ``paragraphs``             -- the per-question corpus (supporting + distractors),
                                each ``{idx, title, paragraph_text, is_supporting}``.
* ``question_decomposition`` -- the gold reasoning hops.

Selection reuses ``select_query_indices`` from ``benchmark_musique`` so the frozen
sample matches what that runner would pick (same seed, same stratification),
keeping the two entry points consistent.

Usage
-----
    python local_example/build_musique_eval_set.py --size 30 --stratify
    # -> local_example/evaluation_data_musique/musique_eval_set.jsonl

Run this once (needs network / HuggingFace access). After that the evaluator can
run fully offline against the frozen file.
"""

import argparse
import json
import os

from datasets import load_dataset

# Reuse the exact selection logic and constants from the existing runner so the
# frozen set is identical to what benchmark_musique would sample.
from benchmark_musique import DATASET, DEFAULT_SEED, DEFAULT_SPLIT, select_query_indices

from agrag.evaluation.datasets.musique.musique import (
    get_musique_answerable,
    get_musique_question_type,
)

# The native MuSiQue fields we preserve verbatim. Keeping the full schema means
# the downstream adapters (query / responses / paragraph docs / evidence facts /
# answerable / hop-type) all work on a frozen row exactly as on a live HF row.
_PRESERVED_FIELDS = (
    "id",
    "question",
    "answer",
    "answer_aliases",
    "answerable",
    "paragraphs",
    "question_decomposition",
)

DEFAULT_OUT = "local_example/evaluation_data_musique/musique_eval_set.jsonl"


def _freeze_row(row):
    """Copy only the preserved native fields from a MuSiQue row (verbatim)."""
    return {field: row[field] for field in _PRESERVED_FIELDS if field in row}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=30, help="Number of rows to freeze (default: 30).")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"MuSiQue split (default: {DEFAULT_SPLIT}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Selection seed (default: {DEFAULT_SEED}).")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify the sample by hop-count question type (recommended for a balanced 2/3/4-hop set).",
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Keep only answerable rows (default: keep unanswerable rows too, for the MuSiQue-Full setting).",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output JSONL path (default: {DEFAULT_OUT}).")
    args = parser.parse_args()

    print(f"Loading {DATASET} split '{args.split}' ...")
    queries_ds = load_dataset(DATASET, split=args.split)

    selected = select_query_indices(
        queries_ds, args.size, seed=args.seed, answerable_only=args.answerable_only, stratify=args.stratify
    )
    if not selected:
        raise SystemExit("No eligible rows selected; nothing to freeze.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    type_counts, answerable_counts = {}, {"answerable": 0, "unanswerable": 0}
    with open(args.out, "w", encoding="utf-8") as f:
        for idx in selected:
            row = queries_ds[idx]
            frozen = _freeze_row(row)
            # Record provenance so a frozen set is traceable back to its source rows.
            frozen["_source_index"] = idx
            f.write(json.dumps(frozen, default=str) + "\n")

            qtype = get_musique_question_type(row)
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            answerable_counts["answerable" if get_musique_answerable(row) else "unanswerable"] += 1

    print(f"\nFroze {len(selected)} real MuSiQue rows -> {args.out}")
    print(f"  seed={args.seed}  split={args.split}  stratify={args.stratify}  answerable_only={args.answerable_only}")
    print(f"  by hop type: {dict(sorted(type_counts.items()))}")
    print(f"  answerability: {answerable_counts}")
    print("\nThis file is self-contained; benchmark_musique.py can now run offline against it.")


if __name__ == "__main__":
    main()
