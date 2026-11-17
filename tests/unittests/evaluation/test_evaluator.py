"""Deterministic, offline tests for EvaluationModule edge cases.

No AWS/OpenAI creds, no network, no model downloads: ``load_dataset`` is patched
to return an in-memory fake dataset, the RAG instance is faked, and a plain
callable metric keeps scoring local.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from agrag.evaluation.evaluator import EvaluationModule


class FakeDataset:
    """Minimal stand-in for a HuggingFace dataset: iterable rows + num_rows."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    @property
    def num_rows(self):
        return len(self._rows)


class FakeRAG:
    """Fake AutoGluonRAG: echoes a deterministic response, needs no pipeline."""

    def __init__(self):
        self.pipeline_initialized = True
        self.calls = []

    def generate_response(self, query, mode=None):
        self.calls.append((query, mode))
        return f"answer:{query}"


def _rows(n):
    return [{"question": f"q{i}", "answers": [f"a{i}"]} for i in range(n)]


def _query_fn(row):
    return row["question"]


def _response_fn(row):
    return row["answers"]


def _dummy_metric(predictions, references, **kwargs):
    """Callable metric: fraction of predictions that contain a reference token."""
    return {"count": len(predictions)}


def _run(evaluator, dataset, **kwargs):
    """Run evaluation with load_dataset patched to return `dataset`."""
    defaults = dict(
        dataset_name="fake/dataset",
        metrics=[_dummy_metric],
        preprocessing_fn=lambda row: row["question"],
        query_fn=_query_fn,
        response_fn=_response_fn,
        save_evaluation_data=False,
    )
    defaults.update(kwargs)
    with patch("agrag.evaluation.evaluator.load_dataset", return_value=dataset):
        return evaluator.run_evaluation(**defaults)


class TestEvaluatorEdgeCases(unittest.TestCase):
    def test_max_eval_size_none_processes_all_rows(self):
        rag = FakeRAG()
        evaluator = EvaluationModule(rag_instance=rag)
        dataset = FakeDataset(_rows(4))

        _run(evaluator, dataset, max_eval_size=None)

        # All rows answered; no TypeError from comparing None >= num_rows.
        self.assertIsNone(evaluator.max_eval_size)
        self.assertEqual(len(rag.calls), 4)

    def test_max_eval_size_respected_when_smaller(self):
        rag = FakeRAG()
        evaluator = EvaluationModule(rag_instance=rag)
        dataset = FakeDataset(_rows(5))

        _run(evaluator, dataset, max_eval_size=2)

        self.assertEqual(evaluator.max_eval_size, 2)
        self.assertEqual(len(rag.calls), 2)

    def test_max_eval_size_too_large_warns_and_clamps(self):
        rag = FakeRAG()
        evaluator = EvaluationModule(rag_instance=rag)
        dataset = FakeDataset(_rows(3))

        with self.assertLogs("agrag.evaluation.evaluator", level="WARNING"):
            _run(evaluator, dataset, max_eval_size=10)

        # Clamped to "process everything".
        self.assertIsNone(evaluator.max_eval_size)
        self.assertEqual(len(rag.calls), 3)

    def test_save_csv_writes_predictions(self):
        rag = FakeRAG()
        evaluator = EvaluationModule(rag_instance=rag)
        dataset = FakeDataset(_rows(3))

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "results.csv")
            captured = {}

            def fake_save(generated_responses, expected_responses, queries, output_csv):
                captured["predictions"] = list(generated_responses)
                captured["references"] = list(expected_responses)
                captured["queries"] = list(queries)
                captured["path"] = output_csv

            with patch("agrag.evaluation.evaluator.save_responses_to_csv", side_effect=fake_save):
                _run(evaluator, dataset, max_eval_size=None, save_csv_path=csv_path)

            # The missing-predictions bug would raise TypeError before this point.
            self.assertEqual(captured["predictions"], ["answer:q0", "answer:q1", "answer:q2"])
            self.assertEqual(captured["references"], [["a0"], ["a1"], ["a2"]])
            self.assertEqual(captured["queries"], ["q0", "q1", "q2"])
            self.assertEqual(captured["path"], csv_path)


if __name__ == "__main__":
    unittest.main()
