"""Integration tests for end-to-end parent-child retrieval wiring.

These exercise ``AutoGluonRAG.initialize_rag_pipeline`` with the real data
processing, vector DB, and retriever modules, stubbing only the heavy
embedding/reranker/generator model loads. They verify that the parent store is
built, attached to the retriever, and usable for expansion on the first query --
for both the non-batched and batched code paths.
"""

import hashlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import yaml

from agrag.agrag import AutoGluonRAG
from agrag.constants import DOC_TEXT_KEY, EMBEDDING_HIDDEN_DIM_KEY, EMBEDDING_KEY, PARENT_ID_KEY

EMBED_DIM = 8


def _embed(text: str) -> np.ndarray:
    """Deterministic per-text embedding: identical text -> identical vector, so a
    query equal to a chunk's text is an exact (L2 distance 0) match."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return np.frombuffer(digest[:EMBED_DIM], dtype=np.uint8).astype("float32")


class FakeEmbeddingModule:
    """Lightweight stand-in for EmbeddingModule; no model download."""

    model_platform = "huggingface"
    model_name = "fake-embedder"

    def encode(self, data: pd.DataFrame, pbar=None, batch_size: int = 32) -> pd.DataFrame:
        data = data.copy()
        embeddings = [_embed(text) for text in data[DOC_TEXT_KEY].tolist()]
        data[EMBEDDING_KEY] = embeddings
        data[EMBEDDING_HIDDEN_DIM_KEY] = [EMBED_DIM] * len(embeddings)
        return data


def _write_config(tmp_dir: str, data_dir: str, batch_size: int = 0) -> str:
    config = {
        "shared": {"pipeline_batch_size": batch_size},
        "data": {
            "data_dir": data_dir,
            "chunk_size": 5,
            "chunk_overlap": 0,
            "chunking_strategy": "parent_child",
            "children_per_parent": 2,
            "file_extns": [".txt"],
            "parse_urls_recursive": False,
        },
        "vector_db": {
            "db_type": "faiss",
            # threshold 1.0 keeps every chunk (nothing treated as a duplicate),
            # so metadata rows align 1:1 with the indexed vectors.
            "similarity_threshold": 1.0,
            "similarity_fn": "cosine",
            "faiss_index_type": "IndexFlatL2",
            "save_index": False,
            "use_existing_vector_db": False,
            "num_gpus": 0,
        },
        "retriever": {"use_reranker": False, "retriever_top_k": 4},
    }
    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def _build_rag(config_path: str) -> AutoGluonRAG:
    rag = AutoGluonRAG(config_file=config_path)
    # Stub the heavy model-loading initializers; keep data/vectordb/retriever real.
    rag.initialize_embeddings_module = lambda: setattr(rag, "embedding_module", FakeEmbeddingModule())
    rag.initialize_reranker_module = lambda: setattr(rag, "reranker_module", None)
    rag.initialize_generator_module = lambda: setattr(rag, "generator_module", MagicMock())
    return rag


class TestParentChildPipelineNonBatched(unittest.TestCase):
    def test_non_batched_attaches_store_and_expands_on_first_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "docs")
            os.makedirs(data_dir)
            # 4 distinct 5-char chunks -> 2 parents (children_per_parent=2).
            with open(os.path.join(data_dir, "doc.txt"), "w") as f:
                f.write("AAAAABBBBBCCCCCDDDDD")

            rag = _build_rag(_write_config(tmp, data_dir, batch_size=0))
            rag.initialize_rag_pipeline()

            # The parent store built during processing is attached to the retriever
            # and its parent cache was reset (still lazy -> None).
            self.assertIsNotNone(rag.parent_store)
            self.assertEqual(len(rag.parent_store), 2)
            self.assertIs(rag.retriever_module.parent_store, rag.parent_store)
            self.assertIsNone(rag.retriever_module._parent_text_cache)

            # First query: an exact child match expands to its parent context.
            records = rag.retriever_module.retrieve("AAAAA", return_metadata=True)
            top = records[0]
            self.assertEqual(top["child_text"], "AAAAA")
            # Parent 0 groups children "AAAAA" and "BBBBB"; expansion pulls both in.
            self.assertIn("AAAAA", top["text"])
            self.assertIn("BBBBB", top["text"])


class TestParentChildPipelineBatched(unittest.TestCase):
    def test_batched_accumulates_store_with_globally_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "docs")
            os.makedirs(data_dir)
            # Two files, each 2 chunks -> 1 parent per file. batch_size=1 forces
            # two separate batches so we can verify cross-batch ID uniqueness.
            with open(os.path.join(data_dir, "a.txt"), "w") as f:
                f.write("AAAAABBBBB")
            with open(os.path.join(data_dir, "b.txt"), "w") as f:
                f.write("CCCCCDDDDD")

            rag = _build_rag(_write_config(tmp, data_dir, batch_size=1))
            rag.initialize_rag_pipeline()

            # Store accumulated across both batches with unique parent_ids.
            self.assertIsNotNone(rag.parent_store)
            self.assertEqual(len(rag.parent_store), 2)
            self.assertEqual(sorted(rag.parent_store[PARENT_ID_KEY].tolist()), [0, 1])
            # Wired into the retriever after batched processing completes.
            self.assertIs(rag.retriever_module.parent_store, rag.parent_store)
            # Metadata (indexed children) also carries globally unique parent_ids.
            self.assertEqual(
                sorted(rag.vector_db_module.metadata[PARENT_ID_KEY].unique().tolist()), [0, 1]
            )


class TestLegacyPipelineNoParentStore(unittest.TestCase):
    def test_legacy_flat_chunking_leaves_parent_store_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "docs")
            os.makedirs(data_dir)
            with open(os.path.join(data_dir, "doc.txt"), "w") as f:
                f.write("AAAAABBBBB")

            config_path = _write_config(tmp, data_dir, batch_size=0)
            # Flip to the default legacy/flat strategy.
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["data"]["chunking_strategy"] = "legacy"
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)

            rag = _build_rag(config_path)
            rag.initialize_rag_pipeline()

            # No parent store; the retriever stays on the dense-only legacy path.
            self.assertIsNone(rag.parent_store)
            self.assertIsNone(rag.retriever_module.parent_store)
            self.assertNotIn(PARENT_ID_KEY, rag.vector_db_module.metadata.columns)


if __name__ == "__main__":
    unittest.main()
