import unittest

from agrag.modules.agentic.evidence import Evidence, EvidenceStore


class TestEvidence(unittest.TestCase):
    def test_from_retrieval_record_maps_fields_and_keeps_extra(self):
        record = {
            "text": "AutoGluon-RAG is a framework.",
            "doc_id": 0,
            "chunk_id": 2,
            "retrieval_score": 0.91,
            "extra_field": "keepme",
        }
        ev = Evidence.from_retrieval_record(
            record, retrieval_query="what is autogluon-rag", rank=1, tool_name="RetrieveTool"
        )
        self.assertEqual(ev.doc_id, 0)
        self.assertEqual(ev.chunk_id, 2)
        self.assertEqual(ev.rank, 1)
        self.assertEqual(ev.retrieval_query, "what is autogluon-rag")
        self.assertEqual(ev.tool_name, "RetrieveTool")
        self.assertEqual(ev.retrieval_score, 0.91)
        self.assertEqual(ev.metadata, {"extra_field": "keepme"})

    def test_citation_prefers_source_then_ids(self):
        self.assertEqual(Evidence(text="x", source="file.pdf").citation(), "file.pdf")
        self.assertEqual(Evidence(text="x", doc_id=1, chunk_id=3).citation(), "doc 1, chunk 3")
        self.assertEqual(Evidence(text="x", evidence_id="e5").citation(), "e5")

    def test_dedup_key_uses_ids_then_text(self):
        self.assertEqual(Evidence(text="a", doc_id=1, chunk_id=0).dedup_key(), (1, 0))
        self.assertEqual(Evidence(text="a").dedup_key(), ("text", "a"))


class TestEvidenceStore(unittest.TestCase):
    def test_add_dedups_by_doc_and_chunk(self):
        store = EvidenceStore()
        first = Evidence(text="chunk", doc_id=0, chunk_id=0)
        dup = Evidence(text="chunk (other query)", doc_id=0, chunk_id=0)
        self.assertTrue(store.add(first))
        self.assertFalse(store.add(dup))
        self.assertEqual(len(store), 1)
        self.assertEqual(first.evidence_id, "e0")

    def test_add_dedups_by_text_when_no_ids(self):
        store = EvidenceStore()
        self.assertTrue(store.add(Evidence(text="same")))
        self.assertFalse(store.add(Evidence(text="same")))
        self.assertEqual(len(store), 1)

    def test_add_many_returns_stored_count(self):
        store = EvidenceStore()
        items = [
            Evidence(text="a", doc_id=0, chunk_id=0),
            Evidence(text="b", doc_id=0, chunk_id=1),
            Evidence(text="dup", doc_id=0, chunk_id=0),
        ]
        self.assertEqual(store.add_many(items), 2)
        self.assertEqual(len(store), 2)

    def test_mark_used_and_texts(self):
        store = EvidenceStore()
        store.add(Evidence(text="a", doc_id=0, chunk_id=0))
        store.add(Evidence(text="b", doc_id=0, chunk_id=1))
        store.mark_used(["e0"])
        self.assertTrue(store.get("e0").used_in_answer)
        self.assertFalse(store.get("e1").used_in_answer)
        self.assertEqual(store.texts(), ["a", "b"])

    def test_to_list_serializes(self):
        store = EvidenceStore()
        store.add(Evidence(text="a", doc_id=0, chunk_id=0))
        serialized = store.to_list()
        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0]["evidence_id"], "e0")
        self.assertEqual(serialized[0]["text"], "a")


if __name__ == "__main__":
    unittest.main()
