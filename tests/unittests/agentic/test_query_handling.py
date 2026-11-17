"""Deterministic tests for query handling in the agentic path.

Covers two correctness guarantees:
1. Synthesis and verification always target the immutable ORIGINAL user query,
   even after a rewrite has changed the working (current) query.
2. The generator query-prefix is applied to answer synthesis in agentic mode
   (matching the standard path) but NOT to verification or query rewriting.

All tests are offline: they use in-memory fakes, no models, network, or creds.
"""

import unittest

from agrag.modules.agentic.agentic_module import AgenticRAGModule


class ScriptedRetriever:
    """Returns records per query; enough evidence for both original and rewrite."""

    def __init__(self, by_query, default=None):
        self.by_query = by_query
        self.default = default or []
        self.queries = []

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.queries.append(query)
        return list(self.by_query.get(query, self.default))


class RecordingGenerator:
    """Records every prompt; scripts rewrite output and verification labels."""

    def __init__(self, verify_labels=None, rewritten="REWRITTEN QUERY", answer="draft answer", model_name="gpt-4"):
        self.verify_labels = list(verify_labels or ["supported"])
        self.rewritten = rewritten
        self.answer = answer
        self.model_name = model_name
        self.prompts = []
        self.synthesis_prompts = []
        self.verifier_prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        if "verifier" in prompt.lower():
            self.verifier_prompts.append(prompt)
            return self.verify_labels.pop(0) if self.verify_labels else "unsupported"
        if prompt.startswith("Rewrite"):
            return self.rewritten
        # Everything else is a synthesis call.
        self.synthesis_prompts.append(prompt)
        return self.answer


def _records(n, doc_id=0):
    return [{"text": f"chunk {i}", "doc_id": doc_id, "chunk_id": i} for i in range(n)]


class TestOriginalQueryAnswering(unittest.TestCase):
    def test_synthesis_and_verification_use_original_query_after_rewrite(self):
        original = "ORIGINAL USER QUESTION"
        # First (original) retrieval is too small -> forces a rewrite; the
        # rewritten query then returns enough evidence to draft & verify.
        retriever = ScriptedRetriever(
            {
                original: _records(1),
                "REWRITTEN QUERY": _records(3, doc_id=1),
            }
        )
        gen = RecordingGenerator(verify_labels=["supported"])
        module = AgenticRAGModule(
            retriever,
            gen,
            config={"min_evidence_count": 2, "use_query_rewrite": True, "max_iterations": 5},
        )

        answer, trace = module.answer(original, return_trace=True)

        self.assertEqual(trace["status"], "answered")
        # The rewrite happened and drove retrieval with the working query...
        self.assertIn("REWRITTEN QUERY", retriever.queries)
        # ...but synthesis addressed the ORIGINAL question, not the rewrite.
        self.assertTrue(gen.synthesis_prompts)
        for prompt in gen.synthesis_prompts:
            self.assertIn(original, prompt)
            self.assertNotIn("REWRITTEN QUERY", prompt)
        # ...and so did verification.
        self.assertTrue(gen.verifier_prompts)
        for prompt in gen.verifier_prompts:
            self.assertIn(original, prompt)
            self.assertNotIn("REWRITTEN QUERY", prompt)


class TestQueryPrefixParity(unittest.TestCase):
    PREFIX = "ANSWER-FORMAT-INSTRUCTION"

    def _module(self, retriever, gen, **extra):
        config = {"min_evidence_count": 2, "use_query_rewrite": False, "query_prefix": self.PREFIX}
        config.update(extra)
        return AgenticRAGModule(retriever, gen, config=config)

    def test_prefix_applied_to_synthesis(self):
        retriever = ScriptedRetriever({"q": _records(3)})
        gen = RecordingGenerator(verify_labels=["supported"])
        module = self._module(retriever, gen)

        module.answer("q")

        self.assertTrue(gen.synthesis_prompts)
        self.assertTrue(all(self.PREFIX in p for p in gen.synthesis_prompts))

    def test_prefix_not_applied_to_verification(self):
        retriever = ScriptedRetriever({"q": _records(3)})
        gen = RecordingGenerator(verify_labels=["supported"])
        module = self._module(retriever, gen)

        module.answer("q")

        self.assertTrue(gen.verifier_prompts)
        self.assertTrue(all(self.PREFIX not in p for p in gen.verifier_prompts))

    def test_no_prefix_when_unset(self):
        retriever = ScriptedRetriever({"q": _records(3)})
        gen = RecordingGenerator(verify_labels=["supported"])
        module = AgenticRAGModule(
            retriever, gen, config={"min_evidence_count": 2, "use_query_rewrite": False}
        )

        module.answer("q")

        self.assertTrue(gen.synthesis_prompts)
        self.assertTrue(all(self.PREFIX not in p for p in gen.synthesis_prompts))


if __name__ == "__main__":
    unittest.main()
