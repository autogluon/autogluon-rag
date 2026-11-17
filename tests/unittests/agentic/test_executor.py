import unittest

from agrag.modules.agentic.agentic_module import DEFAULT_ABSTENTION, AgenticRAGModule


class FakeRetriever:
    """Returns `per_query` records for any query; counts retrieve calls."""

    def __init__(self, records_per_query):
        self.records_per_query = records_per_query
        self.calls = 0
        self.top_ks = []

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.calls += 1
        self.top_ks.append(top_k)
        return list(self.records_per_query)


class FakeGenerator:
    """Scripted responses: verification label controlled per test."""

    def __init__(self, answer="the answer", verify_label="supported", model_name="mistral-7b"):
        self.answer = answer
        self.verify_label = verify_label
        self.model_name = model_name
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        # Heuristic: the verifier prompt contains the word "verifier".
        if "verifier" in prompt.lower():
            return self.verify_label
        # Query rewrite prompt asks to "Rewrite".
        if prompt.startswith("Rewrite"):
            return "rewritten query"
        return self.answer


def _records(n):
    return [{"text": f"chunk {i}", "doc_id": 0, "chunk_id": i, "rank": i} for i in range(n)]


class TestAgenticRAGModule(unittest.TestCase):
    def test_answered_run_returns_answer(self):
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(answer="AutoGluon is an AutoML library.", verify_label="supported")
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2})
        answer = module.answer("what is autogluon")
        self.assertEqual(answer, "AutoGluon is an AutoML library.")

    def test_trace_returned_when_requested(self):
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(verify_label="supported")
        module = AgenticRAGModule(retriever, gen)
        answer, trace = module.answer("q", return_trace=True)
        self.assertIsInstance(trace, dict)
        self.assertEqual(trace["status"], "answered")
        self.assertGreaterEqual(trace["metrics"]["evidence_count"], 2)
        self.assertGreaterEqual(trace["metrics"]["retrieval_calls"], 1)

    def test_abstains_when_no_evidence_and_abstention_allowed(self):
        # Refuse path (opt-in): with allow_abstention=True and no evidence, the
        # agent returns the canned abstention.
        retriever = FakeRetriever([])  # never returns evidence
        gen = FakeGenerator()
        module = AgenticRAGModule(
            retriever,
            gen,
            config={"min_evidence_count": 2, "use_query_rewrite": False, "allow_abstention": True},
        )
        answer = module.answer("q")
        self.assertEqual(answer, DEFAULT_ABSTENTION)

    def test_never_refuses_when_no_evidence_by_default(self):
        # Default (never-refuse): even with no evidence retrieved, the agent
        # synthesizes a best-effort answer rather than the canned abstention.
        retriever = FakeRetriever([])  # never returns evidence
        gen = FakeGenerator(answer="best effort")
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2, "use_query_rewrite": False})
        answer = module.answer("q")
        self.assertEqual(answer, "best effort")
        self.assertNotEqual(answer, DEFAULT_ABSTENTION)

    def test_loop_respects_max_iterations(self):
        # Evidence always insufficient, rewrite enabled: must still terminate.
        retriever = FakeRetriever(_records(1))
        gen = FakeGenerator()
        module = AgenticRAGModule(
            retriever,
            gen,
            config={"min_evidence_count": 5, "max_iterations": 3, "use_query_rewrite": True},
        )
        answer, trace = module.answer("q", return_trace=True)
        self.assertLessEqual(trace["metrics"]["iterations"], 3)
        # The loop must terminate within budget. Under the never-refuse default a
        # forced-abstain terminal returns a best-effort answer; when that draft was
        # not verifier-accepted the status is "answered_unverified" (or "answered" if
        # it was). "max_iterations" is also acceptable if the cap is hit first.
        self.assertIn(
            trace["status"], ("answered", "answered_unverified", "abstained", "max_iterations")
        )

    def test_verification_disabled_accepts_draft(self):
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(answer="unverified answer", verify_label="unsupported")
        module = AgenticRAGModule(retriever, gen, config={"use_verification": False, "min_evidence_count": 2})
        answer, trace = module.answer("q", return_trace=True)
        self.assertEqual(answer, "unverified answer")
        self.assertEqual(trace["verification"]["label"], "unverified")

    def test_unsupported_then_exhausts_and_abstains_when_allowed(self):
        # Refuse path (opt-in): enough evidence to draft, but the verifier always
        # says unsupported. With allow_abstention=True an unaccepted draft is not
        # returned: the agent abstains via max iterations.
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(answer="the answer", verify_label="unsupported")
        module = AgenticRAGModule(
            retriever,
            gen,
            config={
                "min_evidence_count": 2,
                "max_iterations": 3,
                "use_query_rewrite": False,
                "allow_abstention": True,
            },
        )
        answer = module.answer("q")
        # Never accepted -> abstains via max iterations.
        self.assertEqual(answer, DEFAULT_ABSTENTION)

    def test_returns_best_draft_instead_of_abstaining_by_default(self):
        # Default (never-refuse): same setup, but the agent must never emit the
        # canned abstention. It returns its best (verifier-rejected) draft.
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(answer="the answer", verify_label="unsupported")
        module = AgenticRAGModule(
            retriever,
            gen,
            config={"min_evidence_count": 2, "max_iterations": 3, "use_query_rewrite": False},
        )
        answer, trace = module.answer("q", return_trace=True)
        self.assertEqual(answer, "the answer")
        self.assertNotEqual(answer, DEFAULT_ABSTENTION)
        # The draft was returned but never verifier-accepted, so the run is marked
        # ANSWERED_UNVERIFIED (not ANSWERED) to keep the true accept rate observable.
        self.assertEqual(trace["status"], "answered_unverified")

    def test_retrieve_top_k_per_query_reaches_retriever(self):
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(verify_label="supported")
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2, "retrieve_top_k_per_query": 8})
        module.answer("q")
        # Every retrieval call used the configured per-query top_k.
        self.assertTrue(retriever.top_ks)
        self.assertTrue(all(k == 8 for k in retriever.top_ks))

    def test_retrieve_top_k_defaults_to_none_when_unset(self):
        retriever = FakeRetriever(_records(3))
        gen = FakeGenerator(verify_label="supported")
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2})
        module.answer("q")
        self.assertTrue(all(k is None for k in retriever.top_ks))

    def test_multi_retrieve_used_for_multipart_query(self):
        retriever = FakeRetriever(_records(2))
        gen = FakeGenerator(verify_label="supported")
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2})
        _, trace = module.answer("what is autogluon and how does RAG work", return_trace=True)
        # Planner produced subqueries -> multi-retrieve on the first step.
        self.assertTrue(any(s["tool_name"] == "MultiQueryRetrieveTool" for s in trace["steps"]))


class ScriptedRetriever:
    """Returns different records depending on the query it is asked."""

    def __init__(self, by_query, default=None):
        self.by_query = by_query
        self.default = default or []
        self.queries = []

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.queries.append(query)
        return list(self.by_query.get(query, self.default))


class ScriptedGenerator:
    """Returns a scripted sequence of verification labels; fixed rewrite output."""

    def __init__(self, verify_labels, rewritten="REWRITTEN QUERY", answer="draft answer", model_name="mistral-7b"):
        self.verify_labels = list(verify_labels)
        self.rewritten = rewritten
        self.answer = answer
        self.model_name = model_name

    def generate_response(self, prompt):
        if "verifier" in prompt.lower():
            return self.verify_labels.pop(0) if self.verify_labels else "unsupported"
        if prompt.startswith("Rewrite"):
            return self.rewritten
        return self.answer


class TestAgentAdaptation(unittest.TestCase):
    """Regression tests for the reactive loop (rewrite re-retrieves, verification
    changes the next action, no identical re-drafts)."""

    def test_rewrite_triggers_fresh_retrieval(self):
        # First query returns too little evidence; the rewritten query returns
        # enough. The rewritten query MUST reach the retriever.
        retriever = ScriptedRetriever(
            {
                "q": [{"text": "c0", "doc_id": 0, "chunk_id": 0}],
                "REWRITTEN QUERY": [{"text": f"n{i}", "doc_id": 1, "chunk_id": i} for i in range(3)],
            }
        )
        gen = ScriptedGenerator(verify_labels=["supported"])
        module = AgenticRAGModule(
            retriever, gen, config={"min_evidence_count": 2, "use_query_rewrite": True, "max_iterations": 4}
        )
        answer, trace = module.answer("q", return_trace=True)
        self.assertIn("REWRITTEN QUERY", retriever.queries)
        self.assertEqual(trace["status"], "answered")
        # Two retrievals: original + rewritten.
        self.assertGreaterEqual(trace["metrics"]["retrieval_calls"], 2)

    def test_failed_verification_recovers_via_rewrite(self):
        # Enough evidence to draft; first verification fails, second (after a
        # rewrite + re-retrieval) succeeds.
        retriever = ScriptedRetriever(
            {
                "q": [{"text": f"c{i}", "doc_id": 0, "chunk_id": i} for i in range(3)],
                "REWRITTEN QUERY": [{"text": f"n{i}", "doc_id": 9, "chunk_id": i} for i in range(3)],
            }
        )
        gen = ScriptedGenerator(verify_labels=["unsupported", "supported"])
        module = AgenticRAGModule(
            retriever, gen, config={"min_evidence_count": 2, "use_query_rewrite": True, "max_iterations": 5}
        )
        answer, trace = module.answer("q", return_trace=True)
        actions = [s["action_type"] for s in trace["steps"]]
        # Recovery path: draft -> rewrite -> retrieve -> draft, ending answered.
        self.assertEqual(trace["status"], "answered")
        self.assertIn("rewrite_query", actions)
        self.assertEqual(actions.count("draft_answer"), 2)

    def test_failed_verification_recovers_on_default_config(self):
        # Regression: the rewrite-recovery path must complete under the DEFAULT
        # config (no max_iterations override). A too-small default budget would
        # abstain before the rewritten query could be re-drafted.
        retriever = ScriptedRetriever(
            {
                "q": [{"text": f"c{i}", "doc_id": 0, "chunk_id": i} for i in range(3)],
                "REWRITTEN QUERY": [{"text": f"n{i}", "doc_id": 9, "chunk_id": i} for i in range(3)],
            }
        )
        gen = ScriptedGenerator(verify_labels=["unsupported", "supported"])
        # Only the behavior under test is configured; max_iterations is default.
        module = AgenticRAGModule(retriever, gen, config={"min_evidence_count": 2})
        answer, trace = module.answer("q", return_trace=True)
        actions = [s["action_type"] for s in trace["steps"]]
        self.assertEqual(trace["status"], "answered")
        self.assertIn("rewrite_query", actions)
        self.assertEqual(actions.count("draft_answer"), 2)

    def test_no_wasted_rewrite_on_final_iteration(self):
        # With a budget too small to act on a rewrite, the agent must not spend its
        # last iteration on a rewrite whose re-retrieval/re-draft can never run.
        # max_iterations=2 leaves no room: iteration 0 drafts, iteration 1 has
        # nothing to gain from rewriting. Under the never-refuse default the best
        # (verifier-rejected) draft is returned rather than the canned abstention.
        retriever = ScriptedRetriever(
            {}, default=[{"text": f"c{i}", "doc_id": 0, "chunk_id": i} for i in range(3)]
        )
        gen = ScriptedGenerator(verify_labels=["unsupported", "unsupported"])
        module = AgenticRAGModule(
            retriever, gen, config={"min_evidence_count": 2, "use_query_rewrite": True, "max_iterations": 2}
        )
        answer, trace = module.answer("q", return_trace=True)
        actions = [s["action_type"] for s in trace["steps"]]
        self.assertEqual(answer, gen.answer)
        self.assertNotEqual(answer, DEFAULT_ABSTENTION)
        # No rewrite was issued because it could never be acted upon.
        self.assertNotIn("rewrite_query", actions)

    def test_no_infinite_redraft_when_verification_always_fails(self):
        # Verifier never accepts and the rewritten query returns only duplicate
        # (already-seen) evidence, so no NEW evidence ever arrives after a draft.
        # The re-draft path is gated on new evidence, so the agent must not
        # re-draft the identical answer every iteration: at most one draft per
        # distinct query (original + one rewrite, with max_rewrites=1). Under the
        # never-refuse default it then returns the best draft rather than abstaining.
        retriever = ScriptedRetriever(
            {}, default=[{"text": f"c{i}", "doc_id": 0, "chunk_id": i} for i in range(3)]
        )
        gen = ScriptedGenerator(verify_labels=["unsupported", "unsupported", "unsupported"])
        module = AgenticRAGModule(
            retriever,
            gen,
            config={
                "min_evidence_count": 2,
                "use_query_rewrite": True,
                "max_iterations": 6,
                "max_rewrites": 1,
            },
        )
        answer, trace = module.answer("q", return_trace=True)
        actions = [s["action_type"] for s in trace["steps"]]
        self.assertEqual(answer, gen.answer)
        self.assertNotEqual(answer, DEFAULT_ABSTENTION)
        # At most one draft per distinct query (original + one rewrite).
        self.assertLessEqual(actions.count("draft_answer"), 2)

    def test_context_compression_invoked_when_over_budget(self):
        # Large chunks exceed the token budget; compression is enabled and must
        # run before drafting, and its output must feed the synthesizer.
        big = " ".join(["word"] * 50)
        retriever = ScriptedRetriever(
            {}, default=[{"text": big, "doc_id": 0, "chunk_id": i} for i in range(3)]
        )
        gen = ScriptedGenerator(verify_labels=["supported"])
        module = AgenticRAGModule(
            retriever,
            gen,
            config={
                "min_evidence_count": 2,
                "use_context_compression": True,
                "max_context_tokens": 10,
                "max_iterations": 5,
            },
        )
        answer, trace = module.answer("q", return_trace=True)
        actions = [s["action_type"] for s in trace["steps"]]
        self.assertIn("compress_context", actions)
        # Compression runs before the draft.
        self.assertLess(actions.index("compress_context"), actions.index("draft_answer"))
        self.assertEqual(trace["status"], "answered")


class SequentialGenerator:
    """Generator fake for the sequential-hop path.

    Distinguishes the four prompt kinds the sequential executor produces:
    * LLM planner  -> returns a JSON decomposition ("subqueries" list).
    * Hop fill-in  -> returns a scripted self-contained rewrite.
    * Verifier     -> returns "supported".
    * Answer synth -> returns a per-hop / final answer keyed off the question text.
    """

    def __init__(self, plan_subqueries, hop_fill="Which county is Springfield in?", answers=None):
        self.plan_subqueries = plan_subqueries
        self.hop_fill = hop_fill
        self.answers = answers or {}
        self.model_name = "mistral-7b"
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        low = prompt.lower()
        if "decompose" in low and "subqueries" in low:
            import json

            return json.dumps({"subqueries": self.plan_subqueries})
        if "self-contained" in low or "rewritten self-contained query" in low:
            return self.hop_fill
        if "verifier" in low:
            return "supported"
        # Answer synthesis: return a scripted answer if the question text matches a
        # key, else a generic answer.
        for key, val in self.answers.items():
            if key.lower() in low:
                return val
        return "final answer"


class TestSequentialHopExecutor(unittest.TestCase):
    """The opt-in sequential-hop executor threads each hop's resolved answer into
    the next hop's retrieval query."""

    def _module(self, retriever, gen, **overrides):
        config = {
            "min_evidence_count": 1,
            "use_llm_planner": True,
            "use_iterative_planner": True,
            "max_iterations": 8,
        }
        config.update(overrides)
        return AgenticRAGModule(retriever, gen, config=config)

    def test_sequential_path_records_hop_answers(self):
        retriever = ScriptedRetriever({}, default=_records(2))
        gen = SequentialGenerator(
            plan_subqueries=["Where is Springfield?", "Which county is it in?"],
            answers={"where is springfield": "Illinois"},
        )
        _, trace = self._module(retriever, gen).answer(
            "What county is Springfield in?", return_trace=True
        )
        # Two hops resolved and recorded in the chain.
        self.assertEqual(len(trace["hop_answers"]), 2)
        self.assertEqual(trace["hop_answers"][0]["subquery"], "Where is Springfield?")
        self.assertEqual(trace["status"], "answered")

    def test_unresolved_reference_is_filled_before_retrieval(self):
        # The second subquery contains "it" (an unresolved reference), so it must be
        # rewritten via the fill-in prompt before it reaches the retriever.
        retriever = ScriptedRetriever({}, default=_records(2))
        gen = SequentialGenerator(
            plan_subqueries=["Where is Springfield?", "Which county is it in?"],
            hop_fill="Which county is Springfield in?",
        )
        self._module(retriever, gen).answer("What county is Springfield in?")
        # The resolved (filled-in) query reached the retriever, not the raw "it" one.
        self.assertIn("Which county is Springfield in?", retriever.queries)
        self.assertNotIn("Which county is it in?", retriever.queries)

    def test_uses_retrieve_tool_per_hop_not_multi_retrieve(self):
        retriever = ScriptedRetriever({}, default=_records(2))
        gen = SequentialGenerator(
            plan_subqueries=["Where is Springfield?", "Which county is it in?"],
        )
        _, trace = self._module(retriever, gen).answer("q", return_trace=True)
        tool_names = [s["tool_name"] for s in trace["steps"] if s["tool_name"]]
        self.assertIn("RetrieveTool", tool_names)
        self.assertNotIn("MultiQueryRetrieveTool", tool_names)

    def test_single_subquery_plan_falls_through_to_parallel(self):
        # With only one subquery there is no chain to thread, so the sequential path
        # is skipped and the standard loop runs (no hop_answers recorded).
        retriever = ScriptedRetriever({}, default=_records(2))
        gen = SequentialGenerator(plan_subqueries=[])  # planner adds only the original
        _, trace = self._module(retriever, gen).answer("simple question", return_trace=True)
        self.assertEqual(trace["hop_answers"], [])

    def test_disabled_by_default_no_hop_answers(self):
        # Without the flag, even a multi-hop plan uses the parallel path.
        retriever = ScriptedRetriever({}, default=_records(2))
        gen = SequentialGenerator(
            plan_subqueries=["Where is Springfield?", "Which county is it in?"],
        )
        module = AgenticRAGModule(
            retriever,
            gen,
            config={"min_evidence_count": 1, "use_llm_planner": True},  # iterative off
        )
        _, trace = module.answer("q", return_trace=True)
        self.assertEqual(trace["hop_answers"], [])


if __name__ == "__main__":
    unittest.main()
