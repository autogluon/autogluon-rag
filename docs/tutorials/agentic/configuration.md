# Configuring Agentic RAG

The agentic path is configured through an `agent` block in your configuration file. The standard path remains the default; the `agent` block only takes effect when the agentic path runs (see [Getting started with Agentic RAG](quickstart.md)).

## The `agent` config block

```yaml
agent:
  enabled: false            # if true, generate_response uses the agentic path by default
  default_mode: standard    # "standard" or "agentic"
  max_iterations: 5         # hard cap on reasoning-loop iterations
  max_subqueries: 4         # max planned subqueries per query
  retrieve_top_k_per_query: 8
  max_context_tokens: 6000  # approximate context budget for synthesis
  use_query_rewrite: true   # allow rewriting the query after weak retrieval
  use_verification: true    # LLM-judge verification of the draft answer
  min_evidence_count: 2     # minimum evidence required before answering
  allow_abstention: false   # see "Never-refuse default" below
  return_trace: false       # return (answer, trace) instead of just the answer
```

Key parameters:

- `enabled` / `default_mode` control whether the agentic path is used when `generate_response` is called without an explicit `mode`.
- `max_iterations` caps how many times the reasoning loop can plan, retrieve, rewrite, and re-draft before it must return an answer.
- `max_subqueries` is the maximum number of subqueries the planner decomposes a question into.
- `retrieve_top_k_per_query` is how many chunks each retrieval call returns.
- `use_query_rewrite` lets the agent rewrite the query and retrieve again when retrieval is weak.
- `use_verification` runs an LLM-judge check that the draft answer is supported by the retrieved evidence.
- `min_evidence_count` is the minimum supporting evidence required before the agent prefers drafting an answer.
- `return_trace` returns `(answer, trace)` instead of just the answer string (see [Inspecting the agent's trace](reading_traces.md)).

## Abstention

By default `allow_abstention: false`, so the agent does not abstain. On weak or failed verification it iterates, rewriting the query and retrieving more, and when the iteration budget is exhausted it returns a best-effort, evidence-grounded answer rather than a canned "I don't have enough evidence" message. Set `allow_abstention: true` to let it abstain when a question appears unanswerable from the indexed documents.

## All agent parameters

The `agent` block above covers the common parameters. The full commented list of every agent parameter and its default lives in `src/agrag/configs/agent/default.yaml`. Note that this defaults file uses uppercase keys (e.g. `AGENT_MAX_ITERATIONS`), while the `agent:` block in a user configuration file uses the short lowercase keys shown above.

That file also documents several advanced, opt-in flags for tuning multi-hop question answering, for example `use_llm_planner` and `use_strands_planner` (LLM/Strands-based query decomposition), `use_iterative_planner` (sequential, dependent hops), and `verifier_model` (an independent, ideally stronger, model for verification). These are all off by default; refer to `src/agrag/configs/agent/default.yaml` for details before enabling them.
