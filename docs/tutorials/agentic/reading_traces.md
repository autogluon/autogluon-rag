# Inspecting the agent's trace

The agentic path can return a structured trace alongside the answer, describing how the agent arrived at its response: the subqueries it planned, the evidence it retrieved, whether the draft answer was verified, and simple run metrics. This is useful for debugging, evaluation, and understanding why the agent answered the way it did.

## Requesting a trace

Pass `return_trace=True` to `generate_response`. Instead of just the answer string, it returns an `(answer, trace)` tuple:

```python
from agrag.agrag import AutoGluonRAG

agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir="/path/to/docs")
agrag.initialize_rag_pipeline()

answer, trace = agrag.generate_response(
    "What is AutoGluon and how does it compare to AutoKeras?",
    mode="agentic",
    return_trace=True,
)
```

You can also set `return_trace: true` in the `agent` config block to make traces the default for the agentic path.

## What the trace contains

The `trace` is a serializable dictionary. Its main fields are:

- `subqueries`: the subqueries the planner decomposed the question into.
- `evidence`: the retrieved evidence the agent conditioned on, ordered best-first.
- `metrics`: simple run metrics such as `retrieval_calls` and `iterations`.
- `verification`: the verification result (label plus details) for the final draft, or `null` if verification did not run. A top-level `status` field records the overall run outcome.
- `final_answer`: the answer that was returned.

For example, to inspect how much retrieval the agent did:

```python
print(trace["metrics"]["retrieval_calls"])
print(trace["subqueries"])
```

## A worked example

`local_example/benchmark_musique.py` runs the agentic path over a multi-hop QA benchmark with `return_trace=True` and reads fields off the trace (subqueries, evidence, and retrieval metrics) to score and analyze each run. It is a good reference for how to consume the trace programmatically.
