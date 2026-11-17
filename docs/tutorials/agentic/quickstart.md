# Getting started with Agentic RAG

In addition to the standard single-pass pipeline (retrieve → generate), AutoGluon-RAG provides an optional **agentic RAG** path. Instead of answering in one shot, the agent runs a short, bounded reasoning loop that can plan subqueries, retrieve for each of them, rewrite the query when retrieval is weak, and verify that the draft answer is supported by the retrieved evidence.

The agentic path reuses the same retriever and generator as the standard pipeline; it does not re-ingest or re-embed your documents. The standard path remains the default; you opt into the agentic path per call, from the command line, or by default via your config file.

## Using code

Enable the agentic path for a single query by passing `mode="agentic"` to `generate_response`:

```python
from agrag.agrag import AutoGluonRAG

agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir="/path/to/docs")
agrag.initialize_rag_pipeline()

# Standard single-pass RAG (default)
agrag.generate_response("What is AutoGluon?")

# Agentic RAG for this query
answer = agrag.generate_response(
    "What is AutoGluon and how does it compare to AutoKeras?",
    mode="agentic",
)
```

The first agentic query lazily builds the agentic module on top of the already-initialized retriever and generator, so no extra setup is required beyond `initialize_rag_pipeline()`.

## Using the command line

Pass `--mode agentic` to route queries through the agentic path:

```bash
agrag --mode agentic
```

## Making agentic the default

You can make the agentic path the default for every query by setting the `agent` block in your config file — either `enabled: true` or `default_mode: agentic`:

```yaml
agent:
  enabled: true          # or set default_mode: agentic
```

When `mode` is not passed to `generate_response`, the mode is resolved with the following precedence: an explicit `mode` argument, then `agent.default_mode` (when it is set to something other than `"standard"`), then `"agentic"` if `agent.enabled` is set, otherwise `"standard"`. See the [configuration tutorial](configuration.md) for the full set of agent parameters.

## Runnable example

A runnable example over local documents lives in `local_example/`. It uses AWS Bedrock Cohere Embed English v3 embeddings and an AWS Bedrock Claude Sonnet 4.6 generator by default (credentials via the standard AWS chain); a HuggingFace reranker (`BAAI/bge-reranker-base`) is configured but disabled (`use_reranker: false`) in that example. Swap the generator in `local_example/local_config.yaml` for a local HuggingFace model to run without cloud credentials. `local_example/benchmark_musique.py` shows the agentic path running end-to-end on a multi-hop QA benchmark.

## Next steps

- [Configuring Agentic RAG](configuration.md) — the `agent` config knobs and what they do.
- [Inspecting the agent's trace](reading_traces.md) — see the plan, evidence, and verification behind an answer.
- [Configurable parameters for the AutoGluonRAG class](../general/code_parameters.md) — the general constructor parameters.
