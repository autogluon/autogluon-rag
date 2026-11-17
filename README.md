<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-RAG

## News
- 2026/07: OPERA, our data pruning framework for efficient retrieval model finetuning, is accepted at COLM 2026 and released under [`projects/opera`](projects/opera).

## Overview
AutoGluon-RAG is a framework designed to streamline the development of RAG (Retrieval-Augmented Generation) pipelines. RAG has emerged as a crucial approach for tailoring large language models (LLMs) to address domain-specific queries. However, constructing RAG pipelines traditionally involves navigating through a complex array of modules and functionalities, including retrievers, generators, vector database construction, fast semantic search, and handling long-context inputs, among others.

AutoGluon-RAG allows users to create customized RAG pipelines seamlessly, eliminating the need to delve into any technical complexities. Following the AutoML (Automated Machine Learning) philosophy of simplifying model development with minimal code, as exemplified by AutoGluon; AutoGluon-RAG enables users to create a RAG pipeline with just a few lines of code. The framework provides a user-friendly interface, and abstracts away the underlying modules, allowing users to focus on their domain-specific requirements and leveraging the power of RAG pipelines without the need for extensive technical expertise. 

## Goal
In line with the AutoGluon team's commitment to meeting user requirements and expanding its user base, the team aims to develop a new feature that simplifies the creation and deployment of end-to-end RAG (Retrieval-Augmented Generation) pipelines. Given a set of user-provided data or documents, this feature will enable users to develop and deploy a RAG pipeline with minimal coding effort, following the AutoML (Automated Machine Learning) philosophy of three-line solutions.

## Usage
To use this framework, you must first install AutoGluon RAG:
```python
git clone https://github.com/autogluon/autogluon-rag
cd autogluon-rag

# Create a Virtual Environment (using Python, or conda if you prefer)
python3 -m virtualenv venv
source venv/bin/activate

#Install the package
pip install -e .
```
You can now use the package in two ways. 

### Use AutoGluon-RAG through the command line as `agrag`:

```python
AutoGluon-RAG


usage: agrag [-h] [--config_file] [--preset_quality] [--web_urls [...]]
             [--base_urls [...]] [--parse_urls_recursive] [--data_dir]
             [--mode {standard,agentic}]

AutoGluon-RAG - Retrieval-Augmented Generation Pipeline

options:
  -h, --help            show this help message and exit
  --config_file         Path to the configuration file
  --preset_quality      Preset quality settings for the RAG pipeline
                        (default: medium_quality)
  --web_urls [ ...]     List of URLs to use for RAG
  --base_urls [ ...]    List of base URLs to restrict web URL parsing.
                        Only URLs stemming from a base URL will be processed.
  --parse_urls_recursive
                        Enable recursive parsing of all URLs from the
                        provided web URL list
  --data_dir            Directory containing files to use for RAG.
                        Supports local or S3 paths.
  --mode {standard,agentic}
                        Answering mode: 'standard' (single-pass RAG) or
                        'agentic' (multi-step planning/verification). If
                        omitted, the agent.default_mode / agent.enabled
                        config values decide.
```

### Use AutoGluon-RAG through code:
```python
from agrag.agrag import AutoGluonRAG


def ag_rag():
    agrag = AutoGluonRAG(
        preset_quality="medium_quality", # or path to config file
        web_urls=["https://auto.gluon.ai/stable/index.html"],
        base_urls=["https://auto.gluon.ai/stable/"],
        parse_urls_recursive=True,
        data_dir="s3://autogluon-rag-github-dev/autogluon_docs/"
    )
    agrag.initialize_rag_pipeline()
    agrag.generate_response("What is AutoGluon?")


if __name__ == "__main__":
    ag_rag()
```

### Agentic RAG (optional)

In addition to the standard single-pass pipeline (retrieve → generate), AutoGluon-RAG
provides an optional agentic RAG path. Instead of answering in one shot, the agent
runs a short, bounded reasoning loop that can plan subqueries, retrieve for each of them,
rewrite the query when retrieval is weak, verify that the draft answer is supported by the
retrieved evidence, and abstain when the evidence is insufficient. The agentic path reuses
the same retriever and generator as the standard pipeline; it does not re-ingest or
re-embed your documents.

The standard path remains the default. Enable the agentic path per call:

```python
from agrag.agrag import AutoGluonRAG

agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir="/path/to/docs")
agrag.initialize_rag_pipeline()

# Standard single-pass RAG (default)
agrag.generate_response("What is AutoGluon?")

# Agentic RAG for this query
answer = agrag.generate_response("What is AutoGluon and how does it compare to AutoKeras?", mode="agentic")

# Ask for a structured trace (plan, steps, evidence, verification, metrics)
answer, trace = agrag.generate_response("What is AutoGluon?", mode="agentic", return_trace=True)
```

From the command line, pass `--mode agentic` to route queries through the agentic path:

```bash
agrag --mode agentic
```

You can also make agentic the default by setting the `agent` block in your config file
(`enabled: true` or `default_mode: agentic`). The commonly used parameters are shown below.
The full set of agent parameters (including many advanced multi-hop knobs not shown here)
and their defaults lives in `src/agrag/configs/agent/default.yaml`. Note that the defaults
file uses uppercase, flat keys (e.g. `AGENT_ENABLED`, `AGENT_MAX_ITERATIONS`), while a user
config file uses the lowercase, nested `agent:` block shown here (`enabled`, `max_iterations`, ...):

```yaml
agent:
  enabled: false            # if true, generate_response uses the agentic path by default
  default_mode: standard    # "standard" or "agentic"
  max_iterations: 5         # hard cap on reasoning-loop iterations
  max_subqueries: 4         # max planned subqueries per query
  retrieve_top_k_per_query: 8
  max_context_tokens: 6000  # approximate context budget for synthesis
  use_query_rewrite: true   # allow rewriting the query after weak retrieval
  use_context_compression: false
  use_verification: true    # LLM-judge verification of the draft answer
  min_evidence_count: 2     # minimum evidence required before answering
  return_trace: false       # return (answer, trace) instead of just answer
```

A runnable example over local documents lives in `local_example/`. It uses AWS Bedrock Cohere Embed English v3 embeddings and an AWS Bedrock Claude Sonnet 4.6 generator by default (credentials via the standard AWS chain); a HuggingFace reranker (`BAAI/bge-reranker-base`) is configured but disabled (`use_reranker: false`) in that example. Swap the generator in `local_example/local_config.yaml` for a local HuggingFace model to run without cloud credentials.

For a list of configurable parameters that can be passed into the `AutoGluonRAG` class, refer to the tutorial [here](https://github.com/autogluon/autogluon-rag/tree/main/docs/tutorials/general/code_parameters.md). 

You can also use a configuration file with `AutoGluonRAG`.
The configuration file contains the specific parameters to use for each module in the RAG pipeline. For an example of a config file, please refer to `medium_quality_config.yaml` in `src/agrag/configs/presets/`. For specific details about the parameters in each individual module, refer to the `README` files in each module in `src/agrag/modules/`.

There is also a `shared` section in the config file for parameters that do not refer to a specific module. Currently, the parameters in `shared` are: 
```python
pipeline_batch_size: Optional batch size to use for pre-processing stage (Data Processing, Embedding, Vector DB Module). This represents the number of files in each batch. The default value is 20.
```

## Evaluation
For more information about the evaluation module, refer to the code in `src/agrag/evaluation` and the instructions [here](https://github.com/autogluon/autogluon-rag/tree/main/src/agrag/evaluation/README.md).

## Tutorials
For a list of tutorials on using AutoGluon-RAG in different scenarios, refer to the documentation [here](https://github.com/autogluon/autogluon-rag/tree/main/docs/tutorials)

## Reference

For a list of tutorials on using AutoGluon-RAG in different scenarios, refer to the documentation [here](https://github.com/autogluon/autogluon-rag/tree/main/documentation/tutorial.md)

## Citation
If you use OPERA in your research, please cite:
```bibtex
@article{fang2026opera,
  title={OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation},
  author={Fang, Haoyang and Zhang, Shuai and Ma, Yifei and Wang, Hengyi and Hu, Cuixiong and Kirchhoff, Katrin and Wang, Bernie and Karypis, George},
  journal={arXiv preprint arXiv:2603.17205},
  year={2026}
}
```

## Research code
This repository contains research code intended for academic research and experimentation. It is not production-ready and should be reviewed, tested, and secured before use in production.
