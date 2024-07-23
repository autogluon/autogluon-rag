<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-RAG

## Overview
AutoGluon-RAG is a framework designed to streamline the development of RAG (Retrieval-Augmented Generation) pipelines. RAG has emerged as a crucial approach for tailoring large language models (LLMs) to address domain-specific queries. However, constructing RAG pipelines traditionally involves navigating through a complex array of modules and functionalities, including retrievers, generators, vector database construction, fast semantic search, and handling long-context inputs, among others.

AutoGluon-RAG allows users to create customized RAG pipelines seamlessly, eliminating the need to delve into any technical complexities. Following the AutoML (Automated Machine Learning) philosophy of simplifying model development with minimal code, as exemplified by AutoGluon; AutoGluon-RAG enables users to create a RAG pipeline with just a few lines of code. The framework provides a user-friendly interface, and abstracts away the underlying modules, allowing users to focus on their domain-specific requirements and leveraging the power of RAG pipelines without the need for extensive technical expertise. 

## Goal
In line with the AutoGluon team's commitment to meeting user requirements and expanding its user base, the team aims to develop a new feature that simplifies the creation and deployment of end-to-end RAG (Retrieval-Augmented Generation) pipelines. Given a set of user-provided data or documents, this feature will enable users to develop and deploy a RAG pipeline with minimal coding effort, following the AutoML (Automated Machine Learning) philosophy of three-line solutions.

## Usage
To use this framework, you must first install AutoGluon RAG:
```
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

```
AutoGluon-RAG


usage: agrag [-h] --config_file

AutoGluon-RAG - Retrieval-Augmented Generation Pipeline

options:
  -h, --help        show this help message and exit
  --config_file        Path to the configuration file 
```

### Use AutoGluon-RAG through code:
```
from agrag.agrag import AutoGluonRAG


def ag_rag():
    agrag = AutoGluonRAG(config_file="path/to/config")
    agrag.initialize_rag_pipeline()
    agrag.generate_response("What is AutoGluon?")


if __name__ == "__main__":
    ag_rag()
```

These are the <b>optional</b> parameters that can be passed into the `AutoGluonRAG` class:
```
config_file : str
    Path to a configuration file that will be used to set specific parameters in the RAG pipeline.
preset_quality : str
    If you do not wish to use your own configuration file, you can use a preset configuration file which contains pre-defined arguments.
    You must provide the preset quality setting ("good_quality", "medium_quality", or, "best_quality"). Note that if both config_file and preset_quality are provided, config_file will be prioritized.  
model_ids : dict
    Dictionary of model IDs to use for specific modules.
    Example: {"generator_model_id": "mistral.mistral-7b-instruct-v0:2", "retriever_model_id": "BAAI/bge-large-en", "reranker_model_id": "nv_embed"}
data_dir : str
    The directory containing the data files that will be used for the RAG pipeline. If this value is not provided when initializing the object, it must be provided in the config file. If both are provided, the value in the class instantiation will be prioritized. 
pipeline_batch_size: int
    Batch size to use for pre-processing stage (Data Processing, Embedding, Vector DB Module). This represents the number of files in each batch.
    The default value is 20.
```

The configuration file contains the specific parameters to use for each module in the RAG pipeline. For an example of a config file, please refer to `example_config.yaml` in `src/agrag/configs/`. For specific details about the parameters in each individual module, refer to the `README` files in each module in `src/agrag/modules/`.

There is also a `shared` section in the config file for parameters that do not refer to a specific module. Currently, the parameters in `shared` are: 
```
pipeline_batch_size: Optional batch size to use for pre-processing stage (Data Processing, Embedding, Vector DB Module). This represents the number of files in each batch. The default value is 20.
```

## Evaluation
For more information about the evaluation module, refer to the code in `src/agrag/evaluation` and the instructions [here](https://github.com/autogluon/autogluon-rag/tree/main/src/agrag/evaluation/README.md).
