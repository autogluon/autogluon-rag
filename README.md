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
You can now use `agrag` from the command line:

```
AutoGluon-RAG


usage: agrag [-h] --config_file

AutoGluon-RAG - Retrieval-Augmented Generation Pipeline

options:
  -h, --help        show this help message and exit
  --config_file        Path to the configuration file 
```

The configuration file contains the specific parameters to use for each module in the RAG pipeline. For an example of a config file, please refer to `example_config.yaml` in `src/agrag/configs/`. For specific details about the parameters in each individual module, refer to the `README` files in each module in `src/agrag/modules/`.
The configuration file also contains a section called 'misc', which refers to parameters that will be shared across modules throughout the pipeline. For example, `num_gpus`, which refers to the number of GPUs to use in a given module is one such parameter.
