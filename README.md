# AutoGluon-RAG

## Overview
AutoGluon-RAG is a framework designed to streamline the development of RAG (Retrieval-Augmented Generation) pipelines. RAG has emerged as a crucial approach for tailoring large language models (LLMs) to address domain-specific queries. However, constructing RAG pipelines traditionally involves navigating through a complex array of modules and functionalities, including retrievers, generators, vector database construction, fast semantic search, and handling long-context inputs, among others.

AutoGluon-RAG allows users to create customized RAG pipelines seamlessly, eliminating the need to delve into any technical complexities. Following the AutoML (Automated Machine Learning) philosophy of simplifying model development with minimal code, as exemplified by AutoGluon; AutoGluon-RAG enables users to create a RAG pipeline with just a few lines of code. The framework provides a user-friendly interface, and abstracts away the underlying modules, allowing users to focus on their domain-specific requirements and leveraging the power of RAG pipelines without the need for extensive technical expertise. 

## Goal
In line with the AutoGluon team's commitment to meeting user requirements and expanding its user base, the team aims to develop a new feature that simplifies the creation and deployment of end-to-end RAG (Retrieval-Augmented Generation) pipelines. Given a set of user-provided data or documents, this feature will enable users to develop and deploy a RAG pipeline with minimal coding effort, following the AutoML (Automated Machine Learning) philosophy of three-line solutions.

## Usage
```
AutoGluon-RAG


usage: agrag [-h] --data_dir  [--chunk_size] [--chunk_overlap]

AutoGluon-RAG - Retrieval-Augmented Generation Pipeline

options:
  -h, --help        show this help message and exit
  --data_dir        Path to the directory containing the documents to be ingested
                    into the RAG pipeline
  --chunk_size      Maximum chunk length to split the documents into
  --chunk_overlap   Amount of overlap between consecutive chunks. This is the
                    number of characters that will be shared between adjacent
                    chunks
```
