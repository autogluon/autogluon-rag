---
sd_hide_title: true
hide-toc: true
---

# AutoGluon-RAG

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autogluon-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

AutoGluon-RAG: Retrieval-Augmented Generation in 3 Lines of Code!

:::
::::

::::::

```{figure} ./_static/github.png
:width: 60px
:class: sd-m-auto sd-animate-grow50-rot20
:target: https://github.com/autogluon/autogluon-rag
:alt: GitHub Repo
```


AutoGluon-RAG is a framework designed to streamline the development of RAG (Retrieval-Augmented Generation) pipelines. RAG has emerged as a crucial approach for tailoring large language models (LLMs) to address domain-specific queries. However, constructing RAG pipelines traditionally involves navigating through a complex array of modules and functionalities, including retrievers, generators, vector database construction, fast semantic search, and handling long-context inputs, among others.

AutoGluon-RAG allows users to create customized RAG pipelines seamlessly, eliminating the need to delve into any technical complexities. Following the AutoML (Automated Machine Learning) philosophy of simplifying model development with minimal code, as exemplified by AutoGluon; AutoGluon-RAG enables users to create a RAG pipeline with just a few lines of code. The framework provides a user-friendly interface, and abstracts away the underlying modules, allowing users to focus on their domain-specific requirements and leveraging the power of RAG pipelines without the need for extensive technical expertise. 

## Goal
In line with the AutoGluon team's commitment to meeting user requirements and expanding its user base, the team aims to develop a new feature that simplifies the creation and deployment of end-to-end RAG (Retrieval-Augmented Generation) pipelines. Given a set of user-provided data or documents, this feature will enable users to develop and deploy a RAG pipeline with minimal coding effort, following the AutoML (Automated Machine Learning) philosophy of three-line solutions.

## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon-rag)
![](https://img.shields.io/pypi/v/autogluon-rag.svg)
![](https://img.shields.io/pypi/dm/autogluon-rag)

```bash
pip install -U pip
pip install autogluon-rag
```

```{toctree}
---
caption: Tutorials
maxdepth: 1
hidden:
---

RAG <tutorials/index>
```

```{toctree}
---
caption: Common Errors
maxdepth: 1
hidden:
---

---
List of common errors <common_errors/common_errors>
```

```{toctree}
---
caption: Evaluation
maxdepth: 1
hidden:
---

Evaluation Module <evaluation/evaluation>
```

```{toctree}
---
caption: API
maxdepth: 1
hidden:
---

DataProcessingModule <api/agrag.modules.DataProcessingModule>
EmbeddingModule <api/agrag.modules.EmbeddingModule>
VectorDatabaseModule <api/agrag.modules.VectorDatabaseModule>
RetrieverModule <api/agrag.modules.RetrieverModule>
Reranker <api/agrag.modules.Reranker>
GeneratorModule <api/agrag.modules.GeneratorModule>
```

```{toctree}
---
caption: Resources
maxdepth: 2
hidden:
---
GitHub <https://github.com/autogluon/autogluon-rag>
```
