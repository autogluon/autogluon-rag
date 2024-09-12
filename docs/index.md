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


AutoGluon-RAG is a framework designed to simplify the creation of Retrieval-Augmented Generation (RAG) pipelines, making it easier to adapt large language models (LLMs) for domain-specific tasks. Traditionally, building RAG pipelines involves dealing with multiple complex components like retrievers, generators, and vector databases.

AutoGluon-RAG streamlines this process, allowing users to build and deploy customized RAG pipelines with just a few lines of code. By abstracting the technical complexities, it enables users to focus on their specific needs, making it accessible to both beginners and experienced practitioners alike.

## Goal
In alignment with the AutoGluon team's mission to meet evolving user needs and broaden its user base, the team aims to introduce a new feature that simplifies the creation and deployment of end-to-end Retrieval-Augmented Generation (RAG) pipelines. This feature will allow users to build and deploy RAG pipelines with minimal coding effort, leveraging user-provided data or documents. Following the AutoML (Automated Machine Learning) philosophy of delivering solutions with minimal complexity, the goal is to enable users to achieve robust RAG pipelines in just three lines of code.

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
