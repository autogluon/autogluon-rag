# Disabling the reranker and only use embedding model for retrieval.

The reranker is an optional sub-module that can be used within the retriever module to rerank the retrieved text from the Vector DB.
There are two ways to configure the reranker.

One way is through the config file:
```python
retriever:
    use_reranker: true
    reranker_model_name: BAAI/bge-large-en
    reranker_model_platform: huggingface
    reranker_model_platform_args: null
```

The alternate way is through code:
Refer to [this](https://github.com/autogluon/autogluon-rag/tree/main/documentation/tutorials/general/setting_parameters.md) tutorial on how to modify arguments through code after instantiating an  `AutoGluonRAG` object.
```python
agrag.use_reranker = False
agrag.initialize_retriever_module()
```
