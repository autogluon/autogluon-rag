# Using only the generator without any retrieved content.

If you want to use the generator model directly without retrieving any context from the processed data, you can do so by setting the `top_k` parameter to 0 in the retriever module. 

### Using config file

```python
retriever:
  retriever_top_k: 20
```

### Using code

```python
agrag = AutoGluonRAG(intialization_data)
agrag.initialize_rag_pipeline()
agrag.retriever_module.top_k = 0
agrag.generate_response("What is AutoGluon?")
```
