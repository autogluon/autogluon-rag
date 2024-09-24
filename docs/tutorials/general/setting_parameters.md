# Tutorial on setting module parameters in `AutoGluonRAG`. 

```python
agrag = AutoGluon()
agrag.initialize_rag_pipeline()
```

You can access all the module-level parameters using `agrag.args`. 

For example:
```python
agrag.args.embedding_model = "new_model_name"
```

For specific details about the parameters in each individual module, refer to the `README` files in each module in `src/agrag/modules/`.
