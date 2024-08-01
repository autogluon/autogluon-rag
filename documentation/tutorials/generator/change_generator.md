This is a tutorial on changing generator after initializing RAG pipeline:

```python
agrag = AutoGluon()
agrag.initialize_rag_pipeline()

# Change Generator Model Configuration
agrag.args.generator_model_name = "new_model"
agrag.args.generator_query_prefix = "new query prefix"

# Reinitialize Module
agrag.initialize_generator_module()
response = agrag.generate_response(query_text) 
```
**Note**: You <b>must</b> reinitialize the appropriate module after updating its arguments to propagate the changes into the module.
