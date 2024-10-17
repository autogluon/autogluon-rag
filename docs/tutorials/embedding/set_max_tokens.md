# Set max number of tokens per embedding when using AWS Bedrock Embedding Models.

Currently, through AWS Bedrock, you can use the Amazon Titan Text and Cohere Embedding Models. Each of these models has a set number of max tokens. Each text musrt be truncated to the specified number of tokens before creating embeddings.

To keep up with changes in the max tokens per models, as well as newly added models, you can specify the max number of tokens in the configuration of AutoGluon-RAG.


### Using config file

```python
embedding:
  embedding_model: cohere.embed-english-v3
  embedding_model_platform: bedrock
  embedding_model_platform_args:
    bedrock_aws_region: us-west-2
    max_tokens: 8192 # max tokens for Amazon Titan Text Embedding Model
```

### Using code

```python
agrag = AutoGluonRAG(intialization_data)
agrag.initialize_rag_pipeline()
agrag.embedding_module.platform_args["max_tokens"] = 8192 # max tokens for Amazon Titan Text Embedding Model
```
