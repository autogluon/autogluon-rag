## Embedding Module

This module is responsible for generating vector embeddings from the data provided by the Data Processing Module

Here are the configurable parameters for this module:

```
embedding:
  embedding_model: The name of the Huggingface model to use for generating embeddings (default is "BAAI/bge-large-en").
  
  pooling_strategy: The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
  
  normalize_embeddings: Whether to normalize the embeddings.
  
  hf_model_params: Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
  hf_tokenizer_init_params: Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
  
  hf_tokenizer_params: Additional parameters to pass to the `tokenizer` method for the Huggingface model.
  
  hf_forward_params: Additional parameters to pass to the Huggingface model's `forward` method.
  
  normalization_params: Additional parameters to pass to the PyTorch `nn. functional.normalize` method.

  query_instruction_for_retrieval: Instruction for query when using embedding model. 
```