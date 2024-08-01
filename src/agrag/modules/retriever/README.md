## Retriever Module

This module is responsible for retrieving and reranking the most similar (top k) vector embeddings compared to the user provided query related to the documents processed.

Here are the configurable parameters for this module:

```
retriever:
  retriever_top_k: The top-k documents to retrieve (default is 50).
  
  reranker_top_k: The top-k documents to use as context for generation (default is 10).
  
  use_reranker: Whether or not to use a reranker

  reranker_model_name: The name of the Huggingface model to use for reranking the embeddings related to the query.

  reranker_model_platform: The name of the platform where the model is hosted. Currently only Huggingface ("huggingface") models are supported.
  
  reranker_model_platform_args: Additional platform-specific parameters to use when initializing the model, reranking, etc.

  reranker_batch_size: Batch size to use when processing the retrieved embeddings
  
```

#### `reranker_model_platform_args` structure
If you are using `huggingface` platform, the arguments must be structured as:
  ```python
  reranker_model_platform_args = {
      "hf_model_params": {} # Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
      "hf_tokenizer_init_params": {} # Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
      
      "hf_tokenizer_params": {} # Additional parameters to pass to the `tokenizer` method for the Huggingface model.
      
      "hf_forward_params": {} # Additional parameters to pass to the Huggingface model's `forward` method.
  }
  ```
