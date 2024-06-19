## Retriever Module

This module is responsible for retrieving and reranking the most similar (top k) vector embeddings compared to the user provided query related to the documents processed.

Here are the configurable parameters for this module:

```
retriever:
  top_k_embeddings: The top-k documents to retrieve (default is 20).

  use_reranker: Whether or not to use a reranker

  reranker_model_name: The name of the Huggingface model to use for reranking the embeddings related to the query.

  reranker_batch_size: Batch size to use when processing the retrieved embeddings
  
  reranker_hf_model_params: Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
  reranker_hf_tokenizer_init_params: Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
  
  reranker_hf_tokenizer_params: Additional parameters to pass to the `tokenizer` method for the Huggingface model.
  
  reranker_hf_forward_params: Additional parameters to pass to the Huggingface model's `forward` method.
  
```