## Embedding Module

This module is responsible for generating vector embeddings from the data provided by the Data Processing Module

Here are the configurable parameters for this module:

```
embedding:
  embedding_model: The name of the Huggingface or Bedrock model to use for generating embeddings (default is "BAAI/bge-large-en" from Huggingface). Currently only Amazon Titan and Cohere Embedding models are supported on Bedrock.
  
  pooling_strategy: The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
  
  normalize_embeddings: Whether to normalize the embeddings.
  
  hf_model_params: Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
  hf_tokenizer_init_params: Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
  
  hf_tokenizer_params: Additional parameters to pass to the `tokenizer` method for the Huggingface model.
  
  hf_forward_params: Additional parameters to pass to the Huggingface model's `forward` method.
  
  normalization_params: Additional parameters to pass to the PyTorch `nn. functional.normalize` method.

  query_instruction_for_retrieval: Instruction for query when using embedding model. 

  use_bedrock: Whether to use the provided model from AWS Bedrock API. https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
  Currently only Cohere (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html) and Amazon Titan (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan.html) embedding models are supported.
  
  bedrock_embedding_params: Additional parameters to pass into the model when generating the embeddings.
    
  bedrock_aws_region: AWS region where the model is hosted on Bedrock.
```