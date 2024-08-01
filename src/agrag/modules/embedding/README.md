## Embedding Module

This module is responsible for generating vector embeddings from the data provided by the Data Processing Module

Here are the configurable parameters for this module:

```
embedding:
  model_name: The name of theEmbedding model to use for generating embeddings (default is "BAAI/bge-large-en" from Huggingface). 

  model_platform: The name of the platform where the model is hosted. Currently only Huggingface ("huggingface") and Bedrock ("bedrock") models are supported. 
  Currently only Cohere (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html) and Amazon Titan (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan.html) embedding models are supported on Bedrock.
  
  platform_args: Additional platform-specific parameters to use when initializing the model, generating embeddings, etc.
  
  pooling_strategy: The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
  
  normalize_embeddings: Whether to normalize the embeddings.
  
  normalization_params: Additional parameters to pass to the PyTorch `nn. functional.normalize` method.

  query_instruction_for_retrieval: Instruction for query when using embedding model. 
    
  bedrock_aws_region: AWS region where the model is hosted on Bedrock.
```

#### `platform_args` structure
If you are using `huggingface` platform, the arguments must be structured as:
  ```python
  platform_args = {
      "hf_model_params": {} # Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
      "hf_tokenizer_init_params": {} # Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
      
      "hf_tokenizer_params": {} # Additional parameters to pass to the `tokenizer` method for the Huggingface model.
      
      "hf_forward_params": {} # Additional parameters to pass to the Huggingface model's `forward` method.
  }
  ```
If you are using `bedrock` platform, the arguments must be structured as:
  ```python
  platform_args = {
      "bedrock_embedding_params": {} # Additional parameters to pass into the model when generating the embeddings.
  }
  ```
