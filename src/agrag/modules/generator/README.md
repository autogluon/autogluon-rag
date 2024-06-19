## Generator Module

This module is responsible for generating responses based on a query and a given context using various models and libraries such as Huggingface, OpenAI, and AWS Bedrock.

Here are the configurable parameters for this module:

```
generator:
  generator_model_name: The name of the model to use for generating responses. Supported models include Huggingface models, GPT-2, GPT-3 and higher via OpenAI API, and AWS Bedrock models.
  
  num_gpus: Number of GPUs to use for generating responses (default is 0).
  
  generator_hf_model_params: Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
  generator_hf_tokenizer_params: Additional parameters to pass to the `tokenizer` method for the Huggingface model.
  
  generator_hf_tokenizer_init_params: Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
  
  generator_hf_forward_params: Additional parameters to pass to the Huggingface model's `forward` method.
  
  generator_hf_generate_params: Additional parameters to pass to the Huggingface model's `generate` method.
  
  gpt_generate_params: Additional parameters to pass to the OpenAI GPT model's `create` method.
  
  use_vllm: Whether to use the vLLM library for generating responses (default is False).
  
  vllm_sampling_params: Parameters to pass to the vLLM library's `SamplingParams` method.
  
  openai_key_file: The path to the file containing the OpenAI API key.
  
  use_bedrock: Whether to use AWS Bedrock for generating responses (default is False).
  
  bedrock_generate_params: Additional parameters to pass to the AWS Bedrock generate API method.

```