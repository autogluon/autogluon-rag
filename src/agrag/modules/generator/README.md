## Generator Module

This module is responsible for generating responses based on a query and a given context using various models and libraries such as Huggingface, vLLM,  OpenAI (GPT), and AWS Bedrock.

Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html for supported models in Bedrock.\
Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html for `model_id`s to pass into the Bedrock API.

Refer to https://docs.vllm.ai/en/latest/ for more information on vLLM usage.

Refer to https://platform.openai.com/docs/models for supported models through the OpenAI API. 

Refer to https://huggingface.co/models for supported models through the HuggingFace API. 

Here are the configurable parameters for this module:

```
generator:
  generator_model_name: The name of the model to use for generating responses. Supported models include Huggingface models, GPT-2, GPT-3 and higher via OpenAI API, and AWS Bedrock models.

  generator_model_platform: The name of the platform where the model is hosted
      Currently the following models are supported.
        - GPTGenerator for GPT-3 and GPT-4 models. Platform is "openai".
        - BedrockGenerator for AWS Bedrock models. Platform is "bedrock".
        - VLLMGenerator for vLLM models. Platform is "vllm".
        - HFGenerator for HuggingFace models. Platform is "huggingface".

  generator_model_platform_args: Additional platform-specific parameters to use when initializing the model, generating text, etc.
  
  num_gpus: Number of GPUs to use for generating responses. If no value is provided, the maximum available GPUs will be used. Otherwise, the minimum of the provided value and maximum available GPUs will be used.

  generator_query_prefix: Prefix to be added to each query that will be passed into the generator.
```

#### `generator_model_platform_args` structure
If you are using `openai` platform, the arguments must be structured as:
  ```python
  generator_model_platform_args = {
      "gpt_generate_params": {}, #Additional parameters to pass to the OpenAI GPT model's `create` method.
  }
  ```

If you are using `vllm` platform, the arguments must be structured as:
  ```python
  generator_model_platform_args = {
      "vllm_sampling_params": {}, # Parameters to pass to the vLLM library's `SamplingParams` method.
  }
  ```

  If you are using `huggingface` platform, the arguments must be structured as:
  ```python
  generator_model_platform_args = {
      "hf_model_params": {}, # Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
  
      "hf_tokenizer_init_params": {}, # Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
      
      "hf_tokenizer_params": {}, # Additional parameters to pass to the `tokenizer` method for the Huggingface model.
      
      "hf_forward_params": {}, # Additional parameters to pass to the Huggingface model's `forward` method.

      "hf_generate_params": {}, # Additional parameters to pass to the Huggingface model's `generate` method.

      "local_model_path": "path/to/local_model" , # Path to a local model to use for generation.
  }
  ```

  If you are using `bedrock` platform, the arguments must be structured as:
  ```python
  generator_model_platform_args = {
      "bedrock_generate_params": {}, # Additional parameters to pass to the AWS Bedrock generate API method.

      "openai_key_file": "path/to/txt_file", # The path to the file containing the OpenAI API key.
      
      "bedrock_aws_region": "us-west-2" # AWS region where the model is hosted on Bedrock.
  }
  ```
