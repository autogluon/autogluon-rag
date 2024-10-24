
# Use speculative decoding for inference

## What is Speculative Decoding?
Decoding is often a time-consuming step in large autoregressive models like Transformers. Common decoding strategies, such as greedy decoding or beam search, require running the model multiple times—generating `K` tokens can take `n*K` serial runs, where `n` is the size of the beam. Given the high cost of running large models, speculative decoding (also called assisted decoding) offers a more efficient alternative.

The key idea behind speculative decoding is to use a smaller, approximate model (called the draft or assistant model) to generate candidate tokens. These tokens are then validated in a single forward pass by the larger model, speeding up the overall decoding process. This approach achieves the same sampling quality as autoregressive decoding but with significantly reduced computation time—up to 2x faster for large models.

### How Speculative Decoding Works:
1. **Draft Model**: A smaller, more efficient model proposes tokens one at a time.
2. **Target Model Verification**: The larger model verifies these tokens in a single forward pass. It confirms correct tokens and corrects any incorrect ones.
3. **Multiple Tokens Per Pass**: Instead of generating one token per pass, speculative decoding processes multiple tokens simultaneously, reducing overall latency.

For more algorithmic details, check out the following papers:
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Speculative Sampling](https://arxiv.org/abs/2302.01318)

# Using Speculative Decoding in AutoGluon-RAG

In AutoGluon-RAG, speculative decoding can be easily enabled with a few configuration lines. This is supported for both Huggingface models and vLLM-based models. The assistant and LLM model should also share the same tokenizer to avoid re-encoding and decoding tokens.

## Speculative Decoding with Huggingface Models

In Huggingface, the draft model is referred to as the "assistant model.". In the Huggingface transformers framework, the parameter `--assistant_model` is used to specify the draft model.

To use speculative decoding in AutoGluon-RAG with Huggingface models, configure the assistant model like this:

```yaml
generator_model_name: meta-llama/Llama-3.1-8B
generator_model_platform: huggingface
generator_model_platform_args:
  hf_generate_params:
    assistant_model: meta-llama/Llama-3.2-1B
```

## Speculative Decoding with vLLM Models

Speculative decoding with vLLM is also straightforward. Here is an example configuration that sets up vLLM in offline mode to use speculative decoding with a draft model, speculating 5 tokens at a time:

```yaml
generator_model_name: facebook/opt-6.7b
generator_model_platform: vllm
generator_model_platform_args:
  vllm_init_params:
    speculative_model: facebook/opt-125m
    num_speculative_tokens: 5
```

With these configurations, AutoGluon-RAG provides an efficient way to speed up text generation while preserving the quality of the output.

**Summary**: Speculative decoding is a technique used to speed up the decoding process of large autoregressive models, such as Transformers. By using a smaller, approximate model (draft or assistant model) to propose candidate tokens and then verifying them with the larger model in a single forward pass, this method improves text generation speed while maintaining similar sampling quality. 