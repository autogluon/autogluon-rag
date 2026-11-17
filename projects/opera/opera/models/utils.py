from .flag_model import FlagModel


def build_model(model_name_or_path, config, is_inference=True):
    if config.architecture == "bge":
        model = FlagModel(
            model_name_or_path,
            query_instruction_for_retrieval=config.data.query_instruction_for_retrieval,
            use_fp16=config.optimization.fp16,
        )
    elif config.architecture == "qwen":
        # Qwen3-Embedding uses last token pooling and left padding
        use_bf16 = getattr(config.optimization, "bf16", False)
        use_fp16 = config.optimization.fp16 if not use_bf16 else False
        padding_side = getattr(config.pretrain, "padding_side", "left")
        pooling_method = getattr(config.optimization, "sentence_pooling_method", "last")
        model = FlagModel(
            model_name_or_path,
            pooling_method=pooling_method,
            query_instruction_for_retrieval=config.data.query_instruction_for_retrieval,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            padding_side=padding_side,
        )
    else:
        raise ValueError(f"Invalid model architecture: {config.architecture}")

    return model
