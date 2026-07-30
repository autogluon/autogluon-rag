import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from .arguments import OperaTrainingArguments
from .data import BaseCollator, BaseDataset, InfoBatchCollator, InfoBatchDataset, OperaCollator, OperaQueryDataset
from .models import BaseModel, InfoBatchModel, OperaModel
from .trainer import BaseTrainer, InfoBatchTrainer, OperaTrainer
from .utils import get_model_save_path, save_config

logger = logging.getLogger(__name__)


def init_model():
    pass


def finetune(config):
    if config.opera.level == "query":
        DATASET = OperaQueryDataset
        COLLATOR = OperaCollator
        MODEL = OperaModel
        TRAINER = OperaTrainer
    elif config.opera.level == "infobatch":
        DATASET = InfoBatchDataset
        COLLATOR = InfoBatchCollator
        MODEL = InfoBatchModel
        TRAINER = InfoBatchTrainer
    elif config.opera.level == "none":
        DATASET = BaseDataset
        COLLATOR = BaseCollator
        MODEL = BaseModel
        TRAINER = BaseTrainer
    else:
        raise ValueError(f"Invalid pruning level: {config.opera.level}")

    output_dir = get_model_save_path(config)

    training_args = OperaTrainingArguments(**OmegaConf.to_container(config.optimization))
    training_args.output_dir = output_dir

    # Use non-reentrant gradient checkpointing for DDP compatibility
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # data_args
    data_args = config.data
    # data_args.query_instruction_for_retrieval = ""  # instruction is only used in evaluation

    # model_args
    model_args = config.pretrain

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
    )

    # Set padding side if specified (e.g., "left" for Qwen3-Embedding)
    if hasattr(model_args, "padding_side") and model_args.padding_side:
        tokenizer.padding_side = model_args.padding_side
        logger.info(f"Set tokenizer padding_side to: {model_args.padding_side}")

    model_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info("Model Config: %s", model_config)

    model = MODEL(
        model_name=model_args.model_name_or_path,
        normalized=training_args.normalized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_inbatch_neg=training_args.use_inbatch_neg,
    )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = DATASET(
        args=data_args,
        tokenizer=tokenizer,
        config=config,
        # max_steps=config.optimization.max_steps,
        # prune_pct=config.opera.prune_pct,
        # prune_prob=config.opera.prune_prob,
        # delta=config.opera.delta,
        # seed=config.seed,
        # dynamic_weights=config.opera.dynamic_weights
    )

    trainer = TRAINER(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=COLLATOR(
            tokenizer, query_max_len=data_args.query_max_len, passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer,
        opera_config=config,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()

    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        save_config(config=config, config_path=training_args.output_dir)


if __name__ == "__main__":
    finetune()
