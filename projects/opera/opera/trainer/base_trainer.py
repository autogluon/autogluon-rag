import logging
import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import Trainer

logger = logging.getLogger(__name__)


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = "cls", normalized: bool = True):
    # Map OPERA pooling mode names to sentence-transformers names
    pooling_mode_map = {
        "last": "lasttoken",  # sentence-transformers uses 'lasttoken'
        "cls": "cls",
        "mean": "mean",
    }
    st_pooling_mode = pooling_mode_map.get(pooling_mode, pooling_mode)

    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=st_pooling_mode)
    if normalized:
        normalize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_layer], device="cpu")
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    model.save(ckpt_dir)


class BaseTrainer(Trainer):
    def __init__(self, *args, opera_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.opera_config = opera_config

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, "save"):
            raise NotImplementedError(f"MODEL {self.model.__class__.__name__} " f"does not support save interface")
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            try:
                save_ckpt_for_sentence_transformers(
                    output_dir, pooling_mode=self.args.sentence_pooling_method, normalized=self.args.normalized
                )
            except Exception as e:
                logger.warning("Could not save sentence-transformers format: %s", e)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
