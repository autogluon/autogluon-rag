import torch

from .base_trainer import BaseTrainer, save_ckpt_for_sentence_transformers


class OperaTrainer(BaseTrainer):

    @property
    def global_step(self):
        return self.state.global_step

    def _get_train_sampler(self, train_dataset=None):
        # train_dataset argument added for compatibility with newer transformers versions
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not len(dataset):
            return None
        else:
            return dataset.sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss
        loss_indices = outputs.loss_indices
        sim_indices = outputs.sim_indices

        # update similarity
        scores = outputs.normalized_scores
        B, N = scores.shape
        M = N // B
        assert M * B == N, f"M {M}, B: {B}, N: {N}"
        positive_scores = scores[torch.LongTensor(range(B)), torch.LongTensor(range(B)) * M].clone().detach()
        self.train_dataset.get_query_scores_from_trainer(
            losses=loss, positive_scores=positive_scores, indices=sim_indices
        )  # visualize = self.is_world_process_zero()
        self.train_dataset.get_positive_scores_from_trainer(positive_scores=positive_scores, indices=sim_indices)
        self.train_dataset.set_global_step(self.global_step)

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss
