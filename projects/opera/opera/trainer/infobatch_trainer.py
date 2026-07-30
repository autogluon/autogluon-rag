from .base_trainer import BaseTrainer


class InfoBatchTrainer(BaseTrainer):

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
        query_indices = outputs.query_indices
        loss = self.train_dataset.update(loss, query_indices)

        return (loss, outputs) if return_outputs else loss
