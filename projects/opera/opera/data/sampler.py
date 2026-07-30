import math
import random
import warnings
from operator import itemgetter
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from ..constants import LOSS_OFFSET, NUMERICAL_EPSILON, SCORE_PADDING_VALUE
from .scheduler import PruneScheduler
from .utils import load_train_data
from .weighting import score_to_weights

__all__ = ["OperaQueryDataset"]


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and self._num_yielded > self._IterableDataset_len_called
        ):
            warn_msg = (
                "Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                "samples have been fetched. "
            ).format(self._dataset, self._IterableDataset_len_called, self._num_yielded)
            if self._num_workers > 0:
                warn_msg += (
                    "For multiprocessing data-loading, this could be caused by not properly configuring the "
                    "IterableDataset replica at each worker. Please see "
                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                )
            warnings.warn(warn_msg)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


class OperaQueryDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        config,
    ):
        self.dataset, self.neg_dict = load_train_data(config=config)

        self.tokenizer = tokenizer
        self.args = args
        self.config = config

        self.seed = config.seed
        self.max_steps = config.optimization.max_steps
        self.pairwise = config.pairwise

        self.global_step = 0
        self.last_query_update_step = -9999
        self.last_doc_update_step = -9999

        assert not self.pairwise, "pairwise not supported yet, need to combine loss and sim pruning"

        # Online Pruning
        self.query_selection_config = config.opera.query_selection
        self.query_selection_sched = PruneScheduler(
            sched_type=self.query_selection_config.scheduler,
            start=self.query_selection_config.sampling_strength_start,
            end=self.query_selection_config.sampling_strength_end,
            max_steps=self.max_steps,
            delta=self.query_selection_config.delta,
        )
        self.positive_selection_config = config.opera.positive_selection
        self.positive_selection_sched = PruneScheduler(
            sched_type=self.positive_selection_config.scheduler,
            start=self.positive_selection_config.cutoff_ratio_start,
            end=self.positive_selection_config.cutoff_ratio_end,
            max_steps=self.max_steps,
            delta=self.positive_selection_config.delta,
        )
        self.query_selection_scores, self.positive_selection_scores, self.pair_id_to_score_index = self._init_scores()
        self.updated_query_selection_scores = self.query_selection_scores.detach().clone().cpu()
        self.updated_positive_selection_scores = self.positive_selection_scores.detach().clone().cpu()

        self.pos_idx_offset, self.pos_segments, self.num_pos_pairs = self._init_offset(docs_key="doc_text")
        assert (
            self.pos_idx_offset[-1] == self.num_pos_pairs
        ), f"{self.pos_idx_offset[-1]} and {self.num_pos_pairs} not equal"
        assert (
            sum(self.pos_segments) == self.num_pos_pairs
        ), f"{self.pos_segments.sum()} and {self.num_pos_pairs} not equal"

        self.n = len(self.dataset)
        # n0 = n(1-r)/x + rn
        self.n0 = (
            self.n
            * (1 - self.query_selection_config.cutoff_ratio_start)
            / self.query_selection_config.sampling_strength_start
            + self.query_selection_config.cutoff_ratio_start * self.n
        )
        self.n0 = int(self.n0)

        print(f"n: {self.n}")
        print(f"n0: {self.n0}")

        self.device = None
        self.pruning_step = 0

        self.update_query_scores()
        self.update_positive_scores()
        self.prune_queries()

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _init_scores(
        self,
    ):
        # print(self.dataset[0])
        assert "similarity" in self.dataset[0]

        pair_id_to_score_index = {}
        similarities = []
        current_index = 0
        for row in iter(self.dataset):
            if not self.pairwise:
                for i, pos_pair_id in enumerate(row["pos_pair_id"]):
                    pair_id_to_score_index[pos_pair_id] = current_index
                    similarities.append(row["similarity"][i])
                    current_index += 1
            else:
                pair_id_to_score_index[row["pos_pair_id"]] = current_index
                similarities.append(row["similarity"])
                current_index += 1

        positive_selection_scores = torch.FloatTensor(similarities)
        if self.query_selection_config.criteria == "loss":
            if self.query_selection_config.pooling == "mean":
                query_selection_scores = torch.full_like(
                    positive_selection_scores, fill_value=SCORE_PADDING_VALUE, dtype=torch.float
                )  # TODO: a better init for mean?
            elif self.query_selection_config.pooling == "min":
                query_selection_scores = torch.full_like(
                    positive_selection_scores, fill_value=-SCORE_PADDING_VALUE, dtype=torch.float
                )  # TODO: a better init for mean?
            elif self.query_selection_config.pooling == "max":
                query_selection_scores = torch.full_like(
                    positive_selection_scores, fill_value=SCORE_PADDING_VALUE, dtype=torch.float
                )  # TODO: a better init for mean?
            else:
                raise ValueError(
                    f"Invalid value for self.query_selection_config.pooling: {self.query_selection_config.pooling}"
                )
        elif self.query_selection_config.criteria == "sim":
            query_selection_scores = torch.FloatTensor(similarities)
        else:
            raise ValueError(
                f"Invalid value for self.query_selection_config.criteria: {self.query_selection_config.criteria}"
            )

        return query_selection_scores, positive_selection_scores, pair_id_to_score_index

    def _init_offset(self, docs_key="doc_text"):
        offsets = [0]
        total_count = 0
        segments = []
        for row in iter(self.dataset):
            new_offset = offsets[-1] + len(row[docs_key])
            offsets.append(new_offset)
            segments.append(len(row[docs_key]))
            total_count += len(row[docs_key])
        return torch.LongTensor(offsets), segments, total_count

    def _get_query_cutoff_ratio_by_strength(self, sampling_strength):
        # r = (xn0 - n) / (n(x - 1))
        cutoff_ratio = (sampling_strength * self.n0 - self.n) / (self.n * (sampling_strength - 1))

        # try:
        wandb.log(
            {
                "query_sampling_strength": sampling_strength,
                "cutoff_ratio": cutoff_ratio,
            },
            step=self.global_step,
        )
        # except:
        #    print(f"logging failed")
        return cutoff_ratio

    def update_query_scores(
        self,
    ):
        self.query_selection_scores = self.updated_query_selection_scores.detach().clone().cpu()
        sampling_strength = self.query_selection_sched.get_prune_kept_ratio(current_step=self.global_step)
        cutoff_ratio = self._get_query_cutoff_ratio_by_strength(sampling_strength)

        if self.query_selection_config.pooling == "mean":
            padding = SCORE_PADDING_VALUE
            per_query_scores = torch.nn.utils.rnn.pad_sequence(
                torch.split(self.query_selection_scores, self.pos_segments), batch_first=True, padding_value=padding
            )
            valid_mask = per_query_scores > padding
            query_scores = torch.sum(per_query_scores * valid_mask, axis=1) / (
                torch.sum(valid_mask, axis=1) + NUMERICAL_EPSILON
            )
        else:
            raise NotImplementedError

        if self.query_selection_config.keep_higher:  # for  ce loss, the higher the loss, the harder the sample
            self.query_threshold = torch.quantile(query_scores, 1 - cutoff_ratio)  # remove the low loss pairs
        else:  # for ce loss, the lower the loss, the easier the sample
            self.query_threshold = torch.quantile(query_scores, cutoff_ratio)  # remove the high loss pairs

        wandb.log(
            {
                "#query_threshold": self.query_threshold,
            },
            step=self.global_step,
        )

    def update_positive_scores(
        self,
    ):

        self.positive_selection_scores = self.updated_positive_selection_scores.detach().clone().cpu()
        cutoff_ratio = self.positive_selection_sched.get_prune_kept_ratio(current_step=self.global_step)
        if (
            self.positive_selection_config.keep_higher
        ):  # for cosine similarity, the higher the similarity, the easier the sample
            self.positive_threshold = torch.quantile(
                self.positive_selection_scores, 1 - cutoff_ratio
            )  # remove the low similarity pairs
        else:  # for cosine similarity, the lower the similarity, the harder the sample
            self.positive_threshold = torch.quantile(
                self.positive_selection_scores, cutoff_ratio
            )  # remove the high similarity pairs

        wandb.log(
            {
                "#positive_threshold": self.positive_threshold,
            },
            step=self.global_step,
        )

    def prune_queries(
        self,
    ):
        # prune with similarity
        # TODO: + Pooling
        if self.query_selection_config.pooling == "mean":
            padding = SCORE_PADDING_VALUE
            per_query_scores = torch.nn.utils.rnn.pad_sequence(
                torch.split(self.query_selection_scores, self.pos_segments), batch_first=True, padding_value=padding
            )
            valid_mask = per_query_scores > padding
            query_scores = torch.sum(per_query_scores * valid_mask, axis=1) / (
                torch.sum(valid_mask, axis=1) + NUMERICAL_EPSILON
            )

            if self.query_selection_config.keep_higher:  # higher
                kept_mask = query_scores >= self.query_threshold
            else:
                kept_mask = query_scores <= self.query_threshold

            kept_mask = kept_mask.numpy()
            # try:
            #    wandb.log({
            #        "query_threshold": self.query_threshold,
            #        "query_scores>0 counts": torch.sum(query_scores > 0),
            #        "#kept": np.sum(kept_mask),
            #    })
            # except:
            #    print(f"skip wandb logging while init...")

        else:
            raise NotImplementedError

        self.soft_remove_indices = np.where(~kept_mask)[0]
        self.remained_indices = np.where(kept_mask)[0]

    def get_positive_scores_from_trainer(self, positive_scores, indices):
        if self.device is None:
            self.device = positive_scores.device
        else:
            assert self.device == positive_scores.device, f"{self.device} and {positive_scores.device} mismatch."

        assert isinstance(positive_scores, torch.Tensor)
        assert len(indices) == positive_scores.shape[0], "not enough index"

        self.updated_positive_selection_scores[indices.cpu()] = positive_scores.detach().clone().cpu()

    def get_query_scores_from_trainer(self, losses, positive_scores, indices):
        if self.query_selection_config.criteria == "loss":
            scores = losses + LOSS_OFFSET
        elif self.query_selection_config.criteria == "sim":
            scores = positive_scores
        else:
            raise ValueError(
                f"Invalid value for self.query_selection_config.criteria: {self.query_selection_config.criteria}"
            )

        if self.device is None:
            self.device = scores.device
        else:
            assert self.device == scores.device, f"{self.device} and {scores.device} mismatch."

        assert isinstance(scores, torch.Tensor)
        assert len(indices) == scores.shape[0], "not enough index"

        self.updated_query_selection_scores[indices.cpu()] = scores.detach().clone().cpu()

    def __len__(self):
        return self.n0

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        # TODO
        # np.int64 to int
        if isinstance(item, np.int64):
            item = item.item()

        query = self.dataset[item]["query_text"]
        # TODO: test query_instruction_for_retrieval in training
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        assert self.dataset[item]["doc_text"]
        if self.pairwise:  # TODO: pairwise not supported yet
            pos = self.dataset[item]["doc_text"]
            pos_pair_id = self.dataset[item]["pos_pair_id"]
        else:  # if is not pairwise, it's a list
            pos_sims = self.positive_selection_scores[
                [self.pair_id_to_score_index[pos_pair_id] for pos_pair_id in self.dataset[item]["pos_pair_id"]]
            ]
            pos_weights = score_to_weights(
                scores=pos_sims,
                mode="soft_thresholded",
                params={
                    "threshold": self.positive_threshold,
                    "keep_higher": self.positive_selection_config.keep_higher,
                    "sampling_strength": self.positive_selection_config.sampling_strength,
                },
                use_reciprocal=False,
            )
            # print(self.sim_threshold)
            # print(pos_sims)
            # print(pos_weights)
            select_idx = np.random.choice(range(len(self.dataset[item]["doc_text"])), p=pos_weights)
            pos = self.dataset[item]["doc_text"][select_idx]
            pos_pair_id = self.dataset[item]["pos_pair_id"][select_idx]
        passages.append(pos)

        query_id = self.dataset[item]["query_id"]

        if len(self.neg_dict[query_id]["doc_text"]) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.neg_dict[query_id]["doc_text"]))
            negs = random.sample(self.neg_dict[query_id]["doc_text"] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.neg_dict[query_id]["doc_text"], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval + p for p in passages]

        score_index = self.pair_id_to_score_index[pos_pair_id]
        query_index = item

        return query, passages, query_index, score_index

    def prune(
        self,
    ):
        update_query_scores = (
            self.global_step - self.last_query_update_step
        ) >= self.query_selection_config.reset_interval
        update_positive_scores = (
            self.global_step - self.last_doc_update_step
        ) >= self.positive_selection_config.reset_interval
        if update_query_scores:
            self.update_query_scores()
            self.prune_queries()
            self.last_query_update_step = self.global_step
        if update_positive_scores:
            self.update_positive_scores()
            self.last_doc_update_step = self.global_step

        if len(self) > len(self.remained_indices):
            num_to_select = len(self) - len(self.remained_indices)
            # print('#well learned samples %d, #remained samples %d, len(dataset) = %d' % (np.sum(well_learned_mask), np.sum(~well_learned_mask), len(self.dataset)))
            selected_indices = np.random.choice(self.soft_remove_indices, num_to_select, replace=False)
            final_indices = np.concatenate((self.remained_indices, selected_indices))
        else:
            selected_indices = []
            final_indices = self.remained_indices

        np.random.seed(self.global_step + self.seed + self.pruning_step)
        np.random.shuffle(final_indices)
        self.pruning_step += 1
        # print(f"final indices: {final_indices}")

        # print(current_step)
        # print(self.loss_prune_config.reset_interval)
        # if self.global_step % self.query_selection_config.reset_interval == 0:
        #    print(f"Keep {len(self.remained_indices)}+{len(selected_indices)}={len(final_indices)} data out of {len(self.dataset)}.")
        wandb.log(
            {
                "#remained_indices": len(self.remained_indices),
                "#selected_indices": len(selected_indices),
                "#final_indices": len(final_indices),
                "pruning_step": self.pruning_step,
            },
            step=self.global_step,
        )

        return final_indices

    @property
    def sampler(self):
        sampler = OperaQuerySampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedOperaQuerySampler(sampler)
        return sampler


class OperaQuerySampler(object):
    def __init__(self, dataset: OperaQueryDataset):
        self.dataset = dataset
        self.sample_indices = None
        self.iter_obj = None
        self.seed = self.dataset.seed
        self.reset(init_reset=True)

    @property
    def iterations(self):
        return self.dataset.global_step

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self, init_reset=False):
        np.random.seed(self.iterations + self.seed)
        self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)

    def __next__(self):
        return next(self.iter_obj)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self.reset()
        return self


class DistributedOperaQuerySampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: OperaQuerySampler):
            self.dataset = sampler
            self.seed = sampler.seed

        def reset(self):
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            return self.dataset[index]

    def __init__(
        self,
        dataset: OperaQuerySampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedOperaQuerySampler, self).__init__(
            sampler, num_replicas, rank, shuffle, sampler.seed, drop_last
        )
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset  # the real dataset.
        self.seed = sampler.seed
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj
