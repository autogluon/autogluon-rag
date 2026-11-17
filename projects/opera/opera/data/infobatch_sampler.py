import math
import os.path
import random
import warnings
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterator, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .utils import load_train_data

__all__ = ["InfoBatchDataset"]


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatchDataset):
            data, indices = self._next_data()
        else:
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


class InfoBatchDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        config,
    ):
        self.dataset, self.neg_dict = load_train_data(config=config)

        self.tokenizer = tokenizer
        self.args = args

        # InfoBatch
        self.max_steps = config.optimization.max_steps
        self.soft_prune_ratio = config.opera.query_selection.cutoff_ratio_start
        self.delta = config.opera.query_selection.delta

        # if similarity is not None:
        #    self.similarity = similarity
        # else:
        #    self.similarity = np.ones([len(self.dataset)]) * 2
        # assert len(self.similarity) == len(self.dataset)
        self.scores = torch.ones([len(self.dataset)]) * 3
        # self.transform = self.dataset.transform
        self.weights = torch.ones(len(self.dataset))
        # self.save_num = 0

        self.num_pruned_samples = 0

        self.update_counts = 0

        self.device = None

    def update(self, values, query_indices):
        # print(query_indices)
        # query_indices = query_indices.cpu()
        if self.device is None:
            self.device = values.device
            self.weights = self.weights.to(self.device)
        else:
            assert self.device == values.device, f"{self.device} and {values.device} mismatch."

        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(query_indices) == batch_size, "not enough index"
        weights = self.weights[query_indices].to(values.device)
        loss_val = values.detach().clone().cpu()

        self.scores[query_indices.cpu()] = loss_val

        values.mul_(weights)
        self.update_counts += 1
        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        # np.int64 to int
        if isinstance(item, np.int64):
            item = item.item()

        query = self.dataset[item]["query_text"]
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        assert self.dataset[item]["doc_text"]
        pos = random.choice(self.dataset[item]["doc_text"])
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
        return query, passages, item

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        well_learned_mask = self.scores < self.scores.mean()

        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        # print('#well learned samples %d, #remained samples %d, len(dataset) = %d' % (np.sum(well_learned_mask), np.sum(~well_learned_mask), len(self.dataset)))
        selected_indices = np.random.choice(
            well_learned_indices, int(self.keep_ratio * len(well_learned_indices)), replace=False
        )
        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        return remained_indices

    @property
    def sampler(self):
        sampler = InfoBatchSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedInfoBatchSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.max_steps * self.delta

    @property
    def keep_ratio(self):
        return 1 - self.soft_prune_ratio

    def reset_weights(self):
        self.weights[:] = 1


class InfoBatchSampler(object):
    def __init__(self, dataset: InfoBatchDataset):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx % len(self.sample_indices)]  # TODO: to force max_steps trained

    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        # print(f"sampler __next__ called")
        return next(self.iter_obj)  # may raise StopIteration

    def __len__(self):
        # return len(self.sample_indices)
        return len(self.dataset)  # TODO: to force max_steps trained

    def __iter__(self):
        # print(f"sampler __iter__ called")
        self.reset()
        return self


class DistributedInfoBatchSampler(DistributedSampler):
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
        def __init__(self, sampler: InfoBatchSampler):
            self.dataset = sampler
            # self.indices = None

        def reset(
            self,
        ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(
        self,
        dataset: InfoBatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedInfoBatchSampler, self).__init__(sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset  # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of InfoBatch rather than InfoBatch.
        """
        # print(f"dist sampler __iter__ called")
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
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        # print(self.iter_obj)
        return self.iter_obj
