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


@dataclass
class OperaCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        loss_indices = [f[2] for f in features]
        sim_indices = [f[3] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        loss_indices_collated = torch.LongTensor(loss_indices)
        sim_indices_collated = torch.LongTensor(sim_indices)
        return {
            "query": q_collated,
            "passage": d_collated,
            "loss_indices": loss_indices_collated,
            "sim_indices": sim_indices_collated,
        }
