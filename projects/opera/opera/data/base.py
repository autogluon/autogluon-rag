import math
import os
import os.path
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, TrainingArguments

from .utils import load_train_data


class BaseDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        config,
    ):
        self.dataset, self.neg_dict = load_train_data(config=config)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        self.pairwise = config.pairwise

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]["query_text"]
        # TODO: test query_instruction_for_retrieval in training
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []

        assert self.dataset[item]["doc_text"]
        if self.pairwise:
            pos = self.dataset[item]["doc_text"]
        else:  # if is not pairwise, it's a list
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
        return query, passages


@dataclass
class BaseCollator(DataCollatorWithPadding):
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
        return {"query": q_collated, "passage": d_collated}
