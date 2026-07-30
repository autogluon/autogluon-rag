import os
import random

from datasets import Dataset
from tqdm import tqdm

from ..constants import DOC, NEGPAIR, POSPAIR, QUERY, TRAIN
from ..utils import get_data, load_neg_data, load_pos_data


def load_train_data(config):
    query_path = get_data(
        save_path=config.save_path, df_type=QUERY, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    doc_path = get_data(
        save_path=config.save_path, df_type=DOC, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    pospair_path = get_data(
        save_path=config.save_path, df_type=POSPAIR, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    negpair_path = get_data(
        save_path=config.save_path, df_type=NEGPAIR, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )

    pos_data = load_pos_data(
        query_path=query_path, doc_path=doc_path, pospair_path=pospair_path, pairwise=config.pairwise
    )
    neg_data = load_neg_data(query_path=query_path, doc_path=doc_path, negpair_path=negpair_path)

    pos_dataset = Dataset.from_pandas(
        pos_data.reset_index()
    )  # ['pos_pair_id', 'query_id', 'doc_id', 'similarity', 'query_text', 'doc_text']
    neg_dict = neg_data.to_dict(
        orient="index"
    )  # {query_id: dict_keys(['negpair_id', 'doc_id', 'query_text', 'doc_text'])}

    return pos_dataset, neg_dict
