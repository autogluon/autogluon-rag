import os
import random

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from ..constants import CORPUS, DOC, NEGPAIR, POSPAIR, QUERY, TEST, TRAIN
from ..utils import get_data, load_pos_data


def load_eval_data(config):
    query_path = get_data(
        save_path=config.save_path, df_type=QUERY, split=TEST, fallback=True, dataset_name=config.data.test_data
    )
    doc_path = get_data(
        save_path=config.save_path, df_type=DOC, split=TEST, fallback=True, dataset_name=config.data.test_data
    )
    pospair_path = get_data(
        save_path=config.save_path, df_type=POSPAIR, split=TEST, fallback=True, dataset_name=config.data.test_data
    )
    corpus_path = get_data(
        save_path=config.save_path, df_type=CORPUS, split=TEST, fallback=True, dataset_name=config.data.test_data
    )

    pos_data = load_pos_data(query_path=query_path, doc_path=doc_path, pospair_path=pospair_path, pairwise=False)

    corpus_df = pd.read_parquet(corpus_path)

    pos_dataset = Dataset.from_pandas(
        pos_data.reset_index()
    )  # ['pos_pair_id', 'query_id', 'doc_id', 'query_text', 'doc_text']
    corpus_dataset = Dataset.from_pandas(corpus_df.reset_index())  # ['doc_id', 'doc_text']

    return pos_dataset, corpus_dataset
