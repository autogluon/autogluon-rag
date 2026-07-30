import json
import os
from collections import defaultdict

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


def load_config(config_path):
    config = OmegaConf.load(config_path)

    return config


def save_config(config, config_path):
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "opera_config.yaml")
    config = OmegaConf.save(config=config, f=config_path)
    print(f"config is saved to {config_path}")


def save_result(obj, path: str):
    # if not os.path.exists(path):
    #    os.makedirs(path)
    with open(path, "w") as f:
        return json.dump(obj, f, ensure_ascii=False)


def load_pos_data(query_path, doc_path, pospair_path, pairwise=False):
    query_df = pd.read_parquet(query_path)
    doc_df = pd.read_parquet(doc_path)
    pospair_df = pd.read_parquet(pospair_path).reset_index()

    # print(query_df)
    # print(doc_df)
    # print(pospair_df)
    # Merge positive_pairs with query dataframe
    pos_data = pd.merge(pospair_df, query_df, on="query_id", how="left")
    # Merge the resulting dataframe with doc dataframe
    pos_data = pd.merge(pos_data, doc_df, on="doc_id", how="left")

    if not pairwise:
        pos_data = pos_data.groupby("query_id").agg(lambda x: x.iloc[0] if x.name == "query_text" else list(x))
    else:
        pos_data = pos_data.set_index("pos_pair_id")

    return pos_data


def load_neg_data(query_path, doc_path, negpair_path):
    query_df = pd.read_parquet(query_path)
    doc_df = pd.read_parquet(doc_path)
    negpair_df = pd.read_parquet(negpair_path).reset_index()

    # Merge positive_pairs with query dataframe
    neg_data = pd.merge(negpair_df, query_df, on="query_id", how="left")
    # Merge the resulting dataframe with doc dataframe
    neg_data = pd.merge(neg_data, doc_df, on="doc_id", how="left")

    # negative pair data always loaded query wise
    neg_data = neg_data.groupby("query_id").agg(lambda x: x.iloc[0] if x.name == "query_text" else list(x))

    return neg_data


def retrieve_result_from_json_file(main_metrics, result_path):
    with open(result_path, "r") as f:
        results = json.load(f)
    return {m: results[m] for m in main_metrics}
