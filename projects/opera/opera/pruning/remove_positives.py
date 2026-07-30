import numpy as np
import pandas as pd

from ..constants import DOC, POSPAIR, QUERY, TRAIN
from ..utils import compute_emb, get_data, load_pos_data


# pairwise
def prune_similarity(pos_data, dataset_name, model_name, kept_pct, enable_cache, config, keep_hard=False):
    """
    pos_data: a DataFrame with columns "pos_pair_id", "query_id", "doc_id", "query_text", and "doc_text"
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    scores = []

    embeddings_query = compute_emb(
        input_texts=pos_data["query_text"].tolist(),
        input_ids=pos_data["query_id"].to_numpy(),
        data_type=QUERY,
        dataset_name=dataset_name,
        model_name=model_name,
        use_cache=enable_cache,
        save_cache=enable_cache,  # if use_cache then save it
        config=config,
    )
    # embeddings_1 = model.encode_queries(querys, batch_size=256)  # TODO: remove batch size hardcode
    embeddings_doc = compute_emb(
        input_texts=pos_data["doc_text"].tolist(),
        input_ids=pos_data["doc_id"].to_numpy(),
        data_type=DOC,
        dataset_name=dataset_name,
        model_name=model_name,
        use_cache=enable_cache,
        save_cache=False,  # not saving docs here because it's not all the docs in the corpus
        config=config,
    )
    # embeddings_2 = model.encode_corpus(docs, batch_size=256)

    scores = np.einsum("ij,ij->i", embeddings_query, embeddings_doc)

    if keep_hard:
        top_indices = np.argsort(scores)  # for scarce data we might need to keep easy positives
    else:
        top_indices = np.argsort(scores)[::-1]  # by default, keep the hard positive (pairs with larger distance)

    kept_number = int(len(scores) * kept_pct)
    pos_data = pos_data.iloc[top_indices[:kept_number]].copy()
    pos_data["similarity"] = scores[top_indices[:kept_number]]

    return pos_data


# pairwise
def prune_random(pos_data, kept_pct):
    """
    pairs: [{"query":<query>,"doc":<doc>}, ...]
    kept_pct: kept percentage
    """
    assert kept_pct > 0 and kept_pct <= 1

    kept_number = int(len(pos_data) * kept_pct)
    kept_pairs = pos_data.sample(n=kept_number)

    return kept_pairs


# only for training
def offline_pruning(config):
    # skip model build for random pruning
    def get_model_name(config):
        offline_pruning_model = config.offline_pruning.model
        if offline_pruning_model is None:
            offline_pruning_model = config.pretrain.model_name_or_path
        return offline_pruning_model

    pruning_mode = config.offline_pruning.mode
    kept_pct = config.offline_pruning.kept_pct

    query_path = get_data(
        save_path=config.save_path, df_type=QUERY, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    doc_path = get_data(
        save_path=config.save_path, df_type=DOC, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    pospair_path = get_data(
        save_path=config.save_path, df_type=POSPAIR, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )

    query_save_path = get_data(
        save_path=config.save_path, df_type=QUERY, split=TRAIN, fallback=False, dataset_name=config.data.train_data
    )
    pospair_save_path = get_data(
        save_path=config.save_path, df_type=POSPAIR, split=TRAIN, fallback=False, dataset_name=config.data.train_data
    )

    pos_data = load_pos_data(query_path=query_path, doc_path=doc_path, pospair_path=pospair_path, pairwise=True)
    pos_data_original_size = len(pos_data)

    if pruning_mode == "ep":
        model_name = get_model_name(config)
        pos_data = prune_similarity(
            pos_data=pos_data,
            dataset_name=config.data.train_data,
            model_name=model_name,
            kept_pct=kept_pct,
            enable_cache=config.offline_pruning.enable_cache,
            config=config,
            keep_hard=False,
        )
    elif pruning_mode == "hp":
        model_name = get_model_name(config)
        pos_data = prune_similarity(
            pos_data=pos_data,
            dataset_name=config.data.train_data,
            model_name=model_name,
            kept_pct=kept_pct,
            enable_cache=config.offline_pruning.enable_cache,
            config=config,
            keep_hard=True,
        )
    elif pruning_mode == "ran":
        pos_data = prune_random(pos_data=pos_data, kept_pct=kept_pct)
    else:
        raise ValueError(f"dedup_mode is not supported: {pruning_mode}")

    # print(pos_data)
    pos_data.drop(["query_text", "doc_text"], axis=1).to_parquet(pospair_save_path)

    # update the query to save mining time
    query_df = pd.read_parquet(query_path)
    query_original_size = len(query_df)
    used_query_ids = pos_data["query_id"].unique()  # only include queries with positive pairs
    query_df = query_df.loc[used_query_ids]
    query_df.to_parquet(query_save_path)

    print(
        f"Offline pruning got {len(pos_data)} out of {pos_data_original_size} positives pairs from {pospair_path} to {pospair_save_path}."
    )
    print(
        f"Offline pruning got {len(query_df)} out of {query_original_size} queries from {query_path} to {query_save_path}."
    )
