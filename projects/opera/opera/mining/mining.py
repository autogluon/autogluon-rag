import pandas as pd

from ..constants import CORPUS, DOC, NEGPAIR, POSPAIR, QUERY, TRAIN
from ..utils import get_data, load_pos_data
from .mine_hard import find_hard_neg
from .mine_random import find_random_neg


def mining(config):

    mining_model = config.mining.model_name_or_path
    if mining_model is None:
        mining_model = config.pretrain.model_name_or_path

    query_path = get_data(
        save_path=config.save_path, df_type=QUERY, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    doc_path = get_data(
        save_path=config.save_path, df_type=DOC, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    pospair_path = get_data(
        save_path=config.save_path, df_type=POSPAIR, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    corpus_path = get_data(
        save_path=config.save_path, df_type=CORPUS, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )
    negpair_path = get_data(
        save_path=config.save_path, df_type=NEGPAIR, split=TRAIN, fallback=True, dataset_name=config.data.train_data
    )

    corpus_df = pd.read_parquet(corpus_path)
    negpair_df = pd.read_parquet(negpair_path)

    if config.mining.mode == "hard":
        querywise_posdata_df = load_pos_data(
            query_path=query_path, doc_path=doc_path, pospair_path=pospair_path, pairwise=False
        )
        sample_range = config.mining.range_for_sampling.split("-")
        sample_range = [int(x) for x in sample_range]
        mined_neg_df = find_hard_neg(
            querywise_posdata_df=querywise_posdata_df,
            corpus_df=corpus_df,
            negative_number=config.mining.negative_number,
            sample_range=sample_range,
            use_gpu=False,  # never use GPU to avoid cuda OOM
            dataset_name=config.data.train_data,
            model_name=mining_model,
            enable_cache=config.mining.enable_cache,
            config=config,
        )
    elif config.mining.mode == "random":
        query_df = pd.read_parquet(query_path)
        mined_neg_df = find_random_neg(
            query_df=query_df,
            corpus_df=corpus_df,
            negative_number=config.mining.negative_number,
        )
    else:
        raise ValueError(f"Unsupported mining mode: {config.mining.mode}")

    if not config.mining.overwrite_neg and not negpair_df.empty:
        print(len(negpair_df))
        print(len(mined_neg_df))
        # Step 1: Concatenate dataframes A and B
        combined_df = pd.concat([negpair_df, mined_neg_df], ignore_index=True)

        # Step 2: Sort the combined dataframe to prioritize entries from A
        combined_df["source"] = ["A"] * len(negpair_df) + ["B"] * len(mined_neg_df)
        combined_df = combined_df.sort_values(["query_id", "source"])

        # Step 3: Group by query_id and keep the first max_doc_num_per_query entries
        neg_df = combined_df.groupby("query_id").head(config.mining.negative_number)

        # Step 4: Remove the temporary 'source' column and reset the index
        neg_df = neg_df.drop("source", axis=1).reset_index(drop=True).rename(columns={"index": "neg_pair_id"})
    else:
        neg_df = mined_neg_df
        neg_df.index = neg_df.index.set_names(["negpair_id"])

    doc_save_path = get_data(
        save_path=config.save_path, df_type=DOC, split=TRAIN, fallback=False, dataset_name=config.data.train_data
    )
    negpair_save_path = get_data(
        save_path=config.save_path, df_type=NEGPAIR, split=TRAIN, fallback=False, dataset_name=config.data.train_data
    )

    neg_df.to_parquet(negpair_save_path)
    # print(neg_df)

    # update the doc_df to include added negatives
    doc_df = pd.read_parquet(doc_path)
    doc_original_size = len(doc_df)
    if not neg_df.empty:
        used_doc_ids = list(set(doc_df.index.unique()) | set(neg_df["doc_id"].unique()))
    else:
        used_doc_ids = doc_df.index.unique()
    used_doc_df = corpus_df.loc[used_doc_ids]
    # print(used_doc_df)
    used_doc_df.to_parquet(doc_save_path)

    # print(doc_df)

    print(
        f"Mining add {len(mined_neg_df)} to {len(negpair_df)} = {len(neg_df)} negative pairs from {negpair_path} to {negpair_save_path}."
    )
    print(
        f"Mining add {len(used_doc_ids)} to {doc_original_size} = {len(doc_df)} docs from {doc_path} to {doc_save_path}."
    )
