import argparse

# ir_datasets caches downloaded datasets under ~/.ir_datasets by default.
# Set the IR_DATASETS_HOME environment variable to use a different location.
import ir_datasets
import pandas as pd
from tqdm import tqdm

from ..constants import CORPUS, DOC, NEGPAIR, POSPAIR, QUERY
from ..utils import get_raw_data, load_config, seed_everything


def prepare_ir_data(dataset_name, pos_threshold):
    dataset = ir_datasets.load(dataset_name)

    corpus_path = get_raw_data(dataset_name=dataset_name, df_type=CORPUS)
    doc_path = get_raw_data(dataset_name=dataset_name, df_type=DOC)
    negpair_path = get_raw_data(dataset_name=dataset_name, df_type=NEGPAIR)
    pospair_path = get_raw_data(dataset_name=dataset_name, df_type=POSPAIR)
    query_path = get_raw_data(dataset_name=dataset_name, df_type=QUERY)

    corpus_data = []
    doc_id_converter = {}

    current_id = 0
    for doc in tqdm(dataset.docs_iter()):
        doc_id_converter[doc.doc_id] = current_id
        corpus_data.append({"doc_id": current_id, "doc_text": doc.text})
        current_id += 1

    corpus_df = pd.DataFrame(corpus_data)
    assert corpus_df["doc_id"].is_unique
    corpus_df = corpus_df.set_index("doc_id")
    # print(corpus_df)
    corpus_df.to_parquet(corpus_path)

    query_data = []
    pos_pairs = []
    neg_pairs = []
    pos_pair_id = 0
    neg_pair_id = 0

    for query in dataset.queries_iter():
        qid = query.query_id
        query_text = query.text
        query_data.append({"query_id": qid, "query_text": query_text})

    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        if qrel.doc_id in doc_id_converter:
            did = doc_id_converter[qrel.doc_id]
        else:
            print(f"Invalid doc id: {qrel.doc_id}")
            continue
        rel = qrel.relevance

        if int(rel) > pos_threshold:
            pos_pairs.append({"pos_pair_id": pos_pair_id, "query_id": qid, "doc_id": did})
            pos_pair_id += 1
        else:
            neg_pairs.append({"neg_pair_id": neg_pair_id, "query_id": qid, "doc_id": did})
            neg_pair_id += 1

    pos_df = pd.DataFrame(pos_pairs).set_index("pos_pair_id")
    # print(pos_df)
    pos_df.to_parquet(pospair_path)

    query_df = pd.DataFrame(query_data)
    assert query_df["query_id"].is_unique
    query_df = query_df.set_index("query_id")
    used_query_ids = pos_df["query_id"].unique()  # only include queries with positive pairs
    query_df = query_df.loc[used_query_ids]
    # print(query_df)
    query_df.to_parquet(query_path)

    neg_df = pd.DataFrame(neg_pairs)
    if not neg_df.empty:
        neg_df = neg_df.set_index("neg_pair_id")
    # print(neg_df)
    neg_df.to_parquet(negpair_path)

    if not neg_df.empty:
        used_doc_ids = list(set(pos_df["doc_id"].unique()) | set(neg_df["doc_id"].unique()))
    else:
        used_doc_ids = pos_df["doc_id"].unique()
    used_doc_df = corpus_df.loc[used_doc_ids]
    # print(used_doc_df)
    used_doc_df.to_parquet(doc_path)


def prepare_ir_all_splits(config):
    for dataset_name in set(
        [config.data.train_data, config.data.val_data, config.data.test_data]
    ):  # training set and val set could be the same
        print(f"Preparing {dataset_name}...")
        pos_threshold = config.data.pos_threshold
        if "antique" in dataset_name.lower() and "test" in dataset_name.lower():
            pos_threshold = 2.5
        prepare_ir_data(dataset_name=dataset_name, pos_threshold=pos_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    args = parser.parse_args()
    config = load_config(args.cfg)

    seed_everything(config)

    prepare_ir_all_splits(config)
