import json
import os.path
import random

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..constants import DOC, QUERY
from ..utils import compute_emb


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index, query, topk: int = 200, batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index : start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def find_hard_neg(
    querywise_posdata_df,
    corpus_df,
    negative_number,
    sample_range,
    use_gpu,
    dataset_name,
    model_name,
    enable_cache,
    config,
):
    assert querywise_posdata_df.index.name == "query_id"
    assert corpus_df.index.name == "doc_id"
    # corpus_df = corpus_df.reset_index()

    # print(f'inferencing embedding for corpus (number={len(corpus_df)})--------------')
    # p_vecs = model.encode_corpus(corpus_df['doc_text'].tolist(), batch_size=256)
    embeddings_doc = compute_emb(
        input_texts=corpus_df["doc_text"].tolist(),
        input_ids=corpus_df.index.to_numpy(),
        data_type=DOC,
        dataset_name=dataset_name,
        model_name=model_name,
        use_cache=enable_cache,
        save_cache=enable_cache,  # not saving docs here because it's not all the docs in the corpus
        config=config,
    )
    # print(f'inferencing embedding for queries (number={len(querywise_posdata_df)})--------------')
    # q_vecs = model.encode_queries(querywise_posdata_df['query_text'].tolist(), batch_size=1024)
    embeddings_query = compute_emb(
        input_texts=querywise_posdata_df["query_text"].tolist(),
        input_ids=querywise_posdata_df.index.to_numpy(),
        data_type=QUERY,
        dataset_name=dataset_name,
        model_name=model_name,
        use_cache=enable_cache,
        save_cache=enable_cache,  # if use_cache then save it
        config=config,
    )

    print("create index and search------------------")
    index = create_index(embeddings_doc, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, embeddings_query, topk=sample_range[-1])
    assert len(all_inxs) == len(querywise_posdata_df)

    neg_data_query_id = []
    neg_data_doc_id = []
    querywise_posdata_dict = querywise_posdata_df.to_dict("index")
    corpus_dict = corpus_df.to_dict("index")
    for i, query_id in tqdm(enumerate(querywise_posdata_dict)):
        doc_ids = querywise_posdata_dict[query_id]["doc_id"]
        query_text = querywise_posdata_dict[query_id]["query_text"]

        filtered_inx = [
            inx
            for inx in all_inxs[i][sample_range[0] : sample_range[1]]
            if inx != -1
            and corpus_df.index[inx] not in doc_ids
            and corpus_dict[corpus_df.index[inx]]["doc_text"] != query_text
        ]

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)

        neg_data_doc_id.extend(corpus_df.index[inx] for inx in filtered_inx)
        neg_data_query_id.extend([query_id] * len(filtered_inx))

    neg_df = pd.DataFrame({"query_id": neg_data_query_id, "doc_id": neg_data_doc_id})

    return neg_df
