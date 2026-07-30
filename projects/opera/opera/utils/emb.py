import os

import numpy as np

from ..constants import DOC, QUERY
from .paths import get_emb_cache_files


def compute_emb(input_texts, input_ids, data_type, dataset_name, model_name, use_cache, save_cache, config):
    emb_file, id_file = get_emb_cache_files(dataset_name, model_name, data_type)
    if use_cache:
        if os.path.exists(emb_file) and os.path.exists(id_file):
            cached_id = np.load(id_file, allow_pickle=True)
            cached_emb = np.load(emb_file)
            id_to_idx = dict(zip(cached_id, range(len(cached_id))))
            indices = [id_to_idx[id] for id in input_ids]
            output_emb = cached_emb[indices]
            print(f"Cached embeddings loaded.")
            return output_emb
        else:
            print(f"Embedding cache files not exist: {emb_file} or {id_file}. Computing embedding...")
    else:
        print(f"use_cache disabled. Computing embedding...")

    from ..models import build_model

    model = build_model(model_name, config)

    if data_type == DOC:
        output_emb = model.encode_corpus(
            input_texts, batch_size=config.optimization.per_device_train_batch_size * config.evaluate.bs_multiplier
        )
    elif data_type == QUERY:
        output_emb = model.encode_queries(
            input_texts, batch_size=config.optimization.per_device_train_batch_size * config.evaluate.bs_multiplier
        )
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    if save_cache:
        print(f"saving {output_emb.shape} embedding to {emb_file}...")
        np.save(emb_file, output_emb)
        print(f"saving {input_ids.shape} embedding ids to {id_file}...")
        np.save(id_file, input_ids)

    return output_emb
