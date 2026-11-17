import json
import logging
import os

import datasets
import faiss
import numpy as np
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from ..models import build_model
from ..utils import get_model_save_path, get_result_path, save_result
from .utils import load_eval_data

logger = logging.getLogger(__name__)


def index(
    model,
    corpus: datasets.Dataset,
    batch_size: int = 256,
    max_length: int = 512,
    index_factory: str = "Flat",
    save_path: str = None,
    save_embedding: bool = False,
    load_embedding: bool = False,
):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(save_path, mode="r", dtype=dtype).reshape(-1, dim)

    else:
        corpus_embeddings = model.encode_corpus(corpus["doc_text"], batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]

        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(save_path, shape=corpus_embeddings.shape, mode="w+", dtype=corpus_embeddings.dtype)

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i:j] = corpus_embeddings[i:j]
            else:
                memmap[:] = corpus_embeddings

    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    # if model.device == torch.device("cuda"):
    #    # co = faiss.GpuClonerOptions()
    #    co = faiss.GpuMultipleClonerOptions()
    #    co.useFloat16 = True
    #    # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
    #    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(
    model, queries: datasets, faiss_index: faiss.Index, k: int = 100, batch_size: int = 256, max_length: int = 512
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["query_text"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i:j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices


def compute_metric(preds, labels, scores, cutoffs):
    """
    Evaluate MRR and Recall at cutoffs.
    preds: (N, args.k) of texts
    labels: (N, <dynamic>num_pos) of texts
    scores: (N, args.k) of rel scores in [0,1]
    """
    N = len(preds)
    max_cutoff = len(preds[0])
    assert max(cutoffs) <= max_cutoff, f"k {max(cutoffs)} and k {max_cutoff}"

    assert len(preds[-1]) == max_cutoff, f"shape {preds[-1]} and shape {max_cutoff}"
    assert len(labels) == N, f"shape {len(labels)} and shape {N}"
    assert len(scores) == N, f"shape {len(scores)} and shape {N}"
    assert len(scores[0]) == max_cutoff, f"shape {len(scores[0])} and shape {max_cutoff}"

    metrics = {}

    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)

    # Recall, Success, and NDCG
    recalls = np.zeros(len(cutoffs))
    successes = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
            successes[k] += int(len(recall) > 0)

    # NDCG
    ndcgs = np.zeros(len(cutoffs))
    true_labels = []
    for pred, label in zip(preds, labels):
        true_labels_per_query = []
        for pred_text in pred:
            if pred_text in label:
                true_labels_per_query.append(1)
            else:
                true_labels_per_query.append(0)
        true_labels.append(true_labels_per_query)
    for k, cutoff in enumerate(cutoffs):
        if cutoff <= 1:
            ndcgs[k] = -1
        else:
            ndcgs[k] = ndcg_score(true_labels, scores, k=cutoff)

    recalls /= len(preds)
    successes /= len(preds)

    for i, cutoff in enumerate(cutoffs):
        metrics[f"MRR@{cutoff}"] = mrrs[i]
        metrics[f"Recall@{cutoff}"] = recalls[i]
        metrics[f"Success@{cutoff}"] = successes[i]
        metrics[f"NDCG@{cutoff}"] = ndcgs[i]

    return metrics


def evaluate(encoder_name, config):
    cutoffs = config.evaluate.cutoffs
    test_bs = config.optimization.per_device_train_batch_size * config.evaluate.bs_multiplier

    if config.evaluate.criteria == "text_based":
        evaluation_key = "doc_text"
    elif config.evaluate.criteria == "id_based":
        evaluation_key = "doc_id"
        print(
            f"Use id-based evaluation (use doc id to see if matched). Please make sure THERE ARE NO DUPLICATE DOCUMENTS."
        )
    else:
        raise ValueError(f"Invalid evaluation criteria: {config.evaluate.criteria}")

    pos_dataset, corpus_dataset = load_eval_data(config)

    model = build_model(encoder_name, config)

    faiss_index = index(
        model=model,
        corpus=corpus_dataset,
        batch_size=test_bs,
        max_length=config.data.passage_max_len,
        index_factory=config.evaluate.index_factory,
        save_path=config.evaluate.save_path,
        save_embedding=config.evaluate.save_embedding,
        load_embedding=config.evaluate.load_embedding,
    )

    scores, indices = search(
        model=model,
        queries=pos_dataset,
        faiss_index=faiss_index,
        k=config.evaluate.k,
        batch_size=test_bs,
        max_length=config.data.query_max_len,
    )

    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        # indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus_dataset[indice][evaluation_key])

    ground_truths = []
    for sample in pos_dataset:
        ground_truths.append(sample[evaluation_key])

    metrics = compute_metric(retrieval_results, ground_truths, scores, cutoffs=cutoffs)

    print(metrics)
    print("\n".join([str(k) for k in metrics.keys()]))
    print("\n".join([str(v) for v in metrics.values()]))
    print(encoder_name)

    result_path = get_result_path(encoder_name, config)
    save_result(metrics, result_path)

    per_query_path = os.path.splitext(result_path)[0] + "_per_query.jsonl"
    query_ids = list(pos_dataset["query_id"])
    query_texts = list(pos_dataset["query_text"])
    with open(per_query_path, "w") as f:
        for qid, qtext, retrieved, gt, qscores in zip(
            query_ids, query_texts, retrieval_results, ground_truths, scores
        ):
            record = {
                "query_id": qid,
                "query_text": qtext,
                "ground_truth": list(gt),
                "retrieved": list(retrieved),
                "scores": [float(s) for s in qscores],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"per-query rankings saved to {per_query_path}")

    return metrics


def run_evaluations(config):
    eval_modes = config.evaluate.eval_modes

    encoder_names = []
    for eval_mode in eval_modes:
        if eval_mode == "pretrain":
            encoder_names.append(config.pretrain.model_name_or_path)
        elif eval_mode == "finetune":
            encoder_names.append(get_model_save_path(config))
        else:
            raise ValueError(f"Invalid eval_mode: {eval_mode}")

    results = {}
    for encoder_name in encoder_names:
        results[encoder_name] = evaluate(encoder_name, config)

    return results
