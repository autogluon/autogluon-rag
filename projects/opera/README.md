# OPERA

OPERA (Online Data Pruning for Efficient Retrieval Model Adaptation) is a data pruning framework for finetuning dense retrieval models. **Not all query-document training pairs contribute equally to learning, and OPERA exploits this heterogeneity to improve both the quality and the efficiency of domain adaptation.**

The framework implements two ideas. Static pruning keeps only the highest-similarity query-document pairs before training, which improves ranking but can reduce retrieval coverage because it removes queries that have few high-quality documents. Dynamic pruning keeps the full training set and instead adjusts sampling probabilities at both the query and document level throughout training, using cosine-scheduled soft thresholds that sharpen as the model improves. **Dynamic pruning resolves the ranking-versus-coverage tradeoff and reaches much better performance, or comparable performance in less than half the training time of standard finetuning.** Full details are in the paper, which is **accepted at COLM 2026**: https://arxiv.org/abs/2603.17205.

**This project is part of our retrieval-augmented generation exploration in the AutoGluon team, but it is developed, installed, and used on its own.** This release implements the interfaces for `ir_datasets` datasets (including BEIR) and two backbones, `BAAI/bge-large-en-v1.5` and `Qwen/Qwen3-Embedding-0.6B`. Adding more datasets or models is straightforward by following the same interfaces.

## Results

These results were produced by `reproduce_fiqa_bge.sh` on 8 A100 GPUs with `learning_rate: 1e-6`, comparing five conditions on FiQA with `BAAI/bge-large-en-v1.5`: the pretrained base model with no finetuning, standard finetuning with no pruning, static pruning, the InfoBatch baseline, and dynamic pruning (OPERA). Because this open-source release runs on a different software environment than the one used for the paper (different library versions, GPU, and seed), the absolute numbers are not expected to match the paper exactly, but the qualitative findings still hold.

FiQA, `BAAI/bge-large-en-v1.5`, 2000 steps:

| Condition | Recall@20 | NDCG@10 |
|-----------|-----------|---------|
| pretrain | 0.6020 | 0.4896 |
| no pruning | 0.6399 | 0.5160 |
| sp | 0.6328 | 0.5134 |
| infobatch | 0.6375 | 0.5190 |
| dp (OPERA) | **0.6420** | **0.5197** |

FiQA, `BAAI/bge-large-en-v1.5`, 8000 steps:

| Condition | Recall@20 | NDCG@10 |
|-----------|-----------|---------|
| pretrain | 0.6020 | 0.4896 |
| no pruning | 0.6321 | 0.5126 |
| sp | 0.6317 | 0.5134 |
| infobatch | 0.6327 | 0.5115 |
| dp (OPERA) | **0.6409** | **0.5183** |

**Dynamic pruning gives the best Recall@20 and NDCG@10 at both training lengths.** It is also the most robust to longer training: as the step count grows from 2000 to 8000, standard finetuning regresses on both metrics while dynamic pruning stays essentially flat, which reflects the efficiency and robustness the method is designed for. These are single-seed runs with default hyperparameters, so small gaps between methods are within run-to-run noise; the reproducible signal is that pruning matches or beats standard finetuning and that dynamic pruning holds up best. The full comparison is reproducible with the steps in [Reproducing the FiQA Comparison](#reproducing-the-fiqa-comparison).

## Methods

OPERA compares four finetuning strategies, each defined entirely by its config.

| Method | Description | `opera.level` | Offline pruning |
|--------|-------------|---------------|-----------------|
| basic | Standard finetuning with no pruning | `none` | `kept_pct: 1.0` |
| sp | Static pruning, offline filtering only | `none` | `kept_pct < 1.0` |
| dp | Dynamic pruning, the proposed OPERA method | `query` | `kept_pct: 1.0` |
| infobatch | InfoBatch online pruning baseline | `infobatch` | `kept_pct: 1.0` |

## Installation

OPERA requires Python 3.10 or newer and a CUDA-capable GPU. Qwen3-Embedding needs `transformers>=4.51.0`; older versions raise `KeyError: 'qwen3'`.

```bash
conda create -n opera310 python=3.10 -y
conda activate opera310
cd projects/opera
pip install -e .
```

## Environment Setup

Two environment variables tell OPERA where to read data and write outputs. Both are required.

```bash
export OPERA_RAW_DATA_DIR="/path/to/data"      # raw and preprocessed datasets
export OPERA_WORK_ROOT="/path/to/checkpoints"  # checkpoints and results
```

Experiment tracking with Weights and Biases is optional. Set `WANDB_API_KEY` to enable it, or `WANDB_MODE=disabled` to turn it off. Datasets are downloaded and cached by `ir_datasets` under `~/.ir_datasets` by default; set `IR_DATASETS_HOME` to change that location.

## Quick Start

`run.sh` executes the full pipeline for one config and one dataset.

```bash
# Usage: ./run.sh <config> <save_name> <train_data> <val_data> <test_data>
./run.sh configs/opera_qwen_0.6B_dp.yaml nfcorpus_dp \
    beir/nfcorpus/train beir/nfcorpus/dev beir/nfcorpus/test
```

The number of GPUs used for finetuning is read from `NUM_GPUS` (default 8) and the number of training steps from `MAX_STEPS` (default 4000).

```bash
NUM_GPUS=4 MAX_STEPS=2000 ./run.sh configs/opera_bge_large_sp.yaml fiqa_sp \
    beir/fiqa/train beir/fiqa/dev beir/fiqa/test
```

Dataset paths are `ir_datasets` identifiers with the split appended, for example `beir/nfcorpus/train`, `beir/fiqa/dev`, or `antique/test`. Some datasets have no separate validation split, in which case the train split is passed again as the validation split. When a dataset has no labeled test split, its dev split is used for evaluation. The dataset settings used in the paper are described there.

## Reproducing the FiQA Comparison

`reproduce_fiqa_bge.sh` runs the full FiQA comparison for `BAAI/bge-large-en-v1.5` and reports Recall@20 and NDCG@10 for the five conditions in [Results](#results): the pretrained base model with no finetuning, standard finetuning with no pruning, static pruning at `kept_pct=0.5`, the InfoBatch baseline, and dynamic pruning (OPERA). It uses the FiQA splits `beir/fiqa/{train,dev,test}` with hard negatives mined from rank range 10-100.

```bash
NUM_GPUS=8 MAX_STEPS=2000 ./reproduce_fiqa_bge.sh
```

The script finetunes each condition, evaluates it, and prints a summary table. Set `WANDB_MODE=disabled` if you do not want experiment tracking. Results and a full log are written under `$OPERA_WORK_ROOT`. The reference numbers reported above were produced with `MAX_STEPS=2000` and `MAX_STEPS=8000`.

## Pipeline

The pipeline runs six tasks in order. `run.sh` chains them for you, and `launcher.py` can run any single task on its own.

```
prepare_ir -> prune -> mine -> finetune -> evaluate -> gather_results
```

| Task | Description |
|------|-------------|
| `prepare_ir` | Convert an `ir_datasets` dataset to the internal Parquet format |
| `prune` | Apply offline pruning to the training pairs |
| `mine` | Mine hard or random negatives |
| `finetune` | Finetune the retriever |
| `evaluate` | Retrieve with FAISS and compute NDCG and Recall |
| `gather_results` | Aggregate results across training steps |

Each task is invoked through `launcher.py` with a config, a save name, and the task name. Finetuning uses `torchrun` for distributed training.

```bash
python launcher.py -c configs/opera_qwen_0.6B_dp.yaml -s nfcorpus_dp \
    --train_data beir/nfcorpus/train --val_data beir/nfcorpus/dev \
    --test_data beir/nfcorpus/test -t prepare_ir

python launcher.py -c configs/opera_qwen_0.6B_dp.yaml -s nfcorpus_dp -t prune
python launcher.py -c configs/opera_qwen_0.6B_dp.yaml -s nfcorpus_dp -t mine

torchrun --nproc_per_node=8 launcher.py -c configs/opera_qwen_0.6B_dp.yaml \
    -s nfcorpus_dp -t finetune --max_steps 4000

python launcher.py -c configs/opera_qwen_0.6B_dp.yaml -s nfcorpus_dp \
    -t evaluate --max_steps 4000
python launcher.py -c configs/opera_qwen_0.6B_dp.yaml -s nfcorpus_dp -t gather_results
```

## Configs

Configs follow the pattern `opera_{model}_{method}.yaml`. Each config sets `pretrain.model_name_or_path`, `opera.level`, `offline_pruning.kept_pct`, and the pooling method and query instruction appropriate for the backbone. BGE uses CLS pooling; Qwen3-Embedding uses last-token pooling and left padding.

| Config | Model | Method |
|--------|-------|--------|
| `opera_bge_large_nopruning.yaml` | bge-large | basic |
| `opera_bge_large_sp.yaml` | bge-large | sp |
| `opera_bge_large_dp.yaml` | bge-large | dp |
| `opera_bge_large_infobatch.yaml` | bge-large | infobatch |
| `opera_qwen_0.6B_nopruning.yaml` | Qwen3-Embedding-0.6B | basic |
| `opera_qwen_0.6B_sp.yaml` | Qwen3-Embedding-0.6B | sp |
| `opera_qwen_0.6B_dp.yaml` | Qwen3-Embedding-0.6B | dp |
| `opera_qwen_0.6B_infobatch.yaml` | Qwen3-Embedding-0.6B | infobatch |

Any config value can be overridden on the command line. `--max_steps` and `--seed` are common overrides, and `--train_data`, `--val_data`, and `--test_data` set the dataset splits.

Static pruning is controlled by `offline_pruning.kept_pct`, the fraction of pairs kept. Negative mining is controlled by `mining.mode`, which is `hard` or `random`, and `mining.range_for_sampling`, the rank range for hard negatives. Online pruning behavior lives under the `opera` section, where `query_selection` and `positive_selection` set the cutoff ratios, sampling strengths, and schedules.

## Outputs

Each experiment writes to `$OPERA_WORK_ROOT/<save_name>/`. This directory holds the model checkpoints, a snapshot of the config as `opera_config.yaml`, and the evaluation results as JSON. The reported metrics include Recall and NDCG at cutoffs 1, 5, 10, 20, 50, and 100.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{fang2026opera,
  title={OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation},
  author={Fang, Haoyang and Zhang, Shuai and Ma, Yifei and Wang, Hengyi and Hu, Cuixiong and Kirchhoff, Katrin and Wang, Bernie and Karypis, George},
  journal={arXiv preprint arXiv:2603.17205},
  year={2026}
}
```
