import argparse
import copy
import os

import wandb
from opera.eval import run_evaluations
from opera.gather_results import gather_results
from opera.mining import mining
from opera.preprocess import prepare_ir_all_splits
from opera.pruning import offline_pruning
from opera.train import finetune
from opera.utils import get_save_path, load_config, save_config, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opera Launcher")
    parser.add_argument("-c", "--basecfg", type=str, help="The base configuration file for the Opera run.")
    parser.add_argument("-t", "--task", type=str, help="The task to run.")
    parser.add_argument("-s", "--save_name", type=str, help="The save path for the Opera run.")
    parser.add_argument("--train_data", default=None, type=str, help="Path to train_data.")
    parser.add_argument("--val_data", default=None, type=str, help="Path to val_data.")
    parser.add_argument("--test_data", default=None, type=str, help="Path to test_data.")
    parser.add_argument("--train_corpus", default=None, type=str, help="Path to train_data corpus.")
    parser.add_argument("--val_corpus", default=None, type=str, help="Path to val_data corpus.")
    parser.add_argument("--test_corpus", default=None, type=str, help="Path to test_data corpus.")
    # general
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    # model
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Override pretrained model path.")
    # task: finetune
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps.")
    args = parser.parse_args()

    if args.max_steps is not None:
        wandb.init(name=f"{args.save_name}_{args.max_steps}iters")

    # Get the save path for current experiment
    save_path = get_save_path(save_name=args.save_name)
    experiment_config_path = os.path.join(save_path, "opera_config.yaml")

    # Load the config of the current experiment if it exists, otherwise load the
    # base config (this is the first task of the experiment).
    if os.path.exists(experiment_config_path):
        config = load_config(experiment_config_path)
    else:
        config = load_config(args.basecfg)

    # Write dataset paths into the config.
    # Use the training set for validation when no validation set is provided.
    if not args.val_data:
        args.val_data = args.train_data
    if args.train_data:
        config.data.train_data = args.train_data
    if args.val_data:
        config.data.val_data = args.val_data
    if args.test_data:
        config.data.test_data = args.test_data

    # Write seed to config
    if config.seed is None:
        config.seed = args.seed

    # Override model path if provided
    if args.model_name_or_path:
        config.pretrain.model_name_or_path = args.model_name_or_path

    task = args.task

    # Write save_path to config
    if not config.save_path:
        config.save_path = save_path

    # Overwrite max_steps so large-scale experiments can be driven from a single script.
    if args.max_steps:
        config.optimization.max_steps = args.max_steps

    seed_everything(config)

    if task == "prepare_ir":
        prepare_ir_all_splits(config=copy.deepcopy(config))
    elif task == "prune":
        offline_pruning(config=copy.deepcopy(config))
    elif task == "mine":
        mining(config=copy.deepcopy(config))
    elif task == "finetune":
        assert config.save_path, f"Invalid save_path: {config.save_path}"
        finetune(config=copy.deepcopy(config))
    elif task == "evaluate":
        run_evaluations(config=copy.deepcopy(config))
    elif task == "gather_results":
        gather_results(config=copy.deepcopy(config))
    else:
        raise ValueError(f"Task {task} is not supported.")

    if experiment_config_path and int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        save_config(config, experiment_config_path)
