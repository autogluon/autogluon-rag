import os
import re

from .utils import get_result_path, retrieve_result_from_json_file


def gather_results(config):
    save_root = config.save_path

    main_metrics = config.evaluate.main_metrics

    # check if there's pretrain result
    pretrain_results_file = [os.path.join(save_root, f) for f in os.listdir(save_root) if f.endswith("results.json")]
    assert (
        len(pretrain_results_file) <= 1
    ), f"There are multiple pretrain results in this experiment {pretrain_results_file}"
    pretrain_results = {}
    if pretrain_results_file:
        pretrain_results_file = pretrain_results_file[0]
        pretrain_results = {pretrain_results_file: retrieve_result_from_json_file(main_metrics, pretrain_results_file)}

    # for different finetuned models in current dir:
    finetuned_model_dirs = [os.path.join(save_root, f) for f in os.listdir(save_root) if f.startswith("model_ft")]
    finetuned_results = {}
    for ft_model in finetuned_model_dirs:
        assert os.path.isdir(ft_model), f"This should be the save dir of a finetuned model: {ft_model}"
        ft_result_file = get_result_path(encoder_name=ft_model, config=None)  # do not need to pass the config
        finetuned_results[ft_model] = retrieve_result_from_json_file(main_metrics, ft_result_file)

    # TODO: for checkpoints in the finetuned models

    # print results
    for m in main_metrics:
        print(f"######Results for metric: {m}######")

        # print finetuned results
        print("Finetuned results:")
        sorted_iters = [int(re.findall(r"(\d+)it", f)[0]) for f in finetuned_results.keys()]
        sorted_iters.sort()
        sorted_finetuned_models = list(finetuned_results.keys())
        sorted_finetuned_models.sort(key=lambda f: int(re.findall(r"(\d+)it", f)[0]))
        for iter in sorted_iters:
            print(iter)
        for ft_model in sorted_finetuned_models:
            print(finetuned_results[ft_model][m])
        print(f"Model for results above: {ft_model}")

        # print pretrained results
        print("Pretrained results:")
        for pretrained_model_name in pretrain_results:
            print(pretrain_results[pretrained_model_name][m])
            print(f"Model for results above: {pretrained_model_name}")
