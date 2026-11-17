import numpy as np
import torch


def score_to_weights(scores, mode, params, use_reciprocal=False):
    """
    scores: torch tensor
    use_reciprocal:
    mode: softmax, or
    params: a dict with hyperparameters

    return weights: numpy array
    """
    if use_reciprocal:
        scores = 1 / scores.numpy()
    else:
        scores = scores.numpy()

    if mode == "softmax":
        return _softmax_weights(scores=scores, params=params)
    elif mode == "thresholded":
        return _thresholded_weights(scores=scores, params=params)
    elif mode == "soft_thresholded":
        return _soft_thresholded_weights(scores=scores, params=params)
    else:
        raise ValueError(f"mode {mode} not supported in score_to_weights")


def get_param(params, key):
    if key not in params:
        raise ValueError(f"Do not have key {key} in provided parameters: {params}.")
    else:
        return params[key]


def _softmax_weights(scores, params):
    temperature = get_param(params=params, key="temperature")
    scores = scores / temperature
    weights = np.exp(scores - np.max(scores))
    weights = weights / weights.sum(axis=0)

    return weights


def _thresholded_weights(scores, params):
    threshold = get_param(params=params, key="threshold")
    keep_higher = get_param(params=params, key="keep_higher")
    # print(f"threshold: {threshold}")
    # print(f"scores: {scores}")
    # print(f"threshold.item: {threshold.item()}")
    if keep_higher:
        weights = scores > threshold.item()
    else:
        weights = scores < threshold.item()
    # print(f"bool weights: {weights}")
    weights = weights + 1e-6  # avoid all zeros
    # print(f"eps weights: {weights}")
    weights = weights / np.sum(weights)
    # print(f"normed weights: {weights}")

    return weights


def _soft_thresholded_weights(scores, params):
    threshold = get_param(params=params, key="threshold")
    keep_higher = get_param(params=params, key="keep_higher")
    sampling_strength = get_param(params=params, key="sampling_strength")
    # print(f"threshold: {threshold}")
    # print(f"scores: {scores}")
    # print(f"threshold.item: {threshold.item()}")
    if keep_higher:
        weights = scores > threshold.item()
    else:
        weights = scores < threshold.item()
    # print(f"bool weights: {weights}")
    weights = weights * (sampling_strength - 1.0) + 1.0
    # print(f"eps weights: {weights}")
    weights = weights / np.sum(weights)
    # print(f"normed weights: {weights}")

    return weights
