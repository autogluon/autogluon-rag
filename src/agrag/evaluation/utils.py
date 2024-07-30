import csv
import re
import string
from typing import List, Union

import numpy as np
from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher


def preprocess_text(
    text: str,
    regexes_to_ignore: List[str] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> str:
    """
    Preprocesses text by applying specified transformations.

    Parameters:
    ----------
    text : str
        The text to be preprocessed.
    regexes_to_ignore : List[str]
        List of regex expressions to ignore in the text.
    ignore_case : bool
        If True, turns everything to lowercase.
    ignore_punctuation : bool
        If True, removes punctuation.
    ignore_numbers : bool
        If True, removes all digits.

    Returns:
    -------
    str
        The preprocessed text.
    """
    if regexes_to_ignore:
        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)

    if ignore_case:
        text = text.lower()

    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if ignore_numbers:
        text = re.sub(r"\d+", "", text)

    return text.strip()


def inclusive_exact_match_metric(
    predictions: List[str],
    references: List[List[str]],
    regexes_to_ignore: List[str] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> List[bool]:
    """
    Inclusive exact match metric to check if predictions match the references.

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses.
    regexes_to_ignore : List[str]
        List of regex expressions to ignore in the text.
    ignore_case : bool
        If True, turns everything to lowercase.
    ignore_punctuation : bool
        If True, removes punctuation.
    ignore_numbers : bool
        If True, removes all digits.

    Returns:
    -------
    List[bool]
        A list of boolean values indicating if the prediction matches any of the references.
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."

    exact_matches = []

    for gen_resp, exp_resps in zip(predictions, references):
        gen_resp = preprocess_text(gen_resp, regexes_to_ignore, ignore_case, ignore_punctuation, ignore_numbers)
        match_found = False

        for exp_resp in exp_resps:
            exp_resp = preprocess_text(exp_resp, regexes_to_ignore, ignore_case, ignore_punctuation, ignore_numbers)

            if gen_resp == exp_resp or exp_resp in gen_resp:
                match_found = True
                break
        exact_matches.append(match_found)
    return exact_matches


def calculate_exact_match_score(exact_matches: List[bool]) -> float:
    """
    Calculates the exact match score.

    Parameters:
    ----------
    exact_matches : List[bool]
        The exact match results.

    Returns:
    -------
    float
        The exact match score.
    """
    total_responses = len(exact_matches)
    total_matches = sum(exact_matches)
    exact_match_score = total_matches / total_responses if total_responses > 0 else 0
    return exact_match_score


def qa_metric_score(
    predictions: List[str],
    references: List[List[str]],
    queries: List[str],
    qa_metric: Union[PEDANT, TransformerMatcher],
) -> float:
    """
    Computes the QA metric score for the predictions.

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses.
    queries: List[str]
        The original queries for each response.
    qa_metric : Union[PEDANT, TransformerMatcher]
        The QA metric instance to use for evaluation.

    Returns:
    -------
    float
        The average QA metric score.
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."

    scores = []
    for gen_resp, exp_resps, query in zip(predictions, references, queries):
        _, highest_score = qa_metric.get_highest_score(gen_resp, exp_resps, query)
        scores.append(highest_score)

    return np.mean(scores)


def save_responses_to_csv(
    generated_responses: List[str], expected_responses: List[List[str]], exact_matches: List[bool], output_csv: str
):
    """
    Saves the evaluation results to a CSV file.

    Parameters:
    ----------
    generated_responses : List[str]
        The generated responses.
    expected_responses : List[List[str]]
        The expected responses.
    exact_matches : List[bool]
        The exact match results.
    output_csv : str
        The path to the output CSV file.
    """
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generated Response", "Expected Responses", "Exact Match"])

        for gen_resp, exp_resps, match in zip(generated_responses, expected_responses, exact_matches):
            writer.writerow([gen_resp, "; ".join(exp_resps), match])
