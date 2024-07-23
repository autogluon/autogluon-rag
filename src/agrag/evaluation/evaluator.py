import logging
import os
from typing import Any, Callable, Dict, List

import evaluate
import numpy as np
from datasets import load_dataset
from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher

from agrag.agrag import AutoGluonRAG
from agrag.constants import EVALUATION_DIR, EVALUATION_MAX_FILE_SIZE
from agrag.evaluation.datasets.google_natural_questions.evaluate_agrag import evaluate_rag_google_nq
from agrag.evaluation.utils import (
    calculate_exact_match_score,
    custom_exact_match_metric,
    qa_metric_score,
    save_responses_to_csv,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationModule:
    def __init__(
        self,
        agrag: AutoGluonRAG,
        dataset_name: str,
        metrics: List[str],
        preprocessing_fn: Callable,
        query_fn: Callable,
        response_fn: Callable,
        hf_dataset_params: dict = {},
        split: str = "validation",
        save_evaluation_data: bool = True,
        evaluation_dir: str = EVALUATION_DIR,
        save_csv_path: str = None,
        max_eval_size: int = None,
    ):
        """
        Initializes the EvaluationModule with the given AutoGluonRAG instance, dataset, and metrics.

        Parameters:
        ----------
        agrag : AutoGluonRAG
            The AutoGluonRAG instance to be evaluated.
        dataset_name : str
            The name of the dataset to use for evaluation.
        metrics : List[Union[str, Callable]]
            The list of metrics to use for evaluation (e.g., ["bertscore", "exact_match", "pedant", <callable_custom_metric>]).
            We currently support the following metrics:
                1. HuggingFace: bertscore: ["bertscore"]
                2. Inclusive Exact Match: ["exact_match"]
                3. QA Metrics from https://github.com/zli12321/qa_metrics
                4. Custom Metric Function: This can either be a callable Python function or a function from a Python package
        preprocessing_fn : Callable
            A function to preprocess the content before saving.
        query_fn : Callable
            A function to extract the query from the dataset row.
        response_fn : Callable
            A function to extract the expected responses from the dataset row.
        hf_dataset_params: dict
            Additional parameters to pass into HuggingFace `load_dataset` function
        split : str
            The dataset split to use (default is "validation").
        save_evaluation_data : bool
            Whether to save evaluation data to files (default is True).
            You should set this to False if you already have a directory of evaluation files to pass into AutoGluon RAG.
        evaluation_dir : str
            The directory for evaluation data (default is "./evaluation_data").
        save_csv_path : str
            The path to save the evaluation results as a CSV file (default is None).
        max_eval_size : int, optional
            The maximum number of datapoints to process for evaluation (default is None).
            This value should be less than the total number of datapoints.
        """
        self.agrag = agrag
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.dataset = load_dataset(dataset_name, split=split, **hf_dataset_params)
        self.metric_instances = self.initialize_metrics(metrics)
        self.save_evaluation_data = save_evaluation_data
        self.evaluation_dir = evaluation_dir
        self.save_csv_path = save_csv_path
        self.max_eval_size = max_eval_size
        self.preprocessing_fn = preprocessing_fn
        self.query_fn = query_fn
        self.response_fn = response_fn

    def initialize_metrics(self, metrics: List[str]) -> Dict[str, Any]:
        """
        Initializes the evaluation metrics.

        Parameters:
        ----------
        metrics : List[Union[str, Callable]]
            The list of metrics to initialize.

        Returns:
        -------
        Dict[str, Any]
            A dictionary of initialized metric instances.
        """
        metric_instances = {}
        for metric in metrics:
            if isinstance(metric, str):
                if metric == "bertscore":
                    metric_instances[metric] = evaluate.load("bertscore")
                elif metric == "exact_match":
                    metric_instances[metric] = evaluate.load("exact_match")
                elif metric == "pedant":
                    metric_instances[metric] = PEDANT()
                elif metric == "transformer_matcher":
                    metric_instances[metric] = TransformerMatcher("roberta-large")
                else:
                    logger.warning(f"Unsupported metric {metric}. Evaluation will not be performed on this metric.")
            elif callable(metric):
                metric_name = metric.__name__
                metric_instances[metric_name] = metric
        return metric_instances

    def save_documents_to_files(
        self,
        evaluation_dir: str = EVALUATION_DIR,
        max_file_size: int = EVALUATION_MAX_FILE_SIZE,
        preprocessing_fn: Callable = None,
    ):
        """
        Saves the documents from the dataset to text files in the specified directory.

        Parameters:
        ----------
        evaluation_dir : str
            The directory where the documents will be saved.
        max_file_size : int
            The maximum size of each file in bytes (default is 5 MB).
        preprocessing_fn : callable, optional
            A function to preprocess the content before saving.
        """
        if not evaluation_dir:
            raise ValueError("Must provide evaluation directory")

        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)

        file_index = 0
        current_file_size = 0
        current_file = open(os.path.join(evaluation_dir, f"doc_{file_index}.txt"), "w", encoding="utf-8")

        for idx, row in enumerate(self.dataset):
            if self.max_eval_size and idx >= self.max_eval_size:
                break
            content = preprocessing_fn(row) if preprocessing_fn else row["document"]["content"]
            content_size = len(content.encode("utf-8"))

            if current_file_size + content_size > max_file_size:
                current_file.close()
                file_index += 1
                current_file_size = 0
                current_file = open(os.path.join(evaluation_dir, f"doc_{file_index}.txt"), "w", encoding="utf-8")

            current_file.write(content + "\n")
            current_file_size += content_size

        current_file.close()

    def get_queries_and_responses(self, query_fn: Callable, response_fn: Callable):
        """
        Obtains the queries and responses from the dataset.

        Parameters:
        ----------
        query_fn : Callable
            A function to extract the query from the dataset row.
        response_fn : Callable
            A function to extract the expected responses from the dataset row.

        Returns:
        -------
        Tuple[List[str], List[List[str]], List[str]]
            Queries, references, and predictions.
        """
        references = []
        queries = []
        predictions = []

        for idx, row in enumerate(self.dataset):
            if self.max_eval_size and idx >= self.max_eval_size:
                break

            query = query_fn(row)
            expected_responses = response_fn(row)
            if not expected_responses:
                continue

            generated_response = self.agrag.generate_response(query)
            references.append(expected_responses)
            predictions.append(generated_response)
            queries.append(query)

            logger.info(f"Query: {query}")
            logger.info(f"Expected: {expected_responses}")
            logger.info(f"Generated: {generated_response}")

        return queries, references, predictions

    def evaluate_responses(
        self,
        predictions: List[str],
        references: List[List[str]],
        queries: List[str],
    ):
        """
        Evaluates the responses generated by the RAG pipeline.

        Parameters:
        ----------
        predictions : List[str]
            The generated responses.
        references : List[List[str]]
            The expected responses.
        queries: List[str]
            The original queries for each response.
        """

        for metric in self.metrics:
            if isinstance(metric, str):
                metric_instance = self.metric_instances[metric]
                if metric == "bertscore":
                    result = metric_instance.compute(predictions=predictions, references=references, lang="en")
                    logger.info(f"Average BERTScore Precision: {np.mean(result['precision'])}")
                    logger.info(f"Average BERTScore Recall: {np.mean(result['recall'])}")
                    logger.info(f"Average BERTScore F1 Score: {np.mean(result['f1'])}")

                elif metric == "exact_match":
                    exact_matches = custom_exact_match_metric(predictions=predictions, references=references)
                    result = calculate_exact_match_score(exact_matches=exact_matches)
                    logger.info(f"Exact Match Score: {result}")

                elif metric in ["pedant", "transformer_matcher"]:
                    rag_score = qa_metric_score(
                        predictions=predictions, references=references, queries=queries, qa_metric=metric_instance
                    )
                    logger.info(f"RAG Score QA Metric ({metric}): {rag_score}")
            elif callable(metric):
                metric_name = metric.__name__
                metric_instance = self.metric_instances[metric_name]
                result = metric_instance(predictions, references)
                logger.info(f"Custom Metric ({metric_name}) Score: {result}")

    def save_evaluation_results(
        self, output_csv: str, predictions: List[str], references: List[List[str]], queries: List[str]
    ):
        """
        Saves the evaluation results to a CSV file.

        Parameters:
        ----------
        output_csv : str
            The path to the output CSV file.
        predictions : List[str]
            The generated responses.
        references : List[List[str]]
            The expected responses.
        queries: List[str]
            The original queries for each response.
        """
        save_responses_to_csv(predictions, references, queries, output_csv)

    def run_evaluation(self):
        if self.save_evaluation_data:
            self.save_documents_to_files(evaluation_dir=self.evaluation_dir, preprocessing_fn=self.preprocessing_fn)

        self.agrag.initialize_rag_pipeline()

        queries, expected_repsonses, generated_responses = self.get_queries_and_responses(
            query_fn=self.query_fn, response_fn=self.response_fn
        )
        self.evaluate_responses(predictions=generated_responses, references=expected_repsonses, queries=queries)
        if self.save_csv_path:
            self.save_evaluation_results(output_csv=self.save_csv_path, references=expected_repsonses, queries=queries)
