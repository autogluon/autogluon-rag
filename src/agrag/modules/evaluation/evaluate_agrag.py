import csv
import logging
import os
import re
import string
from typing import List

import evaluate
import numpy as np
from bs4 import BeautifulSoup
from datasets import load_dataset
from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher

from agrag.agrag import AutoGluonRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Google Natural Questions dataset
dataset = load_dataset("google-research-datasets/natural_questions", "dev")
MAX_FILE_SIZE = 50 * 1000 * 1000  # 50 MB
bertscore = evaluate.load("bertscore")
exact_match_metric = evaluate.load("exact_match")
pedant = PEDANT()
tm = TransformerMatcher("roberta-large")

# Create evaluation directory
evaluation_dir = "temp_evaluation_data"
if not os.path.exists(evaluation_dir):
    os.makedirs(evaluation_dir)

MAX_SIZE_EVAL = 50


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


def custom_exact_match_metric(predictions: List[str], references: List[List[str]]) -> float:
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."

    exact_matches = []

    for gen_resp, exp_resps in zip(predictions, references):
        gen_resp = preprocess_text(gen_resp)
        match_found = False

        for exp_resp in exp_resps:
            exp_resp = preprocess_text(exp_resp)

            if gen_resp == exp_resp or exp_resp in gen_resp:
                match_found = True
                break
        exact_matches.append(match_found)
    return exact_matches


def save_responses_to_csv(
    generated_responses: List[str], expected_responses: List[List[str]], exact_matches: List[bool], output_csv: str
):
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generated Response", "Expected Responses", "Exact Match"])

        for gen_resp, exp_resps, match in zip(generated_responses, expected_responses, exact_matches):
            writer.writerow([gen_resp, "; ".join(exp_resps), match])


def calculate_exact_match_score(exact_matches: List[bool]) -> float:
    total_responses = len(exact_matches)
    total_matches = sum(exact_matches)
    exact_match_score = total_matches / total_responses if total_responses > 0 else 0
    return exact_match_score


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text


def save_documents_to_files(dataset, evaluation_dir, max_file_size=MAX_FILE_SIZE):
    file_index = 0
    current_file_size = 0
    current_file = open(os.path.join(evaluation_dir, f"doc_{file_index}.txt"), "w", encoding="utf-8")

    max = 0
    for row in dataset["validation"]:
        html_content = row["document"]["html"]
        content = extract_text_from_html(html_content)
        content_size = len(content.encode("utf-8"))

        if current_file_size + content_size > max_file_size:
            current_file.close()
            file_index += 1
            current_file_size = 0
            current_file = open(os.path.join(evaluation_dir, f"doc_{file_index}.txt"), "w", encoding="utf-8")

        current_file.write(content + "\n")
        current_file_size += content_size
        if max == MAX_SIZE_EVAL:
            break
        max += 1

    current_file.close()


def evaluate_responses(dataset, agrag: AutoGluonRAG):
    references = []
    predictions = []
    predictions_norag = []

    scores = []
    scores_norag = []

    max = 0
    for row in dataset["validation"]:
        query = row["question"]["text"]
        short_answer = row["annotations"]["short_answers"]
        print(short_answer)
        expected_responses = []
        for i in range(len(short_answer)):
            text = short_answer[i]["text"]
            if text:
                expected_responses += [text[0]]
        print(expected_responses)

        if not expected_responses:
            continue

        generated_response = agrag.generate_response(query)
        generate_response_no_rag = agrag.generate_response_no_rag(query)

        # max_pair, highest_score = tm.get_highest_score(generated_response, expected_responses, query)
        # scores.append(highest_score)
        # max_pair, highest_score = tm.get_highest_score(generate_response_no_rag, expected_responses, query)
        # scores_norag.append(highest_score)

        references.append(expected_responses)
        predictions.append(generated_response)
        predictions_norag.append(generate_response_no_rag)

        logger.info(f"Query: {query}")
        logger.info(f"Expected: {expected_responses}")
        logger.info(f"Generated: {generated_response}")
        if max == MAX_SIZE_EVAL:
            break
        max += 1

    # print("Predictions: ", predictions)
    # print("References: ", references)
    # result = bertscore.compute(predictions=predictions, references=references, lang="en")
    # result = exact_match_metric.compute(predictions=predictions, references=references, ignore_case=True, ignore_punctuation=True)
    exact_matches = custom_exact_match_metric(predictions=predictions, references=references)
    # save_responses_to_csv(
    #     generated_responses=predictions,
    #     expected_responses=references,
    #     exact_matches=exact_matches,
    #     output_csv="evaluation_results.csv",
    # )
    result = calculate_exact_match_score(exact_matches=exact_matches)
    logger.info(f"Exact Match Score: {result}")
    exact_matches = custom_exact_match_metric(predictions=predictions_norag, references=references)
    result = calculate_exact_match_score(exact_matches=exact_matches)
    logger.info(f"Exact Match Score No RAG: {result}")
    # logger.info(f"Average Score: {np.mean(scores)}")
    # logger.info(f"Average Score No-RAG: {np.mean(scores_norag)}")
    # logger.info(f"Average BERTScore Precision: {np.mean(result['precision'])}")
    # logger.info(f"Average BERTScore Recall: {np.mean(result['recall'])}")
    # logger.info(f"Average BERTScore F1 Score: {np.mean(result['f1'])}")

    # result = bertscore.compute(predictions=predictions_norag, references=references, lang="en")
    # logger.info(f"NO RAG: Average BERTScore Precision: {np.mean(result['precision'])}")
    # logger.info(f"NO RAG: Average BERTScore Recall: {np.mean(result['recall'])}")
    # logger.info(f"NO RAG: Average BERTScore F1 Score: {np.mean(result['f1'])}")


if __name__ == "__main__":
    # save_documents_to_files(dataset, evaluation_dir)
    agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
    agrag.args.generator_query_prefix = (
        "You will be answering questions based on Wikipedia articles. "
        "Your response must contain only the actual answer itself and no extra information or words. Provide just the answer instead of forming full sentences. "
        "Your answer may even be a singular word or number. For example, if the query is \"what percentage of the earth's surface is water?\", your response must be 'roughly 78%', or '78%'. "
    )
    agrag.initialize_rag_pipeline()
    evaluate_responses(dataset, agrag)
