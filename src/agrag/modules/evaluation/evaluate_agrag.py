import logging
import os

import evaluate
import numpy as np
from bs4 import BeautifulSoup
from datasets import load_dataset

from agrag.agrag import AutoGluonRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Google Natural Questions dataset
dataset = load_dataset("google-research-datasets/natural_questions", "dev")
MAX_FILE_SIZE = 50 * 1000 * 1000  # 50 MB
bertscore = evaluate.load("bertscore")

# Create evaluation directory
evaluation_dir = "evaluation_data"
if not os.path.exists(evaluation_dir):
    os.makedirs(evaluation_dir)


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
        if max == 5:
            break
        max += 1

    current_file.close()


def evaluate_responses(dataset, agrag):
    references = []
    predictions = []

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

        references.append(expected_responses)
        predictions.append(generated_response)

        logger.info(f"Query: {query}")
        logger.info(f"Expected: {expected_responses}")
        logger.info(f"Generated: {generated_response}")

    print("Predictions: ", predictions)
    print("References: ", references)
    result = bertscore.compute(predictions=predictions, references=references, lang="en")
    logger.info(f"Average BERTScore Precision: {np.mean(result['precision'])}")
    logger.info(f"Average BERTScore Recall: {np.mean(result['recall'])}")
    logger.info(f"Average BERTScore F1 Score: {np.mean(result['f1'])}")


if __name__ == "__main__":
    save_documents_to_files(dataset, evaluation_dir)
    agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
    agrag.args.generator_query_prefix = (
        "You will be answering questions based on Wikipedia articles. "
        "Your response must contain only the actual answer itself and no extra information or words. Provide just the answer instead of forming full sentences. "
        "Your answer may even be a singular word or number. For example, if the query is \"what percentage of the earth's surface is water?\", your response must be 'roughly 78%', or '78%'. "
    )
    agrag.initialize_rag_pipeline()
    evaluate_responses(dataset, agrag)
