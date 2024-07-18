import os
import logging
from datasets import load_dataset
import evaluate
from agrag.agrag import AutoGluonRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Google Natural Questions dataset
dataset = load_dataset("google-research-datasets/natural_questions", "dev")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
bleu_metric = evaluate.load('exact_match')

# Create evaluation directory
evaluation_dir = "evaluation_data"
if not os.path.exists(evaluation_dir):
    os.makedirs(evaluation_dir)


def save_documents_to_files(dataset, evaluation_dir, max_file_size=MAX_FILE_SIZE):
    file_index = 0
    current_file_size = 0
    current_file = open(os.path.join(evaluation_dir, f"doc_{file_index}.txt"), "w", encoding="utf-8")

    max = 0
    for row in dataset["validation"]:
        content = row["document"]["html"]
        content_size = len(content.encode('utf-8'))

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
        expected_response = row["annotations"]["short_answers"]["text"]

        generated_response = agrag.generate_response(query)

        references.append([expected_response])
        predictions.append(generated_response)

        logger.info(f"Query: {query}")
        logger.info(f"Expected: {expected_response}")
        logger.info(f"Generated: {generated_response}")

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    logger.info(f"BLEU Score: {bleu_score}")

    return bleu_score

if __name__ == "__main__":
    save_documents_to_files(dataset, evaluation_dir)
    agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
    agrag.initialize_rag_pipeline()
    evaluate_responses(dataset, agrag)
