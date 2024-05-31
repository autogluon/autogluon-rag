import argparse
import logging
import os

import yaml

from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.retriever.retriever import RetrieverModule
from agrag.modules.vector_db.vector_database import VectorDatabaseModule
from agrag.defaults import DATA_PROCESSING_MODULE_DEFAULTS

CURRENT_DIR = os.path.dirname(__file__)


logger = logging.getLogger("rag-logger")


def get_defaults_from_config():
    DATA_PROCESSING_MODULE_CONFIG = os.path.join(CURRENT_DIR, "configs/data_processing/default.yaml")
    global DATA_PROCESSING_MODULE_DEFAULTS
    with open(DATA_PROCESSING_MODULE_CONFIG, "r") as f:
        doc = yaml.safe_load(f)
        DATA_PROCESSING_MODULE_DEFAULTS = dict(
            (k, v if v else doc["data"][k]) for k, v in DATA_PROCESSING_MODULE_DEFAULTS.items()
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoGluon-RAG - Retrieval-Augmented Generation Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the directory containing the documents to be ingested into the RAG pipeline",
        required=True,
        metavar="",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Maximum chunk length to split the documents into",
        metavar="",
        required=False,
        default=DATA_PROCESSING_MODULE_DEFAULTS["CHUNK_SIZE"],
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        help="Amount of overlap between consecutive chunks. This is the number of characters that will be shared between adjacent chunks",
        metavar="",
        required=False,
        default=DATA_PROCESSING_MODULE_DEFAULTS["CHUNK_OVERLAP"],
    )

    args = parser.parse_args()
    return args


def initialize_rag_pipeline() -> RetrieverModule:
    get_defaults_from_config()

    args = get_args()
    data_dir = args.data_dir
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    logger.info(f"Processing Data from provided documents at {data_dir}")
    data_processing_module = DataProcessingModule(data_dir, chunk_size, chunk_overlap)
    processed_data = data_processing_module.process_data()

    embedding_module = EmbeddingModule()
    embeddings = embedding_module.create_embeddings(processed_data)

    vector_database_module = VectorDatabaseModule()
    vector_database = vector_database_module.construct_vector_database(embeddings)

    retriever_module = RetrieverModule(vector_database)

    return retriever_module


def ag_rag():
    print("\n\nAutoGluon-RAG\n\n")

    logger.info("Initializing RAG Pipeline")
    retriever_module = initialize_rag_pipeline()
    generator_module = GeneratorModule()

    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            # correctly shutdown modules (VectorDB connection; for example)
            break

        retrieved_data = retriever_module.retrieve(query_text)

        response = generator_module.generate_response(retrieved_data)

        print("Response:", response)


if __name__ == "__main__":
    ag_rag()
