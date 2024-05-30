import argparse

from agrag.modules.data_ingestion import DataIngestionModule
from agrag.modules.embedding import EmbeddingModule
from agrag.modules.generator import GeneratorModule
from agrag.modules.retriever import RetrieverModule
from agrag.modules.vector_database import VectorDatabaseModule


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
        default=512,
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        help="Amount of overlap between consecutive chunks. This is the number of characters that will be shared between adjacent chunks",
        metavar="",
        required=False,
        default=128,
    )

    args = parser.parse_args()
    return args


def initialize_rag_pipeline() -> RetrieverModule:
    args = get_args()
    data_dir = args.data_dir
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    data_ingestion_module = DataIngestionModule(data_dir, chunk_size, chunk_overlap)
    processed_data = data_ingestion_module.process_data()

    embedding_module = EmbeddingModule()
    embeddings = embedding_module.create_embeddings(processed_data)

    vector_database_module = VectorDatabaseModule()
    vector_database = vector_database_module.construct_vector_database(embeddings)

    retriever_module = RetrieverModule(vector_database)

    return retriever_module


def ag_rag():
    print("\n\nAutoGluon-RAG\n\n")

    retriever_module = initialize_rag_pipeline()
    generator_module = GeneratorModule()

    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'exit' to quit): "
        )
        if query_text.lower() == "exit":
            # correctly shutdown modules (VectorDB connection; for example)
            break

        retrieved_data = retriever_module.retrieve(query_text)

        response = generator_module.generate_response(retrieved_data)

        print("Response:", response)


if __name__ == "__main__":
    ag_rag()
