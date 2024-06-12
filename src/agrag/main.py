import logging
import os

import torch
from tqdm import tqdm

from agrag.args import Arguments
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.retriever.retriever import RetrieverModule
from agrag.modules.vector_db.utils import load_index, save_index
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger("rag-logger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def initialize_rag_pipeline() -> RetrieverModule:
    args = Arguments()

    data_dir = args.data_dir
    if not data_dir:
        raise ValueError("Error: 'data_dir' must be specified in the configuration file under 'data' section.")

    logger.info(f"Retrieving Data from {data_dir}")
    data_processing_module = DataProcessingModule(
        data_dir=data_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, s3_bucket=args.data_s3_bucket
    )

    with tqdm(total=100, desc="Data Preprocessing", unit="chunk") as pbar:
        processed_data = data_processing_module.process_data()
        pbar.n = 100
        pbar.refresh()

    total_steps = len(processed_data)
    with tqdm(total=total_steps, desc="Embedding Generation", unit="step") as pbar:

        embedding_module = EmbeddingModule(
            hf_model=args.hf_embedding_model,
            pooling_strategy=args.pooling_strategy,
            normalize_embeddings=args.normalize_embeddings,
            hf_model_params=args.hf_model_params,
            hf_tokenizer_init_params=args.hf_tokenizer_init_params,
            hf_tokenizer_params=args.hf_tokenizer_params,
            hf_forward_params=args.hf_forward_params,
            normalization_params=args.normalization_params,
            query_instruction_for_retrieval=args.query_instruction_for_retrieval,
        )
        embeddings = embedding_module.encode(processed_data, pbar)

    db_type = args.vector_db_type

    num_gpus = args.vector_db_num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using max number of GPUs: {num_gpus}")
    else:
        logger.info(f"Using number of GPUs: {num_gpus}")

    vector_db_index_path = os.path.join(args.vector_db_index_path, db_type, "index.idx")
    vector_database_module = VectorDatabaseModule(
        db_type=db_type,
        params=args.vector_db_args,
        similarity_threshold=args.vector_db_sim_threshold,
        similarity_fn=args.vector_db_sim_fn,
        s3_bucket=args.vector_db_s3_bucket,
        num_gpus=num_gpus,
    )

    logger.info(f"Using Vector DB: {db_type}")

    load_index_successful = False

    if args.use_existing_vector_db_index:
        logger.info(f"Loading existing index from {vector_db_index_path}")
        with tqdm(total=total_steps, desc="Existing Vector DB Loading", unit="step") as pbar:
            vector_database_module.index = load_index(
                db_type,
                vector_db_index_path,
                vector_database_module.s3_bucket,
                vector_database_module.s3_client,
                pbar,
            )
        load_index_successful = True if vector_database_module.index else False

    if not load_index_successful:
        logger.info(f"Constructing new index and saving at {vector_db_index_path}")
        with tqdm(total=total_steps, desc="Vector DB Construction", unit="step") as pbar:
            vector_database_module.construct_vector_database(embeddings, pbar)
            basedir = os.path.dirname(vector_db_index_path)
            if not os.path.exists(basedir):
                logger.info(f"Creating directory for Vector Index save at {basedir}")
                os.makedirs(basedir)
            save_index(
                db_type,
                vector_database_module.index,
                vector_db_index_path,
                vector_database_module.s3_bucket,
                vector_database_module.s3_client,
            )

    retriever_module = RetrieverModule(vector_database_module.index)

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
