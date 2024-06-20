import logging
import os

import pandas as pd
import torch
from tqdm import tqdm

from agrag.args import Arguments
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.generator.utils import format_query
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.utils import load_index, load_metadata, save_index, save_metadata
from agrag.modules.vector_db.vector_database import VectorDatabaseModule
from agrag.utils import parse_path, read_openai_key

logger = logging.getLogger("rag-logger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def initialize_rag_pipeline(args: Arguments) -> RetrieverModule:

    db_type = args.vector_db_type

    index_path = args.vector_db_index_path
    vector_db_s3_bucket, vector_db_index_path = parse_path(index_path)

    metadata_path = args.metadata_index_path
    _, metadata_index_path = parse_path(metadata_path)

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

    num_gpus = args.vector_db_num_gpus

    vector_database_module = VectorDatabaseModule(
        db_type=db_type,
        params=args.vector_db_args,
        similarity_threshold=args.vector_db_sim_threshold,
        similarity_fn=args.vector_db_sim_fn,
        s3_bucket=vector_db_s3_bucket,
        num_gpus=num_gpus,
    )

    logger.info(f"Using Vector DB: {db_type}")

    load_index_successful = False

    if args.use_existing_vector_db_index:
        logger.info(f"Loading existing index from {index_path}")
        vector_database_module.index = load_index(
            db_type,
            vector_db_index_path,
            vector_database_module.s3_bucket,
            vector_database_module.s3_client,
        )
        logger.info(f"Loading existing metadata from {metadata_path}")
        vector_database_module.metadata = load_metadata(
            metadata_index_path,
            vector_database_module.s3_bucket,
            vector_database_module.s3_client,
        )
        load_index_successful = (
            True if vector_database_module.index and vector_database_module.metadata is not None else False
        )

    if not load_index_successful:
        data_dir = args.data_dir
        if not data_dir:
            raise ValueError("Error: 'data_dir' must be specified in the configuration file under 'data' section.")

        logger.info(f"Retrieving Data from {data_dir}")
        data_s3_bucket, data_dir = parse_path(data_dir)

        data_processing_module = DataProcessingModule(
            data_dir=data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            s3_bucket=data_s3_bucket,
        )

        with tqdm(total=100, desc="Data Preprocessing Module", unit="chunk") as pbar:
            processed_data = data_processing_module.process_data()
            pbar.n = 100
            pbar.refresh()

        with tqdm(total=len(processed_data), desc="\nEmbedding Module", unit="step") as pbar:

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
            embeddings = embedding_module.encode(processed_data, pbar, batch_size=args.embedding_batch_size)

        logger.info(f"\nConstructing new index and saving at {vector_db_index_path}")
        with tqdm(total=3, desc="Vector DB Module", unit="step") as pbar:
            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
                logger.info(f"Using max number of GPUs for Vector DB: {num_gpus}")
            else:
                logger.info(f"Using number of GPUs: {num_gpus} for Vector DB")
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
            save_metadata(
                vector_database_module.metadata,
                metadata_index_path,
                vector_database_module.s3_bucket,
                vector_database_module.s3_client,
            )

    num_gpus = args.retriever_num_gpus
    reranker = None
    if args.use_reranker:
        logger.info(f"\nUsing reranker {args.reranker_model_name}")
        reranker = Reranker(
            model_name=args.reranker_model_name,
            batch_size=args.reranker_batch_size,
            hf_forward_params=args.reranker_hf_forward_params,
            hf_tokenizer_init_params=args.reranker_hf_tokenizer_init_params,
            hf_tokenizer_params=args.reranker_hf_tokenizer_params,
            hf_model_params=args.reranker_hf_model_params,
            num_gpus=num_gpus,
        )

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using max number of GPUs for Retrieval: {num_gpus}")
    else:
        logger.info(f"Using number of GPUs: {num_gpus} for Retrieval")

    logger.info(f"\nInitializing Retrieval Module")
    retriever_module = RetrieverModule(
        vector_database_module=vector_database_module,
        embedding_module=embedding_module,
        top_k=args.top_k_embeddings,
        reranker=reranker,
        num_gpus=num_gpus,
    )

    return retriever_module


def ag_rag():
    logger.info("\n\nAutoGluon-RAG\n\n")
    args = Arguments()
    logger.info("Initializing RAG Pipeline")
    retriever_module = initialize_rag_pipeline(args)
    openai_api_key = read_openai_key(args.openai_key_file)
    generator_module = GeneratorModule(
        model_name=args.generator_model_name,
        hf_model_params=args.generator_hf_model_params,
        hf_tokenizer_init_params=args.generator_hf_tokenizer_init_params,
        hf_tokenizer_params=args.generator_hf_tokenizer_params,
        hf_generate_params=args.generator_hf_generate_params,
        gpt_generate_params=args.gpt_generate_params,
        vllm_sampling_params=args.vllm_sampling_params,
        num_gpus=args.generator_num_gpus,
        use_vllm=args.use_vllm,
        openai_api_key=openai_api_key,
        bedrock_generate_params=args.bedrock_generate_params,
        use_bedrock=args.use_bedrock,
    )

    query_prefix = args.generator_query_prefix

    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            # correctly shutdown modules (VectorDB connection; for example)
            break

        retrieved_context = retriever_module.retrieve(query_text)

        if query_prefix:
            query_text = f"{query_prefix}\n{query_text}"

        formatted_query = format_query(query_text, retrieved_context)

        response = generator_module.generate_response(formatted_query)

        logger.info(f"\nResponse: {response}\n")


if __name__ == "__main__":
    ag_rag()
