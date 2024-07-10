import concurrent.futures
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from agrag.args import Arguments
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.data_processing.utils import download_directory_from_s3, get_all_file_paths
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.generator.utils import format_query
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.utils import load_index, load_metadata, save_index, save_metadata
from agrag.modules.vector_db.vector_database import VectorDatabaseModule
from agrag.utils import get_num_gpus, read_openai_key

logger = logging.getLogger("rag-logger")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

PRESETS_CONFIG_DIRECTORY = os.path.join(os.path.dirname(__file__), "configs/presets")


class AutoGluonRAG:
    def __init__(
        self,
        config_file: Optional[str] = None,
        preset_quality: Optional[str] = None,
        model_ids: Dict = None,
        data_dir: str = "",
    ):
        """
        Initializes the AutoGluonRAG class with either a configuration file or a preset quality setting.

        Parameters:
        ----------
        config_file : str, optional
            Path to the configuration file.
        preset_quality : str, optional
            Preset quality setting (e.g., "good", "medium", "best").
        model_ids : dict, optional
            Dictionary of model IDs to use for specific modules.
            Example: {"generator_model_id": "mistral.mistral-7b-instruct-v0:2", "retriever_model_id": "BAAI/bge-large-en", "reranker_model_id": "nv_embed"}
        data_dir : str
            The directory containing the data files that will be used for the RAG pipeline
        """
        logger.info("\n\nAutoGluon-RAG\n\n")

        self.preset_quality = preset_quality
        self.model_ids = model_ids
        self.batch_size = 2

        if config_file:
            self._load_config()
        elif self.preset_quality:
            self._load_preset()

        self.config = config_file or self._load_preset()

        self.args = Arguments(self.config)

        # will short-circuit to provided data_dir if config data_dir also provided
        self.data_dir = data_dir or self.args.data_dir

        if not self.data_dir:
            raise ValueError("data_dir argument must be provided")

        self.data_processing_module = None
        self.embedding_module = None
        self.vector_db_module = None
        self.reranker_module = None
        self.retriever_module = None
        self.generator_module = None

    def _load_config(self, config_file: str):
        """Load configuration data from a user-defined config file."""
        try:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Error: File not found - {config_file}")
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file - {config_file}: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred while loading {config_file}: {exc}")

    def _load_preset(self):
        """Loads a preset configuration based on the preset quality setting."""
        presets = {"medium_quality": os.path.join(PRESETS_CONFIG_DIRECTORY, "medium_quality_config.yaml")}
        logger.info(f"Loading Preset '{self.preset_quality}' configuration")
        return presets[self.preset_quality]

    def initialize_data_module(self):
        """Initializes the Data Processing module."""
        self.data_processing_module = DataProcessingModule(
            data_dir=self.data_dir,
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            file_exts=self.args.data_file_extns,
        )
        logger.info("Data Processing module initialized")

    def initialize_embeddings_module(self):
        """Initializes the Embedding module."""
        self.embedding_module = EmbeddingModule(
            hf_model=self.args.hf_embedding_model,
            pooling_strategy=self.args.pooling_strategy,
            normalize_embeddings=self.args.normalize_embeddings,
            hf_model_params=self.args.hf_model_params,
            hf_tokenizer_init_params=self.args.hf_tokenizer_init_params,
            hf_tokenizer_params=self.args.hf_tokenizer_params,
            hf_forward_params=self.args.hf_forward_params,
            normalization_params=self.args.normalization_params,
            query_instruction_for_retrieval=self.args.query_instruction_for_retrieval,
        )
        logger.info("Embedding module initialized")

    def initialize_vectordb_module(self):
        """Initializes the Vector DB module."""
        db_type = self.args.vector_db_type
        logger.info(f"Using Vector DB: {db_type}")
        num_gpus = get_num_gpus(self.args.vector_db_num_gpus)
        logger.info(f"Using number of GPUs: {num_gpus} for Vector DB Module")
        self.vector_db_module = VectorDatabaseModule(
            db_type=db_type,
            params=self.args.vector_db_args,
            similarity_threshold=self.args.vector_db_sim_threshold,
            similarity_fn=self.args.vector_db_sim_fn,
            num_gpus=num_gpus,
        )
        logger.info("Vector DB module initialized")

    def initialize_retriever_module(self):
        """Initializes the Retriever module."""
        num_gpus = get_num_gpus(self.args.retriever_num_gpus)
        logger.info(f"Using number of GPUs: {num_gpus} for Retriever Module")
        self.retriever_module = RetrieverModule(
            vector_database_module=self.vector_db_module,
            embedding_module=self.embedding_module,
            top_k=self.args.retriever_top_k,
            reranker=self.reranker_module,
            num_gpus=num_gpus,
        )
        logger.info("Retriever module initialized")

    def initialize_generator_module(self):
        """Initializes the Generator module."""
        openai_api_key = read_openai_key(self.args.openai_key_file)
        num_gpus = get_num_gpus(self.args.generator_num_gpus)
        logger.info(f"Using number of GPUs: {num_gpus} for Generator Module")

        self.generator_module = GeneratorModule(
            model_name=self.args.generator_model_name,
            hf_model_params=self.args.generator_hf_model_params,
            hf_tokenizer_init_params=self.args.generator_hf_tokenizer_init_params,
            hf_tokenizer_params=self.args.generator_hf_tokenizer_params,
            hf_generate_params=self.args.generator_hf_generate_params,
            gpt_generate_params=self.args.gpt_generate_params,
            vllm_sampling_params=self.args.vllm_sampling_params,
            num_gpus=num_gpus,
            use_vllm=self.args.use_vllm,
            openai_api_key=openai_api_key,
            bedrock_generate_params=self.args.bedrock_generate_params,
            use_bedrock=self.args.use_bedrock,
            local_model_path=self.args.generator_local_model_path,
        )
        logger.info("Generator module initialized")

    def initialize_reranker_module(self):
        """Initializes the Reranker module."""
        reranker_model = self.args.reranker_model_name
        logger.info(f"\nUsing reranker {reranker_model}")

        num_gpus = get_num_gpus(self.args.retriever_num_gpus)
        logger.info(f"Using number of GPUs: {num_gpus} for Reranker Module")

        self.reranker_module = Reranker(
            model_name=reranker_model,
            batch_size=self.args.reranker_batch_size,
            top_k=self.args.reranker_top_k,
            hf_forward_params=self.args.reranker_hf_forward_params,
            hf_tokenizer_init_params=self.args.reranker_hf_tokenizer_init_params,
            hf_tokenizer_params=self.args.reranker_hf_tokenizer_params,
            hf_model_params=self.args.reranker_hf_model_params,
            num_gpus=num_gpus,
        )
        logger.info("Reranker module initialized")

    def process_data(self) -> pd.DataFrame:
        """
        Processes the data in the provided data directory using the initialized Data Processing module.

        This method extracts and chunks text from all files in the specified data directory,
        and compiles the results into a single DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing processed text chunks from all files in the directory.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_data_module()
        processed_data = agrag.process_data()
        """
        logger.info(f"Retrieving Data from {self.data_processing_module.data_dir}")
        processed_data = self.data_processing_module.process_data()
        return processed_data

    def generate_embeddings(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates embeddings from the processed data using the initialized Embedding module.

        Parameters:
        ----------
        processed_data : pd.DataFrame
            A DataFrame containing the processed text chunks for which embeddings are to be generated.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the original data with an additional column for the generated embeddings.

        Example:
        --------
        processed_data = pd.DataFrame({
            "doc_id": [1, 2],
            "chunk_id": [1, 1],
            "text": ["This is a test sentence.", "This is another test sentence."]
        })
        embeddings = agrag.generate_embeddings(processed_data)
        """
        embeddings = self.embedding_module.encode(processed_data, batch_size=self.args.embedding_batch_size)
        return embeddings

    def construct_vector_db(self, embeddings: pd.DataFrame):
        """
        Constructs the vector database using the provided embeddings.

        This method initializes the vector database with the given embeddings, storing
        the embeddings in the vector database and associating metadata.

        Parameters:
        ----------
        embeddings : pd.DataFrame
            A DataFrame containing the embeddings and associated metadata.

        Example:
        --------
        embeddings = agrag.generate_embeddings(processed_data)
        agrag.construct_vector_db(embeddings)
        """
        logger.info(f"\nConstructing Vector DB index")
        self.vector_db_module.construct_vector_database(embeddings)

    def load_existing_vector_db(self, index_path: str, metadata_path: str):
        """
        Loads an existing Vector Database from the specified paths in the configuration.

        Parameters:
        index_path : str
            The path from where the index will be loaded
        metadata_path : str
            The path to the metadata file.

        Returns:
        -------
        bool
            True if the index and metadata were successfully loaded, False otherwise.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_vectordb_module()
        success = agrag.load_existing_vector_db("path/to/index", "path/to/metadata")
        """
        logger.info(f"Loading existing index from {index_path}")
        self.vector_db_module.index = load_index(self.args.vector_db_type, index_path)

        logger.info(f"Loading existing metadata from {metadata_path}")
        self.vector_db_module.metadata = load_metadata(metadata_path)

        load_index_successful = (
            True if self.vector_db_module.index and self.vector_db_module.metadata is not None else False
        )
        return load_index_successful

    def save_index_and_metadata(self, index_path, metadata_path):
        """
        Saves the vector database index and metadata to the specified paths in the configuration.

        This method ensures the directories for saving the index and metadata exist, then saves the
        vector database index and metadata to their respective paths.

        Parameters:
        index_path : str
            The path where the index will be saved.
        metadata_path : str
            The path where the metadata will be saved.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_vectordb_module()
        agrag.save_index_and_metadata()
        """
        logger.info(f"\nSaving Vector DB at {index_path}")
        save_index(
            self.vector_db_module.db_type,
            self.vector_db_module.index,
            index_path,
        )
        logger.info(f"\nSaving Metadata at {metadata_path}")
        save_metadata(
            self.vector_db_module.metadata,
            metadata_path,
        )

    def retrieve_context_for_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves relevant context for the provided query using the Retriever module.

        This method searches the vector database for the top-k most similar embeddings to the given query
        and returns the associated context.

        Parameters:
        ----------
        query : str
            The user query for which relevant context is to be retrieved.

        Returns:
        -------
        List[Dict[str, Any]]
            A list of relevant context chunks for the query.

        Example:
        --------
        context = agrag.retrieve_context_for_query("How do I use this package?")
        """
        return self.retriever_module.retrieve(query)

    def generate_response(self, query: str) -> str:
        """
        Generates a response to the provided query using the Generator module.

        This method first retrieves relevant context for the query using the Retriever module,
        formats the query and context appropriately, and then generates a response using the Generator module.

        Parameters:
        ----------
        query : str
            The user query for which a response is to be generated.

        Returns:
        -------
        str
            The generated response.

        Example:
        --------
        response = agrag.generate_response("What is AutoGluon?")
        """

        retrieved_context = self.retrieve_context_for_query(query)

        query_prefix = self.args.generator_query_prefix
        if query_prefix:
            query = f"{query_prefix}\n{query}"
        formatted_query = format_query(
            model_name=self.generator_module.model_name, query=query, context=retrieved_context
        )

        response = self.generator_module.generate_response(formatted_query)

        logger.info(f"\nResponse: {response}\n")

    def process_and_store_in_batches(self):
        """
        Processes data, generates embeddings, and stores them in the vector database in batches.

        This method handles the entire process for each batch sequentially: processing documents,
        generating embeddings, and storing them in the vector database before moving on to the next batch.
        """
        if self.data_processing_module.s3_bucket:
            self.data_processing_module.data_dir = download_directory_from_s3(
                s3_bucket=self.data_processing_module.s3_bucket,
                data_dir=self.data_processing_module.data_dir,
                s3_client=self.data_processing_module.s3_client,
            )

        file_paths = get_all_file_paths(self.data_processing_module.data_dir, self.data_processing_module.file_exts)

        for i in range(0, len(file_paths), self.batch_size):
            print(f"Batch {i}")
            batch_file_paths = file_paths[i : i + self.batch_size]

            processed_data = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(
                    self.data_processing_module.process_file, batch_file_paths, range(len(batch_file_paths))
                )
                for result in results:
                    processed_data.append(result)

            processed_data = pd.concat(processed_data).reset_index(drop=True)
            embeddings = self.generate_embeddings(processed_data)

            print(embeddings.shape)
            print(embeddings[0].shape)

            # Store the embeddings in the vector database
            if i == 0:
                self.construct_vector_db(embeddings)
            else:
                self.vector_db_module.index.add(np.array(embeddings))

            # Clear memory
            del processed_data
            del embeddings

    def initialize_rag_pipeline(self):
        """
        Initializes the entire RAG pipeline by setting up all necessary modules.

        This method initializes the Data Processing, Embedding, Vector DB, Reranker, Retriever,
        and Generator modules based on the provided configuration. It then processes the data,
        generates embeddings, constructs the vector database, and optionally saves the index and metadata.

        If the configuration specifies to use an existing vector DB index, it loads the existing index
        and metadata instead of creating a new one.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_rag_pipeline()
        """
        self.initialize_data_module()
        self.initialize_embeddings_module()
        self.initialize_vectordb_module()
        self.initialize_reranker_module()
        self.initialize_retriever_module()
        self.initialize_generator_module()
        load_index = self.args.use_existing_vector_db_index
        load_index_successful = False
        if load_index:
            self.load_existing_vector_db(self.args.vector_db_index_load_path, self.args.metadata_index_load_path)
            load_index_successful = (
                True if self.vector_db_module.index and self.vector_db_module.metadata is not None else False
            )

        if not load_index or not load_index_successful:
            # processed_data = self.process_data()
            # embeddings = self.generate_embeddings(processed_data=processed_data)
            # self.construct_vector_db(embeddings=embeddings)
            self.process_and_store_in_batches()
            if self.args.save_vector_db_index:
                self.save_index_and_metadata(self.args.vector_db_index_save_path, self.args.metadata_index_save_path)
