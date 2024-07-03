import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from agrag.args import Arguments  # Importing the Arguments class
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.generator.utils import format_query
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.utils import load_index, load_metadata, save_index, save_metadata
from agrag.modules.vector_db.vector_database import VectorDatabaseModule
from agrag.utils import read_openai_key

logger = logging.getLogger("rag-logger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

CONFIG_DIRECTORY = os.path.join(os.path.dirname(__file__), "configs")


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
        self.preset_quality = preset_quality
        self.model_ids = model_ids

        if self.config:
            self._load_config()
        elif self.preset_quality:
            self._load_preset()

        self.args = Arguments(config_file)

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

    def _load_preset(self):
        """Loads a preset configuration based on the preset quality setting."""
        presets = {"medium": os.path.join(CONFIG_DIRECTORY, "example_config.yaml")}
        self.args.config = self.args._load_config(presets[self.preset_quality])
        logger.info(f"Preset '{self.preset_quality}' configuration loaded")

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
        self.vector_db_module = VectorDatabaseModule(
            db_type=self.args.vector_db_type,
            params=self.args.vector_db_args,
            similarity_threshold=self.args.vector_db_sim_threshold,
            similarity_fn=self.args.vector_db_sim_fn,
            s3_bucket=self.args.vector_db_s3_bucket,
            num_gpus=self.args.vector_db_num_gpus,
        )
        logger.info("Vector DB module initialized")

    def initialize_retriever_module(self):
        """Initializes the Retriever module."""
        self.retriever_module = RetrieverModule(
            vector_database_module=self.vector_db_module,
            embedding_module=self.embedding_module,
            top_k=self.args.retriever_top_k,
            reranker=self.reranker_module,
            num_gpus=self.args.retriever_num_gpus,
        )
        logger.info("Retriever module initialized")

    def initialize_generator_module(self):
        """Initializes the Generator module."""
        openai_api_key = read_openai_key(self.args.openai_key_file)

        self.generator_module = GeneratorModule(
            model_name=self.args.generator_model_name,
            hf_model_params=self.args.generator_hf_model_params,
            hf_tokenizer_init_params=self.args.generator_hf_tokenizer_init_params,
            hf_tokenizer_params=self.args.generator_hf_tokenizer_params,
            hf_generate_params=self.args.generator_hf_generate_params,
            gpt_generate_params=self.args.gpt_generate_params,
            vllm_sampling_params=self.args.vllm_sampling_params,
            num_gpus=self.args.generator_num_gpus,
            use_vllm=self.args.use_vllm,
            openai_api_key=openai_api_key,
            bedrock_generate_params=self.args.bedrock_generate_params,
            use_bedrock=self.args.use_bedrock,
            local_model_path=self.args.generator_local_model_path,
        )
        logger.info("Generator module initialized")

    def initialize_reranker_module(self):
        """Initializes the Reranker module."""
        self.reranker_module = Reranker(
            model_name=self.args.reranker_model_name,
            batch_size=self.args.reranker_batch_size,
            top_k=self.args.reranker_top_k,
            hf_forward_params=self.args.reranker_hf_forward_params,
            hf_tokenizer_init_params=self.args.reranker_hf_tokenizer_init_params,
            hf_tokenizer_params=self.args.reranker_hf_tokenizer_params,
            hf_model_params=self.args.reranker_hf_model_params,
            num_gpus=self.args.reranker_num_gpus,
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
        self.vector_db_module.construct_vector_database(embeddings)

    def load_existing_vector_db(self):
        """
        Loads an existing Vector Database from the specified paths in the configuration.

        Returns:
        -------
        bool
            True if the index and metadata were successfully loaded, False otherwise.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_vectordb_module()
        success = agrag.load_existing_vector_db()
        """
        index_path = self.args.vector_db_index_load_path
        logger.info(f"Loading existing index from {index_path}")
        self.vector_db_module.index = load_index(self.args.vector_db_type, index_path)

        metadata_path = self.args.metadata_index_load_path
        logger.info(f"Loading existing metadata from {metadata_path}")
        self.vector_db_module.metadata = load_metadata(metadata_path)

        load_index_successful = (
            True if self.vector_db_module.index and self.vector_db_module.metadata is not None else False
        )
        return load_index_successful

    def save_index_and_metadata(self):
        """
        Saves the vector database index and metadata to the specified paths in the configuration.

        This method ensures the directories for saving the index and metadata exist, then saves the
        vector database index and metadata to their respective paths.

        Example:
        --------
        agrag = AutoGluonRAG(config_file="path/to/config")
        agrag.initialize_vectordb_module()
        agrag.save_index_and_metadata()
        """
        index_path = self.args.vector_db_index_save_path
        metadata_path = self.args.metadata_index_save_path
        basedir = os.path.dirname(index_path)
        if not os.path.exists(basedir):
            logger.info(f"Creating directory for Vector Index save at {basedir}")
            os.makedirs(basedir)
        save_index(
            self.vector_db_module.db_type,
            self.vector_db_module.index,
            index_path,
        )
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
        response = agrag.generate_response("How do I use this package?")
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
        return response

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
        if self.args.use_existing_vector_db_index:
            self.load_existing_vector_db()
        else:
            processed_data = self.process_data()
            embeddings = self.generate_embeddings(processed_data=processed_data)
            self.construct_vector_db(embeddings=embeddings)
            if self.args.save_index:
                self.save_index_and_metadata()
