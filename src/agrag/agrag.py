import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.utils.html import extract_sub_links

from agrag.args import Arguments
from agrag.constants import LOGGER_NAME
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.data_processing.utils import get_all_file_paths
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.generator.utils import format_query
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.utils import load_index, load_metadata, save_index, save_metadata
from agrag.modules.vector_db.vector_database import VectorDatabaseModule
from agrag.utils import get_num_gpus

logger = logging.getLogger(LOGGER_NAME)
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
        preset_quality: Optional[str] = "medium_quality",
        model_ids: Dict = None,
        data_dir: str = "",
        web_urls: List = [],
        base_urls: List = [],
        login_info: dict = {},
        parse_urls_recursive: bool = True,
        pipeline_batch_size: int = 0,
    ):
        """
        Initializes the AutoGluonRAG class with either a configuration file or a preset quality setting.

        Parameters:
        ----------
        config_file : str, optional
            Path to the configuration file.
        preset_quality : str, optional
            Preset quality setting (e.g., "good", "medium", "best"). Default is "medium_quality"
        model_ids : dict, optional
            Dictionary of model IDs to use for specific modules.
            Example: {"generator_model_id": "mistral.mistral-7b-instruct-v0:2", "retriever_model_id": "BAAI/bge-large-en", "reranker_model_id": "nv_embed"}
        data_dir : str
            The directory containing the data files that will be used for the RAG pipeline
        web_urls : List[str]
            List of website URLs to be ingested and processed.
        base_urls : List[str]
            List of optional base URLs to check for links recursively. The base URL controls which URLs will be processed during recursion.
            The base_url does not need to be the same as the web_url. For example. the web_url can be "https://auto.gluon.ai/stable/index.html", and the base_urls will be "https://auto.gluon.ai/stable/"/
        login_info: dict
            A dictionary containing login credentials for each URL. Required if the target URL requires authentication.
            Must be structured as {target_url: {"login_url": <login_url>, "credentials": {"username": "your_username", "password": "your_password"}}}
            The target_url is a url that is present in the list of web_urls
        parse_urls_recursive: bool
            Whether to parse each URL in the provided recursively. Setting this to True means that the child links present in each parent webpage will also be processed.
        pipeline_batch_size: int
            Optional batch size to use for pre-processing stage (Data Processing, Embedding, Vector DB Module)

        Methods:
        -------
        initialize_data_module()
            Initializes the Data Processing module.

        initialize_embeddings_module()
            Initializes the Embedding module.

        initialize_vectordb_module()
            Initializes the Vector DB module.

        initialize_retriever_module()
            Initializes the Retriever module.

        initialize_generator_module()
            Initializes the Generator module.

        initialize_reranker_module()
            Initializes the Reranker module.

        process_data() -> pd.DataFrame
            Processes the data in the provided data directory using the initialized Data Processing module.

        generate_embeddings(processed_data: pd.DataFrame) -> pd.DataFrame
            Generates embeddings from the processed data using the initialized Embedding module.

        construct_vector_db(embeddings: pd.DataFrame)
            Constructs the vector database using the provided embeddings.

        load_existing_vector_db(index_path: str, metadata_path: str) -> bool
            Loads an existing Vector Database from the specified paths in the configuration.

        save_index_and_metadata(index_path: str, metadata_path: str)
            Saves the vector database index and metadata to the specified paths in the configuration.

        retrieve_context_for_query(query: str) -> List[Dict[str, Any]]
            Retrieves relevant context for the provided query using the Retriever module.

        generate_response(query: str) -> str
            Generates a response to the provided query using the Generator module.

        batched_processing()
            Processes documents, generates embeddings, and stores them in the vector database in batches.

        initialize_rag_pipeline()
            Initializes the entire RAG pipeline by setting up all necessary modules.
        """
        logger.info("\n\nAutoGluon-RAG\n\n")

        self.args = Arguments(config_file)

        self.preset_quality = preset_quality
        self.model_ids = model_ids

        self.config = config_file or self._load_preset()
        self.args = Arguments(self.config) if not self.args else self.args

        # will short-circuit to provided data_dir if config value also provided
        self.data_dir = data_dir or self.args.data_dir
        self.web_urls = web_urls or self.args.web_urls
        self.base_urls = base_urls or self.args.base_urls
        self.parse_urls_recursive = parse_urls_recursive or self.args.parse_urls_recursive
        self.login_info = login_info or self.args.login_info

        if not self.data_dir and not self.web_urls:
            raise ValueError("Either data_dir or web_urls argument must be provided")

        self.data_processing_module = None
        self.embedding_module = None
        self.vector_db_module = None
        self.reranker_module = None
        self.retriever_module = None
        self.generator_module = None

        self.batch_size = pipeline_batch_size or self.args.pipeline_batch_size

        self.pipeline_initialized = False

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
            web_urls=self.web_urls,
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            file_exts=self.args.data_file_extns,
            html_tags_to_extract=self.args.html_tags_to_extract,
            login_info=self.login_info,
        )
        logger.info("Data Processing module initialized")

    def initialize_embeddings_module(self):
        """Initializes the Embedding module."""
        self.embedding_module = EmbeddingModule(
            model_name=self.args.embedding_model,
            model_platform=self.args.embedding_model_platform,
            platform_args=self.args.embedding_model_platform_args,
            pooling_strategy=self.args.pooling_strategy,
            normalize_embeddings=self.args.normalize_embeddings,
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
            faiss_index_type=self.args.faiss_index_type,
            faiss_index_params=self.args.faiss_index_params,
            faiss_search_params=self.args.faiss_search_params,
            milvus_db_name=self.args.milvus_db_name,
            milvus_search_params=self.args.milvus_search_params,
            milvus_collection_name=self.args.milvus_collection_name,
            milvus_index_params=self.args.milvus_index_params,
            milvus_create_params=self.args.milvus_create_params,
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
            use_reranker=self.args.use_reranker,
        )
        logger.info("Retriever module initialized")

    def initialize_generator_module(self):
        """Initializes the Generator module."""
        num_gpus = get_num_gpus(self.args.generator_num_gpus)
        logger.info(f"Using number of GPUs: {num_gpus} for Generator Module")

        self.generator_module = GeneratorModule(
            model_name=self.args.generator_model_name,
            model_platform=self.args.generator_model_platform,
            platform_args=self.args.generator_model_platform_args,
            num_gpus=num_gpus,
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
            model_platform=self.args.reranker_model_platform,
            platform_args=self.args.reranker_model_platform_args,
            batch_size=self.args.reranker_batch_size,
            top_k=self.args.reranker_top_k,
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
        logger.info(f"Retrieving and Processing Data from {self.data_processing_module.data_dir}")
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
        save_index(self.vector_db_module.db_type, self.vector_db_module.index, index_path)
        logger.info(f"\nSaving Metadata at {metadata_path}")
        save_metadata(self.vector_db_module.metadata, metadata_path)

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

        retrieved_context = ""
        if self.retriever_module.top_k > 0:
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

    def batched_processing(self):
        """
        Processes documents, generates embeddings, and stores them in the vector database in batches.
        Each batch is processed sequentially.

        - All file paths from the provided data directory are retrieved.
        - The first batch of documents is processed.
        - Embeddings for this batch of processed documents are generated.
        - The embeddings for the current batch are stored in the vector database.
        - Memory is cleared (processed data and generated embeddings for the batch) to prevent memory overload.

        """

        file_paths = get_all_file_paths(self.data_processing_module.data_dir, self.data_processing_module.file_exts)

        web_urls = []
        if self.parse_urls_recursive:
            for idx, url in enumerate(self.web_urls):
                loader = RecursiveUrlLoader(url=url, max_depth=1)
                docs = loader.load()
                urls = extract_sub_links(
                    raw_html=docs[0].page_content, url=url, base_url=self.base_urls[idx], continue_on_failure=True
                )
                urls = [url] + urls
                logger.info(
                    f"\nFound {len(urls)} URLs by recursively parsing the webpage {url} with base URL {self.base_urls[idx]}."
                )
                web_urls.extend(urls)
                if url in self.login_info:
                    for sub_url in urls:
                        self.login_info[sub_url] = self.login_info[url]

        batch_num = 1
        start_doc_id = 0

        for i in range(0, max(len(file_paths), len(web_urls)), self.batch_size):
            logger.info(f"Batch {batch_num}")

            batch_file_paths = file_paths[i : i + self.batch_size]
            batch_urls = web_urls[i : i + self.batch_size]

            # Data Processing
            processed_files_data, last_doc_id = self.data_processing_module.process_files(
                batch_file_paths, start_doc_id=start_doc_id
            )
            processed_urls_data, last_doc_id = self.data_processing_module.process_urls(
                batch_urls, login_info=self.login_info, start_doc_id=last_doc_id
            )
            start_doc_id = last_doc_id
            processed_data = pd.concat([processed_files_data, processed_urls_data]).reset_index(drop=True)

            # Embedding
            embeddings = self.generate_embeddings(processed_data)

            # Vector DB
            self.construct_vector_db(embeddings)

            # Clear memory
            del processed_data
            del embeddings

            batch_num += 1

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
            if self.batch_size == 0:
                logger.info(
                    f"\nNot using batching since batch size of {self.batch_size} was provided. You can change this value by setting pipeline_batch_size in the config file or when initializing AutoGluon RAG."
                )
                processed_data = self.process_data()
                embeddings = self.generate_embeddings(processed_data=processed_data)
                self.construct_vector_db(embeddings=embeddings)
            else:
                logger.info(
                    f"\nUsing batch size of {self.batch_size}. You can change this value by setting pipeline_batch_size in the config file or when initializing AutoGluon RAG."
                )
                self.batched_processing()

            if self.args.save_vector_db_index:
                self.save_index_and_metadata(self.args.vector_db_index_save_path, self.args.metadata_index_save_path)

        self.pipeline_initialized = True
