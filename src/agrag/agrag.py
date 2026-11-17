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
from agrag.modules.retriever.retrievers.sparse_retriever import BM25Retriever
from agrag.modules.vector_db.utils import (
    compare_index_fingerprint,
    load_index,
    load_index_fingerprint,
    load_metadata,
    load_parent_store,
    save_index,
    save_index_fingerprint,
    save_metadata,
    save_parent_store,
)
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
            Preset quality setting ("low_quality", "medium_quality", or "high_quality"; only
            "medium_quality" is currently implemented). Default is "medium_quality".
        model_ids : dict, optional
            Reserved for future use. Currently stored but not consumed by the
            pipeline; set model names via the config file or ``agrag.args``
            (e.g. ``generator_model_name``, ``embedding_model``) instead.
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
        self.args = Arguments(self.config) if self.config else self.args

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
        # Optional independent verifier generator (agentic path only); populated by
        # initialize_verifier_generator_module() when agent_verifier_model is set.
        self.verifier_generator_module = None
        self.agentic_module = None
        # Parent-chunk store for parent-child (hierarchical) chunking; populated
        # during data processing or when loading an existing index. None with
        # legacy flat chunking.
        self.parent_store = None

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
        if self.preset_quality not in presets:
            raise ValueError(
                f"Unsupported preset_quality '{self.preset_quality}'. "
                f"Supported presets: {sorted(presets)}. Pass a config_file for other settings."
            )
        logger.info(f"Loading Preset '{self.preset_quality}' configuration")
        return presets[self.preset_quality]

    def initialize_data_module(self):
        """Initializes the Data Processing module."""
        self.data_processing_module = DataProcessingModule(
            data_dir=self.data_dir,
            web_urls=self.web_urls,
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap,
            chunking_strategy=self.args.chunking_strategy,
            children_per_parent=self.args.children_per_parent,
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
        # Construct a BM25 sparse retriever only when hybrid retrieval is enabled;
        # it builds its index lazily from the vector DB metadata on first use.
        sparse_retriever = None
        if self.args.use_hybrid:
            sparse_retriever = BM25Retriever(k1=self.args.bm25_k1, b=self.args.bm25_b)
        self.retriever_module = RetrieverModule(
            vector_database_module=self.vector_db_module,
            embedding_module=self.embedding_module,
            top_k=self.args.retriever_top_k,
            reranker=self.reranker_module,
            num_gpus=num_gpus,
            use_reranker=self.args.use_reranker,
            sparse_retriever=sparse_retriever,
            use_hybrid=self.args.use_hybrid,
            use_rrf=self.args.use_rrf,
            rrf_k=self.args.rrf_k,
            dense_weight=self.args.dense_weight,
            sparse_weight=self.args.sparse_weight,
            use_mmr=self.args.use_mmr,
            mmr_lambda=self.args.mmr_lambda,
            chunk_read=self.args.chunk_read,
            parent_store=getattr(self, "parent_store", None),
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

    def initialize_verifier_generator_module(self):
        """Build a SEPARATE generator for the agentic answer verifier, if configured.

        When ``agent_verifier_model`` is set, the verifier judges with a different
        (ideally stronger) model than the one that drafts the answer -- breaking the
        correlated-error failure mode where the same model both writes and rubber-stamps
        a plausible-but-wrong answer. It reuses the generator platform/args (region,
        bedrock params) so only the model id changes. When unset, returns ``None`` and
        the agentic module falls back to sharing ``self.generator_module`` (prior
        behavior). Kept off the standard RAG path entirely.
        """
        verifier_model = self.args.agent_verifier_model
        if not verifier_model or verifier_model == self.args.generator_model_name:
            self.verifier_generator_module = None
            return
        num_gpus = get_num_gpus(self.args.generator_num_gpus)
        self.verifier_generator_module = GeneratorModule(
            model_name=verifier_model,
            model_platform=self.args.generator_model_platform,
            platform_args=self.args.generator_model_platform_args,
            num_gpus=num_gpus,
        )
        logger.info("Verifier generator module initialized with model %s", verifier_model)

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
        # Capture the parent store built during parent-child chunking (None with
        # legacy flat chunking) so it can be attached to the retriever and saved.
        self.parent_store = getattr(self.data_processing_module, "parent_store", None)
        return processed_data

    def _attach_parent_store_to_retriever(self):
        """Wire the freshly built parent store into the retriever.

        The retriever is initialized before data processing runs, so it starts
        with ``parent_store=None``. After processing (batched or not) populates
        ``self.parent_store``, this attaches it and resets the retriever's parent
        text cache so parent expansion works on the very first query. A None store
        (legacy flat chunking) leaves the retriever's dense-only path unchanged.
        """
        if getattr(self, "retriever_module", None) is None:
            return
        self.retriever_module.parent_store = self.parent_store
        self.retriever_module._parent_text_cache = None

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

        # Refuse to reuse an index built under a different embedding/metric config:
        # e.g. an IndexFlatIP index built from normalized vectors, loaded under an
        # L2/unnormalized config, ranks results backwards and silently degrades
        # retrieval. A legacy index with no fingerprint sidecar loads with a warning
        # (backward compatible), but any concrete field mismatch is a hard error.
        stored_fingerprint = load_index_fingerprint(metadata_path)
        mismatches = compare_index_fingerprint(stored_fingerprint, self._index_fingerprint())
        if mismatches:
            if stored_fingerprint is None:
                logger.warning(
                    "Loaded index has no fingerprint; cannot verify it matches the current "
                    "embedding/metric config. Proceeding, but rebuild the index if retrieval looks wrong."
                )
            else:
                raise ValueError(
                    "Refusing to load a vector DB index that does not match the current config:\n  - "
                    + "\n  - ".join(mismatches)
                    + "\nRebuild the index (set use_existing_vector_db false) or restore the matching config."
                )

        # Load the optional parent store (None for indexes built with flat
        # chunking or before parent-child chunking existed).
        self.parent_store = load_parent_store(metadata_path)
        if self.retriever_module is not None:
            self.retriever_module.parent_store = self.parent_store
            self.retriever_module._parent_text_cache = None

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
        # Persist the parent store alongside metadata when parent-child chunking
        # produced one (no-op otherwise).
        if getattr(self, "parent_store", None) is not None:
            save_parent_store(self.parent_store, metadata_path)
        # Persist a fingerprint of the embedding/metric config the index was built
        # under, so a later load can refuse a mismatched/stale index rather than
        # silently reusing vectors built under different assumptions.
        save_index_fingerprint(self._index_fingerprint(), metadata_path)

    def _index_fingerprint(self) -> Dict[str, Any]:
        """Describe the embedding/metric config that defines index compatibility.

        Fields mirror ``vector_db.utils.INDEX_FINGERPRINT_FIELDS``. ``embedding_dim``
        is read from the built FAISS index (``.d``) when available so it reflects the
        actual stored vectors, not just config; the rest come from ``self.args``.
        """
        embedding_dim = None
        index = getattr(self.vector_db_module, "index", None)
        if index is not None:
            embedding_dim = getattr(index, "d", None)
        return {
            "embedding_model": self.args.embedding_model,
            "embedding_dim": embedding_dim,
            "normalize_embeddings": self.args.normalize_embeddings,
            "similarity_fn": self.args.vector_db_sim_fn,
            "faiss_index_type": self.args.faiss_index_type,
        }

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

    def _agent_config(self) -> Dict[str, Any]:
        """Assemble the agent configuration dict from Arguments."""
        return {
            "max_iterations": self.args.agent_max_iterations,
            "max_subqueries": self.args.agent_max_subqueries,
            "retrieve_top_k_per_query": self.args.agent_retrieve_top_k_per_query,
            "max_context_tokens": self.args.agent_max_context_tokens,
            "use_query_rewrite": self.args.agent_use_query_rewrite,
            "use_context_compression": self.args.agent_use_context_compression,
            "use_verification": self.args.agent_use_verification,
            "allow_abstention": self.args.agent_allow_abstention,
            "max_rewrites": self.args.agent_max_rewrites,
            "min_evidence_count": self.args.agent_min_evidence_count,
            "min_subgoal_coverage": self.args.agent_min_subgoal_coverage,
            "min_relevance": self.args.agent_min_relevance,
            "use_fused_retrieval": self.args.agent_use_fused_retrieval,
            "rrf_k": self.args.rrf_k,
            "use_llm_planner": self.args.agent_use_llm_planner,
            "use_llm_policy": self.args.agent_use_llm_policy,
            "use_strands_planner": self.args.agent_use_strands_planner,
            "use_strands_policy": self.args.agent_use_strands_policy,
            "use_iterative_planner": self.args.agent_use_iterative_planner,
            "use_hop_recovery": self.args.agent_use_hop_recovery,
            "use_replan_recovery": self.args.agent_use_replan_recovery,
            "max_recovery_attempts": self.args.agent_max_recovery_attempts,
            "max_total_hops": self.args.agent_max_total_hops,
            "deep_hop_top_k": self.args.agent_deep_hop_top_k,
            "deep_hop_threshold": self.args.agent_deep_hop_threshold,
            "use_entity_grounding": self.args.agent_use_entity_grounding,
            "final_answer_drop_prefix": self.args.agent_final_answer_drop_prefix,
            "use_direct_answer_fallback": self.args.agent_use_direct_answer_fallback,
            # Real cost budgets (wall-clock seconds, generator-call count,
            # retrieval-call count). Each None -> unbounded; on breach the run
            # returns its best-effort answer tagged ANSWERED_UNVERIFIED with
            # reason ``budget_exhausted``.
            "max_wall_clock_s": self.args.agent_max_wall_clock_s,
            "max_llm_calls": self.args.agent_max_llm_calls,
            "max_retrieval_calls": self.args.agent_max_retrieval_calls,
            # Share the standard-path query prefix so answer formatting is
            # consistent across standard and agentic modes.
            "query_prefix": self.args.generator_query_prefix,
            "extract_answer": self.args.agent_extract_answer,
            "verify_retry": self.args.agent_verify_retry,
            # Dual-candidate synthesis + arbitration and self-consistency on the
            # final draft (both default off/1 -> unchanged behavior).
            "dual_synthesis": self.args.agent_dual_synthesis,
            "self_consistency": self.args.agent_self_consistency,
        }

    def initialize_agentic_module(self):
        """Initializes the Agentic RAG module, reusing the retriever and generator."""
        from agrag.modules.agentic.agentic_module import AgenticRAGModule

        # Build the (optional) independent verifier generator before wiring the module.
        self.initialize_verifier_generator_module()
        self.agentic_module = AgenticRAGModule(
            retriever_module=self.retriever_module,
            generator_module=self.generator_module,
            verifier_generator_module=self.verifier_generator_module,
            config=self._agent_config(),
        )
        logger.info("Agentic RAG module initialized")

    def generate_response(self, query: str, mode: str = None, return_trace: bool = None):
        """
        Generates a response to the provided query using the Generator module.

        By default this uses the standard RAG path: it retrieves relevant context
        for the query using the Retriever module, formats the query and context
        appropriately, and generates a response using the Generator module.

        When ``mode="agentic"`` (or the ``agent`` config block enables it), the
        request is routed to the Agentic RAG path instead, which supports
        planning, multi-query retrieval, evidence tracking, answer verification,
        and abstention.

        Parameters:
        ----------
        query : str
            The user query for which a response is to be generated.
        mode : str, optional
            "standard" (default) or "agentic". If not provided, falls back to the
            ``agent.default_mode`` config value, or "agentic" when ``agent.enabled``
            is set.
        return_trace : bool, optional
            When True, returns ``(answer, trace)`` where ``trace`` is a
            serializable dict describing the run. Supported in both modes: the
            agentic trace describes the full agent loop; the standard trace
            carries the exact structured evidence the generator conditioned on
            plus simple retrieval metrics. When None/False the return value is the
            answer string only (unchanged default behavior). In agentic mode a
            None default falls back to the ``agent.return_trace`` config value.

        Returns:
        -------
        str or Tuple[str, dict]
            The generated response, or ``(response, trace)`` when
            ``return_trace=True``.

        Example:
        --------
        response = agrag.generate_response("What is AutoGluon?")
        answer, trace = agrag.generate_response("What is AutoGluon?", mode="standard", return_trace=True)
        response = agrag.generate_response("How should this repo support agents?", mode="agentic")
        """
        resolved_mode = self._resolve_mode(mode)
        if resolved_mode == "agentic":
            return self._generate_response_agentic(query, return_trace=return_trace)

        # Standard path: retrieve exactly once. When a trace is requested we pull
        # structured records (so the trace can carry the exact evidence used);
        # otherwise we keep the original text-only retrieval byte-for-byte.
        original_query = query
        records = None
        retrieved_context = ""
        if self.retriever_module.top_k > 0:
            if return_trace:
                records = self.retriever_module.retrieve(query, return_metadata=True)
                retrieved_context = [self._record_text(rec) for rec in records] if records else ""
            else:
                retrieved_context = self.retrieve_context_for_query(query)

        query_prefix = self.args.generator_query_prefix
        if query_prefix:
            query = f"{query_prefix}\n{query}"
        formatted_query = format_query(
            model_name=self.generator_module.model_name, query=query, context=retrieved_context
        )

        response = self.generator_module.generate_response(formatted_query)

        logger.info(f"\nResponse: {response}\n")

        if return_trace:
            return response, self._standard_trace(original_query, records, response)
        return response

    @staticmethod
    def _record_text(record: Any) -> str:
        """Extract chunk text from a retriever record (dict) or raw string."""
        if isinstance(record, dict):
            return record.get("text", "")
        return str(record)

    def _standard_trace(self, query: str, records, answer: str) -> Dict[str, Any]:
        """Build a serializable trace for a standard-RAG run.

        The trace records the exact structured evidence the generator conditioned
        on (the retriever records, best-first) plus lightweight retrieval metrics.
        It mirrors the agentic trace's ``original_query``/``final_answer``/
        ``evidence``/``metrics`` keys so downstream consumers (e.g. the benchmark)
        can treat both modes uniformly. Standard RAG performs a single retrieval,
        so ``retrieval_calls`` is 1 when retrieval ran and 0 when ``top_k`` is 0.
        """
        evidence: List[Dict[str, Any]] = []
        if records:
            for rank, record in enumerate(records):
                if isinstance(record, dict):
                    evidence.append(dict(record))
                else:
                    evidence.append({"text": str(record), "rank": rank})
        return {
            "mode": "standard",
            "original_query": query,
            "final_answer": answer,
            "evidence": evidence,
            "metrics": {
                "retrieval_calls": 1 if self.retriever_module.top_k > 0 else 0,
                "evidence_count": len(evidence),
            },
        }

    def _resolve_mode(self, mode: str = None) -> str:
        """Resolve the answering mode from the explicit arg then config.

        Precedence: explicit ``mode`` argument > ``agent.default_mode`` config >
        "agentic" when ``agent.enabled`` is True > "standard".
        """
        if mode:
            return mode
        if self.args.agent_default_mode and self.args.agent_default_mode != "standard":
            return self.args.agent_default_mode
        if self.args.agent_enabled:
            return "agentic"
        return "standard"

    def _generate_response_agentic(self, query: str, return_trace: bool = None):
        """Route the query through the Agentic RAG path."""
        if getattr(self, "agentic_module", None) is None:
            self.initialize_agentic_module()
        if return_trace is None:
            return_trace = self.args.agent_return_trace
        return self.agentic_module.answer(query, return_trace=return_trace)

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

            processed_files_data, last_doc_id = self.data_processing_module.process_files(
                batch_file_paths, start_doc_id=start_doc_id
            )
            processed_urls_data, last_doc_id = self.data_processing_module.process_urls(
                batch_urls, login_info=self.login_info, start_doc_id=last_doc_id
            )
            start_doc_id = last_doc_id
            processed_data = pd.concat([processed_files_data, processed_urls_data]).reset_index(drop=True)

            # Parent-child chunking: group this batch's children into parents. The
            # data module keeps a running parent_id counter and appends to its
            # parent store, so IDs stay globally unique and the store accumulates
            # across batches. No-op for legacy flat chunking.
            if self.data_processing_module.chunking_strategy == "parent_child" and not processed_data.empty:
                processed_data = self.data_processing_module.build_parent_child(processed_data)
                self.parent_store = self.data_processing_module.parent_store

            embeddings = self.generate_embeddings(processed_data)

            self.construct_vector_db(embeddings)

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

            # Attach the parent store (if parent-child chunking produced one) to
            # the retriever, which was initialized before data was processed.
            self._attach_parent_store_to_retriever()

            if self.args.save_vector_db_index:
                self.save_index_and_metadata(self.args.vector_db_index_save_path, self.args.metadata_index_save_path)

        self.pipeline_initialized = True
