import argparse
import logging
import os

import yaml

CURRENT_DIR = os.path.dirname(__file__)

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Arguments:
    """
    A class to handle the per-module arguments and loading of configuration files for the AutoGluon-RAG pipeline.

    Attributes:
    ----------
    config_file : str
        Path to configuration file

    Methods:
    -------
    _parse_args() -> argparse.Namespace
        Parses command-line arguments.
    _load_config(config_file: str) -> dict
        Loads configuration from the specified YAML file.
    _load_defaults(default_file: str) -> dict
        Loads default values from the specified YAML file.
    """

    def __init__(self, config_file: str = None):
        if config_file:
            # Use through config-file
            self.config = self._load_config(config_file)
        else:
            # Use through command-line
            self.args = self._parse_args()
            self.config = self._load_config(self.args.config_file)
        self.data_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/data_processing/default.yaml"))
        self.embedding_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/embedding/default.yaml"))
        self.vector_db_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/vector_db/default.yaml"))
        self.retriever_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/retriever/default.yaml"))
        self.generator_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/generator/default.yaml"))
        self.shared_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/shared/default.yaml"))
        self.agent_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/agent/default.yaml"))

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="AutoGluon-RAG - Retrieval-Augmented Generation Pipeline")

        parser.add_argument("--config_file", type=str, help="Path to the configuration file", metavar="")

        parser.add_argument(
            "--preset_quality",
            type=str,
            choices=["low_quality", "medium_quality", "high_quality"],
            default="medium_quality",
            help="Preset quality settings for the RAG pipeline (default: medium_quality)",
            metavar="",
        )
        parser.add_argument("--web_urls", type=str, nargs="*", help="List of URLs to use for RAG", metavar="")
        parser.add_argument(
            "--base_urls",
            type=str,
            nargs="*",
            help="List of base URLs to restrict web URL parsing. Only URLs stemming from a base URL will be processed.",
            metavar="",
        )
        parser.add_argument(
            "--parse_urls_recursive",
            action="store_true",
            help="Enable recursive parsing of all URLs from the provided web URL list",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            help="Directory containing files to use for RAG. Supports local or S3 paths.",
            metavar="",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["standard", "agentic"],
            default=None,
            help="Answering mode. 'standard' (default) uses the single-pass RAG path; "
            "'agentic' uses the multi-step agentic RAG path (planning, multi-query "
            "retrieval, verification, abstention). If omitted, the agent.default_mode / "
            "agent.enabled config values decide.",
            metavar="",
        )

        return parser.parse_args()

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from a YAML file."""
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Error: File not found - {config_file}")
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file - {config_file}: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred while loading {config_file}: {exc}")
        return {}

    def _load_defaults(self, default_file: str) -> dict:
        """Load default values from a YAML file."""
        try:
            with open(default_file, "r") as f:
                defaults = yaml.safe_load(f)
            return defaults
        except FileNotFoundError:
            logger.error(f"Error: File not found - {default_file}")
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file - {default_file}: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred while loading {default_file}: {exc}")
        return {}

    @property
    def pipeline_batch_size(self):
        return self.config.get("shared", {}).get(
            "pipeline_batch_size", self.shared_defaults.get("PIPELINE_BATCH_SIZE")
        )

    @pipeline_batch_size.setter
    def pipeline_batch_size(self, value):
        self.config["shared"]["pipeline_batch_size"] = value

    @property
    def data_dir(self):
        return self.config.get("data", {}).get("data_dir", None)

    @data_dir.setter
    def data_dir(self, value):
        self.config["data"]["data_dir"] = value

    @property
    def web_urls(self):
        return self.config.get("data", {}).get("web_urls", [])

    @web_urls.setter
    def web_urls(self, value):
        self.config["data"]["web_urls"] = value

    @property
    def base_urls(self):
        return self.config.get("data", {}).get("base_urls", [])

    @base_urls.setter
    def base_urls(self, value):
        self.config["data"]["base_urls"] = value

    @property
    def html_tags_to_extract(self):
        return self.config.get("data", {}).get("html_tags_to_extract", self.data_defaults.get("SUPPORTED_HTML_TAGS"))

    @html_tags_to_extract.setter
    def html_tags_to_extract(self, value):
        self.config["data"]["html_tags_to_extract"] = value

    @property
    def login_info(self):
        return self.config.get("data", {}).get("login_info", {})

    @login_info.setter
    def login_info(self, value):
        self.config["data"]["login_info"] = value

    @property
    def parse_urls_recursive(self):
        return self.config.get("data", {}).get("parse_urls_recursive", self.data_defaults.get("PARSE_URLS_RECURSIVE"))

    @parse_urls_recursive.setter
    def parse_urls_recursive(self, value):
        self.config["data"]["parse_urls_recursive"] = value

    @property
    def chunk_size(self):
        return self.config.get("data", {}).get("chunk_size", self.data_defaults.get("CHUNK_SIZE"))

    @chunk_size.setter
    def chunk_size(self, value):
        self.config["data"]["chunk_size"] = value

    @property
    def chunk_overlap(self):
        return self.config.get("data", {}).get("chunk_overlap", self.data_defaults.get("CHUNK_OVERLAP"))

    @chunk_overlap.setter
    def chunk_overlap(self, value):
        self.config["data"]["chunk_overlap"] = value

    @property
    def chunking_strategy(self):
        return self.config.get("data", {}).get("chunking_strategy", self.data_defaults.get("CHUNKING_STRATEGY"))

    @chunking_strategy.setter
    def chunking_strategy(self, value):
        self.config.setdefault("data", {})["chunking_strategy"] = value

    @property
    def children_per_parent(self):
        return self.config.get("data", {}).get("children_per_parent", self.data_defaults.get("CHILDREN_PER_PARENT"))

    @children_per_parent.setter
    def children_per_parent(self, value):
        self.config.setdefault("data", {})["children_per_parent"] = value

    @property
    def data_file_extns(self):
        return self.config.get("data", {}).get("file_extns", self.data_defaults.get("SUPPORTED_FILE_EXTENSIONS"))

    @data_file_extns.setter
    def data_file_extns(self, value):
        self.config["data"]["file_extns"] = value

    @property
    def embedding_model(self):
        return self.config.get("embedding", {}).get(
            "embedding_model", self.embedding_defaults.get("DEFAULT_EMBEDDING_MODEL")
        )

    @embedding_model.setter
    def embedding_model(self, value):
        self.config["embedding"]["embedding_model"] = value

    @property
    def embedding_model_platform(self):
        return self.config.get("embedding", {}).get(
            "embedding_model_platform", self.embedding_defaults.get("DEFAULT_EMBEDDING_MODEL_PLATFORM")
        )

    @embedding_model_platform.setter
    def embedding_model_platform(self, value):
        self.config["embedding"]["embedding_model_platform"] = value

    @property
    def embedding_model_platform_args(self):
        return self.config.get("embedding", {}).get("embedding_model_platform_args", {})

    @embedding_model_platform_args.setter
    def embedding_model_platform_args(self, value):
        self.config["embedding"]["embedding_model_platform_args"] = value

    @property
    def pooling_strategy(self):
        return self.config.get("embedding", {}).get(
            "pooling_strategy", self.embedding_defaults.get("POOLING_STRATEGY")
        )

    @pooling_strategy.setter
    def pooling_strategy(self, value):
        self.config["embedding"]["pooling_strategy"] = value

    @property
    def normalize_embeddings(self):
        return self.config.get("embedding", {}).get(
            "normalize_embeddings", self.embedding_defaults.get("NORMALIZE_EMBEDDINGS")
        )

    @normalize_embeddings.setter
    def normalize_embeddings(self, value):
        self.config["embedding"]["normalize_embeddings"] = value

    @property
    def normalization_params(self):
        return self.config.get("embedding", {}).get("normalization_params", {})

    @normalization_params.setter
    def normalization_params(self, value):
        self.config["embedding"]["normalization_params"] = value

    @property
    def query_instruction_for_retrieval(self):
        return self.config.get("embedding", {}).get("query_instruction_for_retrieval", "")

    @query_instruction_for_retrieval.setter
    def query_instruction_for_retrieval(self, value):
        self.config["embedding"]["query_instruction_for_retrieval"] = value

    @property
    def embedding_batch_size(self):
        return self.config.get("embedding", {}).get(
            "embedding_batch_size", self.embedding_defaults.get("EMBEDDING_BATCH_SIZE")
        )

    @embedding_batch_size.setter
    def embedding_batch_size(self, value):
        self.config["embedding"]["embedding_batch_size"] = value

    @property
    def vector_db_type(self):
        return self.config.get("vector_db", {}).get("db_type", self.vector_db_defaults.get("DB_TYPE"))

    @vector_db_type.setter
    def vector_db_type(self, value):
        self.config["vector_db"]["db_type"] = value

    @property
    def vector_db_args(self):
        return self.config.get("vector_db", {}).get("params", {"gpu": self.vector_db_defaults.get("GPU")})

    @vector_db_args.setter
    def vector_db_args(self, value):
        self.config["vector_db"]["params"] = value

    @property
    def vector_db_sim_threshold(self):
        return self.config.get("vector_db", {}).get(
            "similarity_threshold", self.vector_db_defaults.get("SIMILARITY_THRESHOLD")
        )

    @vector_db_sim_threshold.setter
    def vector_db_sim_threshold(self, value):
        self.config["vector_db"]["similarity_threshold"] = value

    @property
    def vector_db_sim_fn(self):
        return self.config.get("vector_db", {}).get("similarity_fn", self.vector_db_defaults.get("SIMILARITY_FN"))

    @vector_db_sim_fn.setter
    def vector_db_sim_fn(self, value):
        self.config["vector_db"]["similarity_fn"] = value

    @property
    def use_existing_vector_db_index(self):
        return self.config.get("vector_db", {}).get(
            "use_existing_vector_db", self.vector_db_defaults.get("USE_EXISTING_INDEX")
        )

    @use_existing_vector_db_index.setter
    def use_existing_vector_db_index(self, value):
        self.config["vector_db"]["use_existing_vector_db"] = value

    @property
    def save_vector_db_index(self):
        return self.config.get("vector_db", {}).get("save_index", self.vector_db_defaults.get("SAVE_INDEX"))

    @save_vector_db_index.setter
    def save_vector_db_index(self, value):
        self.config["vector_db"]["save_index"] = value

    @property
    def vector_db_num_gpus(self):
        return self.config.get("vector_db", {}).get("num_gpus", None)

    @vector_db_num_gpus.setter
    def vector_db_num_gpus(self, value):
        self.config["vector_db"]["num_gpus"] = value

    @property
    def vector_db_index_save_path(self):
        return self.config.get("vector_db", {}).get(
            "vector_db_index_save_path", self.vector_db_defaults.get("INDEX_PATH")
        )

    @vector_db_index_save_path.setter
    def vector_db_index_save_path(self, value):
        self.config["vector_db"]["vector_db_index_save_path"] = value

    @property
    def metadata_index_save_path(self):
        return self.config.get("vector_db", {}).get(
            "metadata_index_save_path", self.vector_db_defaults.get("METADATA_PATH")
        )

    @metadata_index_save_path.setter
    def metadata_index_save_path(self, value):
        self.config["vector_db"]["metadata_index_save_path"] = value

    @property
    def vector_db_index_load_path(self):
        return self.config.get("vector_db", {}).get(
            "vector_db_index_load_path", self.vector_db_defaults.get("INDEX_PATH")
        )

    @vector_db_index_load_path.setter
    def vector_db_index_load_path(self, value):
        self.config["vector_db"]["vector_db_index_load_path"] = value

    @property
    def metadata_index_load_path(self):
        return self.config.get("vector_db", {}).get(
            "metadata_index_load_path", self.vector_db_defaults.get("METADATA_PATH")
        )

    @metadata_index_load_path.setter
    def metadata_index_load_path(self, value):
        self.config["vector_db"]["metadata_index_load_path"] = value

    @property
    def faiss_index_type(self):
        return self.config.get("vector_db", {}).get(
            "faiss_index_type", self.vector_db_defaults.get("FAISS_INDEX_TYPE")
        )

    @faiss_index_type.setter
    def faiss_index_type(self, value):
        self.config["vector_db"]["faiss_index_type"] = value

    @property
    def faiss_index_params(self):
        return self.config.get("vector_db", {}).get(
            "faiss_index_params", self.vector_db_defaults.get("FAISS_INDEX_PARAMS")
        )

    @faiss_index_params.setter
    def faiss_index_params(self, value):
        self.config["vector_db"]["faiss_index_params"] = value

    @property
    def faiss_search_params(self):
        return self.config.get("vector_db", {}).get(
            "faiss_search_params", self.vector_db_defaults.get("FAISS_SEARCH_PARAMS")
        )

    @faiss_search_params.setter
    def faiss_search_params(self, value):
        self.config["vector_db"]["faiss_search_params"] = value

    @property
    def milvus_search_params(self):
        return self.config.get("vector_db", {}).get(
            "milvus_search_params", self.vector_db_defaults.get("MILVUS_INDEX_PARAMS")
        )

    @milvus_search_params.setter
    def milvus_search_params(self, value):
        self.config["vector_db"]["milvus_search_params"] = value

    @property
    def milvus_collection_name(self):
        return self.config.get("vector_db", {}).get(
            "milvus_collection_name", self.vector_db_defaults.get("MILVUS_DB_COLLECTION_NAME")
        )

    @milvus_collection_name.setter
    def milvus_collection_name(self, value):
        self.config["vector_db"]["milvus_collection_name"] = value

    @property
    def milvus_db_name(self):
        return self.config.get("vector_db", {}).get("milvus_db_name", self.vector_db_defaults.get("MILVUS_DB_NAME"))

    @milvus_db_name.setter
    def milvus_db_name(self, value):
        self.config["vector_db"]["milvus_db_name"] = value

    @property
    def milvus_index_params(self):
        return self.config.get("vector_db", {}).get(
            "milvus_index_params", self.vector_db_defaults.get("MILVUS_INDEX_PARAMS")
        )

    @milvus_index_params.setter
    def milvus_index_params(self, value):
        self.config["vector_db"]["milvus_index_params"] = value

    @property
    def milvus_create_params(self):
        return self.config.get("vector_db", {}).get(
            "milvus_create_params", self.vector_db_defaults.get("MILVUS_CREATE_PARAMS")
        )

    @milvus_create_params.setter
    def milvus_create_params(self, value):
        self.config["vector_db"]["milvus_create_params"] = value

    @property
    def retriever_top_k(self):
        return self.config.get("retriever", {}).get("retriever_top_k", self.retriever_defaults.get("RETRIEVER_TOP_K"))

    @retriever_top_k.setter
    def retriever_top_k(self, value):
        self.config["retriever"]["retriever_top_k"] = value

    @property
    def reranker_top_k(self):
        return self.config.get("retriever", {}).get("reranker_top_k", self.retriever_defaults.get("RERANKER_TOP_K"))

    @reranker_top_k.setter
    def reranker_top_k(self, value):
        self.config["retriever"]["reranker_top_k"] = value

    @property
    def use_reranker(self):
        return self.config.get("retriever", {}).get("use_reranker", self.retriever_defaults.get("USE_RERANKER"))

    @use_reranker.setter
    def use_reranker(self, value):
        self.config["retriever"]["use_reranker"] = value

    @property
    def reranker_model_name(self):
        return self.config.get("retriever", {}).get(
            "reranker_model_name", self.retriever_defaults.get("RERANKER_MODEL")
        )

    @reranker_model_name.setter
    def reranker_model_name(self, value):
        self.config["retriever"]["reranker_model_name"] = value

    @property
    def reranker_model_platform(self):
        return self.config.get("retriever", {}).get(
            "reranker_model_platform", self.retriever_defaults.get("DEFAULT_RERANKER_MODEL_PLATFORM")
        )

    @reranker_model_platform.setter
    def reranker_model_platform(self, value):
        self.config["retriever"]["reranker_model_platform"] = value

    @property
    def reranker_model_platform_args(self):
        return self.config.get("retriever", {}).get("reranker_model_platform_args", {})

    @reranker_model_platform_args.setter
    def reranker_model_platform_args(self, value):
        self.config["retriever"]["reranker_model_platform_args"] = value

    @property
    def reranker_batch_size(self):
        return self.config.get("retriever", {}).get(
            "reranker_batch_size", self.retriever_defaults.get("RERANKER_BATCH_SIZE")
        )

    @reranker_batch_size.setter
    def reranker_batch_size(self, value):
        self.config["retriever"]["reranker_batch_size"] = value

    @property
    def reranker_hf_model_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_model_params", {})

    @reranker_hf_model_params.setter
    def reranker_hf_model_params(self, value):
        self.config["retriever"]["reranker_hf_model_params"] = value

    @property
    def reranker_hf_tokenizer_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_tokenizer_params", {})

    @reranker_hf_tokenizer_params.setter
    def reranker_hf_tokenizer_params(self, value):
        self.config["retriever"]["reranker_hf_tokenizer_params"] = value

    @property
    def reranker_hf_tokenizer_init_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_tokenizer_params", {})

    @reranker_hf_tokenizer_init_params.setter
    def reranker_hf_tokenizer_init_params(self, value):
        self.config["retriever"]["reranker_hf_tokenizer_params"] = value

    @property
    def reranker_hf_forward_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_forward_params", {})

    @reranker_hf_forward_params.setter
    def reranker_hf_forward_params(self, value):
        self.config["retriever"]["reranker_hf_forward_params"] = value

    @property
    def retriever_num_gpus(self):
        return self.config.get("retriever", {}).get("num_gpus", None)

    @retriever_num_gpus.setter
    def retriever_num_gpus(self, value):
        self.config["retriever"]["num_gpus"] = value

    @property
    def use_hybrid(self):
        return self.config.get("retriever", {}).get("use_hybrid", self.retriever_defaults.get("USE_HYBRID"))

    @use_hybrid.setter
    def use_hybrid(self, value):
        self.config.setdefault("retriever", {})["use_hybrid"] = value

    @property
    def use_rrf(self):
        return self.config.get("retriever", {}).get("use_rrf", self.retriever_defaults.get("USE_RRF"))

    @use_rrf.setter
    def use_rrf(self, value):
        self.config.setdefault("retriever", {})["use_rrf"] = value

    @property
    def rrf_k(self):
        return self.config.get("retriever", {}).get("rrf_k", self.retriever_defaults.get("RRF_K"))

    @rrf_k.setter
    def rrf_k(self, value):
        self.config.setdefault("retriever", {})["rrf_k"] = value

    @property
    def dense_weight(self):
        return self.config.get("retriever", {}).get("dense_weight", self.retriever_defaults.get("DENSE_WEIGHT"))

    @dense_weight.setter
    def dense_weight(self, value):
        self.config.setdefault("retriever", {})["dense_weight"] = value

    @property
    def sparse_weight(self):
        return self.config.get("retriever", {}).get("sparse_weight", self.retriever_defaults.get("SPARSE_WEIGHT"))

    @sparse_weight.setter
    def sparse_weight(self, value):
        self.config.setdefault("retriever", {})["sparse_weight"] = value

    @property
    def use_mmr(self):
        return self.config.get("retriever", {}).get("use_mmr", self.retriever_defaults.get("USE_MMR"))

    @use_mmr.setter
    def use_mmr(self, value):
        self.config.setdefault("retriever", {})["use_mmr"] = value

    @property
    def mmr_lambda(self):
        return self.config.get("retriever", {}).get("mmr_lambda", self.retriever_defaults.get("MMR_LAMBDA"))

    @mmr_lambda.setter
    def mmr_lambda(self, value):
        self.config.setdefault("retriever", {})["mmr_lambda"] = value

    @property
    def chunk_read(self):
        return self.config.get("retriever", {}).get("chunk_read", self.retriever_defaults.get("CHUNK_READ"))

    @chunk_read.setter
    def chunk_read(self, value):
        self.config.setdefault("retriever", {})["chunk_read"] = value

    @property
    def bm25_k1(self):
        return self.config.get("retriever", {}).get("bm25_k1", self.retriever_defaults.get("BM25_K1"))

    @bm25_k1.setter
    def bm25_k1(self, value):
        self.config.setdefault("retriever", {})["bm25_k1"] = value

    @property
    def bm25_b(self):
        return self.config.get("retriever", {}).get("bm25_b", self.retriever_defaults.get("BM25_B"))

    @bm25_b.setter
    def bm25_b(self, value):
        self.config.setdefault("retriever", {})["bm25_b"] = value

    @property
    def generator_model_name(self):
        return self.config.get("generator", {}).get(
            "generator_model_name", self.generator_defaults.get("GENERATOR_MODEL")
        )

    @generator_model_name.setter
    def generator_model_name(self, value):
        self.config["generator"]["generator_model_name"] = value

    @property
    def generator_model_platform(self):
        return self.config.get("generator", {}).get(
            "generator_model_platform", self.generator_defaults.get("DEFAULT_GENERATOR_MODEL_PLATFORM")
        )

    @generator_model_platform.setter
    def generator_model_platform(self, value):
        self.config["generator"]["generator_model_platform"] = value

    @property
    def generator_model_platform_args(self):
        return self.config.get("generator", {}).get(
            "generator_model_platform_args",
            self.generator_defaults.get("DEFAULT_GENERATOR_MODEL_PLATFORM_ARGS", {}),
        )

    @generator_model_platform_args.setter
    def generator_model_platform_args(self, value):
        self.config["generator"]["generator_model_platform_args"] = value

    @property
    def generator_num_gpus(self):
        return self.config.get("generator", {}).get("num_gpus", 0)

    @generator_num_gpus.setter
    def generator_num_gpus(self, value):
        self.config["generator"]["num_gpus"] = value

    @property
    def generator_query_prefix(self):
        return self.config.get("generator", {}).get("generator_query_prefix", "")

    @generator_query_prefix.setter
    def generator_query_prefix(self, value):
        self.config["generator"]["generator_query_prefix"] = value

    @property
    def agent_enabled(self):
        return self.config.get("agent", {}).get("enabled", self.agent_defaults.get("AGENT_ENABLED"))

    @agent_enabled.setter
    def agent_enabled(self, value):
        self.config.setdefault("agent", {})["enabled"] = value

    @property
    def agent_default_mode(self):
        return self.config.get("agent", {}).get("default_mode", self.agent_defaults.get("AGENT_DEFAULT_MODE"))

    @agent_default_mode.setter
    def agent_default_mode(self, value):
        self.config.setdefault("agent", {})["default_mode"] = value

    @property
    def agent_max_iterations(self):
        return self.config.get("agent", {}).get("max_iterations", self.agent_defaults.get("AGENT_MAX_ITERATIONS"))

    @agent_max_iterations.setter
    def agent_max_iterations(self, value):
        self.config.setdefault("agent", {})["max_iterations"] = value

    @property
    def agent_max_subqueries(self):
        return self.config.get("agent", {}).get("max_subqueries", self.agent_defaults.get("AGENT_MAX_SUBQUERIES"))

    @agent_max_subqueries.setter
    def agent_max_subqueries(self, value):
        self.config.setdefault("agent", {})["max_subqueries"] = value

    @property
    def agent_retrieve_top_k_per_query(self):
        return self.config.get("agent", {}).get(
            "retrieve_top_k_per_query", self.agent_defaults.get("AGENT_RETRIEVE_TOP_K_PER_QUERY")
        )

    @agent_retrieve_top_k_per_query.setter
    def agent_retrieve_top_k_per_query(self, value):
        self.config.setdefault("agent", {})["retrieve_top_k_per_query"] = value

    @property
    def agent_max_context_tokens(self):
        return self.config.get("agent", {}).get(
            "max_context_tokens", self.agent_defaults.get("AGENT_MAX_CONTEXT_TOKENS")
        )

    @agent_max_context_tokens.setter
    def agent_max_context_tokens(self, value):
        self.config.setdefault("agent", {})["max_context_tokens"] = value

    @property
    def agent_use_query_rewrite(self):
        return self.config.get("agent", {}).get(
            "use_query_rewrite", self.agent_defaults.get("AGENT_USE_QUERY_REWRITE")
        )

    @agent_use_query_rewrite.setter
    def agent_use_query_rewrite(self, value):
        self.config.setdefault("agent", {})["use_query_rewrite"] = value

    @property
    def agent_use_context_compression(self):
        return self.config.get("agent", {}).get(
            "use_context_compression", self.agent_defaults.get("AGENT_USE_CONTEXT_COMPRESSION")
        )

    @agent_use_context_compression.setter
    def agent_use_context_compression(self, value):
        self.config.setdefault("agent", {})["use_context_compression"] = value

    @property
    def agent_use_verification(self):
        return self.config.get("agent", {}).get("use_verification", self.agent_defaults.get("AGENT_USE_VERIFICATION"))

    @agent_use_verification.setter
    def agent_use_verification(self, value):
        self.config.setdefault("agent", {})["use_verification"] = value

    @property
    def agent_min_evidence_count(self):
        return self.config.get("agent", {}).get(
            "min_evidence_count", self.agent_defaults.get("AGENT_MIN_EVIDENCE_COUNT")
        )

    @agent_min_evidence_count.setter
    def agent_min_evidence_count(self, value):
        self.config.setdefault("agent", {})["min_evidence_count"] = value

    @property
    def agent_min_subgoal_coverage(self):
        return self.config.get("agent", {}).get(
            "min_subgoal_coverage", self.agent_defaults.get("AGENT_MIN_SUBGOAL_COVERAGE")
        )

    @agent_min_subgoal_coverage.setter
    def agent_min_subgoal_coverage(self, value):
        self.config.setdefault("agent", {})["min_subgoal_coverage"] = value

    @property
    def agent_min_relevance(self):
        return self.config.get("agent", {}).get("min_relevance", self.agent_defaults.get("AGENT_MIN_RELEVANCE"))

    @agent_min_relevance.setter
    def agent_min_relevance(self, value):
        self.config.setdefault("agent", {})["min_relevance"] = value

    @property
    def agent_allow_abstention(self):
        return self.config.get("agent", {}).get(
            "allow_abstention", self.agent_defaults.get("AGENT_ALLOW_ABSTENTION")
        )

    @agent_allow_abstention.setter
    def agent_allow_abstention(self, value):
        self.config.setdefault("agent", {})["allow_abstention"] = value

    @property
    def agent_max_rewrites(self):
        return self.config.get("agent", {}).get("max_rewrites", self.agent_defaults.get("AGENT_MAX_REWRITES"))

    @agent_max_rewrites.setter
    def agent_max_rewrites(self, value):
        self.config.setdefault("agent", {})["max_rewrites"] = value

    @property
    def agent_return_trace(self):
        return self.config.get("agent", {}).get("return_trace", self.agent_defaults.get("AGENT_RETURN_TRACE"))

    @agent_return_trace.setter
    def agent_return_trace(self, value):
        self.config.setdefault("agent", {})["return_trace"] = value

    @property
    def agent_use_fused_retrieval(self):
        return self.config.get("agent", {}).get(
            "use_fused_retrieval", self.agent_defaults.get("AGENT_USE_FUSED_RETRIEVAL")
        )

    @agent_use_fused_retrieval.setter
    def agent_use_fused_retrieval(self, value):
        self.config.setdefault("agent", {})["use_fused_retrieval"] = value

    @property
    def agent_use_llm_planner(self):
        return self.config.get("agent", {}).get("use_llm_planner", self.agent_defaults.get("AGENT_USE_LLM_PLANNER"))

    @agent_use_llm_planner.setter
    def agent_use_llm_planner(self, value):
        self.config.setdefault("agent", {})["use_llm_planner"] = value

    @property
    def agent_use_llm_policy(self):
        return self.config.get("agent", {}).get("use_llm_policy", self.agent_defaults.get("AGENT_USE_LLM_POLICY"))

    @agent_use_llm_policy.setter
    def agent_use_llm_policy(self, value):
        self.config.setdefault("agent", {})["use_llm_policy"] = value

    @property
    def agent_use_strands_planner(self):
        return self.config.get("agent", {}).get(
            "use_strands_planner", self.agent_defaults.get("AGENT_USE_STRANDS_PLANNER")
        )

    @agent_use_strands_planner.setter
    def agent_use_strands_planner(self, value):
        self.config.setdefault("agent", {})["use_strands_planner"] = value

    @property
    def agent_use_strands_policy(self):
        return self.config.get("agent", {}).get(
            "use_strands_policy", self.agent_defaults.get("AGENT_USE_STRANDS_POLICY")
        )

    @agent_use_strands_policy.setter
    def agent_use_strands_policy(self, value):
        self.config.setdefault("agent", {})["use_strands_policy"] = value

    @property
    def agent_use_iterative_planner(self):
        return self.config.get("agent", {}).get(
            "use_iterative_planner", self.agent_defaults.get("AGENT_USE_ITERATIVE_PLANNER")
        )

    @agent_use_iterative_planner.setter
    def agent_use_iterative_planner(self, value):
        self.config.setdefault("agent", {})["use_iterative_planner"] = value

    @property
    def agent_extract_answer(self):
        return self.config.get("agent", {}).get(
            "extract_answer", self.agent_defaults.get("AGENT_EXTRACT_ANSWER")
        )

    @agent_extract_answer.setter
    def agent_extract_answer(self, value):
        self.config.setdefault("agent", {})["extract_answer"] = value

    @property
    def agent_verify_retry(self):
        return self.config.get("agent", {}).get(
            "verify_retry", self.agent_defaults.get("AGENT_VERIFY_RETRY")
        )

    @agent_verify_retry.setter
    def agent_verify_retry(self, value):
        self.config.setdefault("agent", {})["verify_retry"] = value

    @property
    def agent_use_hop_recovery(self):
        return self.config.get("agent", {}).get(
            "use_hop_recovery", self.agent_defaults.get("AGENT_USE_HOP_RECOVERY")
        )

    @agent_use_hop_recovery.setter
    def agent_use_hop_recovery(self, value):
        self.config.setdefault("agent", {})["use_hop_recovery"] = value

    @property
    def agent_use_replan_recovery(self):
        return self.config.get("agent", {}).get(
            "use_replan_recovery", self.agent_defaults.get("AGENT_USE_REPLAN_RECOVERY")
        )

    @agent_use_replan_recovery.setter
    def agent_use_replan_recovery(self, value):
        self.config.setdefault("agent", {})["use_replan_recovery"] = value

    @property
    def agent_max_recovery_attempts(self):
        return self.config.get("agent", {}).get(
            "max_recovery_attempts", self.agent_defaults.get("AGENT_MAX_RECOVERY_ATTEMPTS")
        )

    @agent_max_recovery_attempts.setter
    def agent_max_recovery_attempts(self, value):
        self.config.setdefault("agent", {})["max_recovery_attempts"] = value

    @property
    def agent_max_total_hops(self):
        return self.config.get("agent", {}).get(
            "max_total_hops", self.agent_defaults.get("AGENT_MAX_TOTAL_HOPS")
        )

    @agent_max_total_hops.setter
    def agent_max_total_hops(self, value):
        self.config.setdefault("agent", {})["max_total_hops"] = value

    @property
    def agent_deep_hop_top_k(self):
        return self.config.get("agent", {}).get(
            "deep_hop_top_k", self.agent_defaults.get("AGENT_DEEP_HOP_TOP_K")
        )

    @agent_deep_hop_top_k.setter
    def agent_deep_hop_top_k(self, value):
        self.config.setdefault("agent", {})["deep_hop_top_k"] = value

    @property
    def agent_deep_hop_threshold(self):
        return self.config.get("agent", {}).get(
            "deep_hop_threshold", self.agent_defaults.get("AGENT_DEEP_HOP_THRESHOLD")
        )

    @agent_deep_hop_threshold.setter
    def agent_deep_hop_threshold(self, value):
        self.config.setdefault("agent", {})["deep_hop_threshold"] = value

    @property
    def agent_use_entity_grounding(self):
        return self.config.get("agent", {}).get(
            "use_entity_grounding", self.agent_defaults.get("AGENT_USE_ENTITY_GROUNDING")
        )

    @agent_use_entity_grounding.setter
    def agent_use_entity_grounding(self, value):
        self.config.setdefault("agent", {})["use_entity_grounding"] = value

    @property
    def agent_final_answer_drop_prefix(self):
        return self.config.get("agent", {}).get(
            "final_answer_drop_prefix", self.agent_defaults.get("AGENT_FINAL_ANSWER_DROP_PREFIX")
        )

    @agent_final_answer_drop_prefix.setter
    def agent_final_answer_drop_prefix(self, value):
        self.config.setdefault("agent", {})["final_answer_drop_prefix"] = value

    @property
    def agent_use_direct_answer_fallback(self):
        return self.config.get("agent", {}).get(
            "use_direct_answer_fallback", self.agent_defaults.get("AGENT_USE_DIRECT_ANSWER_FALLBACK")
        )

    @agent_use_direct_answer_fallback.setter
    def agent_use_direct_answer_fallback(self, value):
        self.config.setdefault("agent", {})["use_direct_answer_fallback"] = value

    @property
    def agent_verifier_model(self):
        return self.config.get("agent", {}).get(
            "verifier_model", self.agent_defaults.get("AGENT_VERIFIER_MODEL")
        )

    @agent_verifier_model.setter
    def agent_verifier_model(self, value):
        self.config.setdefault("agent", {})["verifier_model"] = value

    @property
    def agent_dual_synthesis(self):
        return self.config.get("agent", {}).get(
            "dual_synthesis", self.agent_defaults.get("AGENT_DUAL_SYNTHESIS")
        )

    @agent_dual_synthesis.setter
    def agent_dual_synthesis(self, value):
        self.config.setdefault("agent", {})["dual_synthesis"] = value

    @property
    def agent_self_consistency(self):
        return self.config.get("agent", {}).get(
            "self_consistency", self.agent_defaults.get("AGENT_SELF_CONSISTENCY")
        )

    @agent_self_consistency.setter
    def agent_self_consistency(self, value):
        self.config.setdefault("agent", {})["self_consistency"] = value

    @property
    def agent_max_wall_clock_s(self):
        return self.config.get("agent", {}).get(
            "max_wall_clock_s", self.agent_defaults.get("AGENT_MAX_WALL_CLOCK_S")
        )

    @agent_max_wall_clock_s.setter
    def agent_max_wall_clock_s(self, value):
        self.config.setdefault("agent", {})["max_wall_clock_s"] = value

    @property
    def agent_max_llm_calls(self):
        return self.config.get("agent", {}).get(
            "max_llm_calls", self.agent_defaults.get("AGENT_MAX_LLM_CALLS")
        )

    @agent_max_llm_calls.setter
    def agent_max_llm_calls(self, value):
        self.config.setdefault("agent", {})["max_llm_calls"] = value

    @property
    def agent_max_retrieval_calls(self):
        return self.config.get("agent", {}).get(
            "max_retrieval_calls", self.agent_defaults.get("AGENT_MAX_RETRIEVAL_CALLS")
        )

    @agent_max_retrieval_calls.setter
    def agent_max_retrieval_calls(self, value):
        self.config.setdefault("agent", {})["max_retrieval_calls"] = value
