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

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="AutoGluon-RAG - Retrieval-Augmented Generation Pipeline")
        parser.add_argument(
            "--config_file", type=str, help="Path to the configuration file", metavar="", required=True
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
    def web_urls(self, value):
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
        return self.config.get("generator", {}).get("generator_model_platform_args", {})

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
