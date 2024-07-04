import argparse
import logging
import os

import yaml

CURRENT_DIR = os.path.dirname(__file__)

logger = logging.getLogger("rag-logger")


class Arguments:
    """
    A class to handle the per-module arguments and loading of configuration files for the AutoGluon-RAG pipeline.

    Attributes:
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.
    config : dict
        The loaded configuration from the specified YAML file.
    data_defaults : dict
        The default values for the data processing module loaded from a YAML file.
    embedding_defaults : dict
        The default values for the embedding module loaded from a YAML file.

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

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="AutoGluon-RAG - Retrieval-Augmented Generation Pipeline")
        parser.add_argument(
            "--config_file",
            type=str,
            help="Path to the configuration file",
            metavar="",
            required=True,
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
    def data_dir(self):
        return self.config.get("data", {}).get("data_dir", None)

    @property
    def chunk_size(self):
        return self.config.get("data", {}).get("chunk_size", self.data_defaults.get("CHUNK_SIZE"))

    @property
    def chunk_overlap(self):
        return self.config.get("data", {}).get("chunk_overlap", self.data_defaults.get("CHUNK_OVERLAP"))

    @property
    def data_file_extns(self):
        return self.config.get("data", {}).get("file_extns", [])

    @property
    def hf_embedding_model(self):
        return self.config.get("embedding", {}).get(
            "embedding_model", self.embedding_defaults.get("DEFAULT_EMBEDDING_MODEL")
        )

    @property
    def pooling_strategy(self):
        return self.config.get("embedding", {}).get(
            "pooling_strategy", self.embedding_defaults.get("POOLING_STRATEGY")
        )

    @property
    def normalize_embeddings(self):
        return self.config.get("embedding", {}).get(
            "normalize_embeddings", self.embedding_defaults.get("NORMALIZE_EMBEDDINGS")
        )

    @property
    def hf_model_params(self):
        return self.config.get("embedding", {}).get("hf_model_params", {})

    @property
    def hf_tokenizer_params(self):
        return self.config.get("embedding", {}).get("hf_tokenizer_params", {"truncation": True, "padding": True})

    @property
    def hf_tokenizer_init_params(self):
        return self.config.get("embedding", {}).get("hf_tokenizer_params", {})

    @property
    def hf_forward_params(self):
        return self.config.get("embedding", {}).get("hf_forward_params", {})

    @property
    def normalization_params(self):
        return self.config.get("embedding", {}).get("normalization_params", {})

    @property
    def query_instruction_for_retrieval(self):
        return self.config.get("embedding", {}).get("query_instruction_for_retrieval", "")

    @property
    def embedding_batch_size(self):
        return self.config.get("embedding", {}).get("embedding_batch_size", "")

    @property
    def vector_db_type(self):
        return self.config.get("vector_db", {}).get("db_type", self.vector_db_defaults.get("DB_TYPE"))

    @property
    def vector_db_args(self):
        return self.config.get("vector_db", {}).get("params", {"gpu": self.vector_db_defaults.get("GPU")})

    @property
    def vector_db_sim_threshold(self):
        return self.config.get("vector_db", {}).get(
            "similarity_threshold", self.vector_db_defaults.get("SIMILARITY_THRESHOLD")
        )

    @property
    def vector_db_sim_fn(self):
        return self.config.get("vector_db", {}).get("similarity_fn", self.vector_db_defaults.get("SIMILARITY_FN"))

    @property
    def use_existing_vector_db_index(self):
        return self.config.get("vector_db", {}).get(
            "use_existing_vector_db", self.vector_db_defaults.get("USE_EXISTING_INDEX")
        )

    @property
    def save_vector_db_index(self):
        return self.config.get("vector_db", {}).get("save_index", self.vector_db_defaults.get("SAVE_INDEX"))

    @property
    def vector_db_num_gpus(self):
        return self.config.get("vector_db", {}).get("num_gpus", None)

    @property
    def vector_db_index_save_path(self):
        return self.config.get("vector_db", {}).get(
            "vector_db_index_save_path", self.vector_db_defaults.get("INDEX_PATH")
        )

    @property
    def metadata_index_save_path(self):
        return self.config.get("vector_db", {}).get(
            "metadata_index_save_path", self.vector_db_defaults.get("METADATA_PATH")
        )

    @property
    def vector_db_index_load_path(self):
        return self.config.get("vector_db", {}).get(
            "vector_db_index_load_path", self.vector_db_defaults.get("INDEX_PATH")
        )

    @property
    def metadata_index_load_path(self):
        return self.config.get("vector_db", {}).get(
            "metadata_index_load_path", self.vector_db_defaults.get("METADATA_PATH")
        )

    @property
    def retriever_top_k(self):
        return self.config.get("retriever", {}).get("retriever_top_k", self.retriever_defaults.get("RETRIEVER_TOP_K"))

    @property
    def reranker_top_k(self):
        return self.config.get("retriever", {}).get("reranker_top_k", self.retriever_defaults.get("RERANKER_TOP_K"))

    @property
    def use_reranker(self):
        return self.config.get("retriever", {}).get("use_reranker", self.retriever_defaults.get("USE_RERANKER"))

    @property
    def reranker_model_name(self):
        return self.config.get("retriever", {}).get(
            "reranker_model_name", self.retriever_defaults.get("RERANKER_MODEL")
        )

    @property
    def reranker_batch_size(self):
        return self.config.get("retriever", {}).get(
            "reranker_batch_size", self.retriever_defaults.get("RERANKER_BATCH_SIZE")
        )

    @property
    def reranker_hf_model_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_model_params", {})

    @property
    def reranker_hf_tokenizer_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_tokenizer_params", {})

    @property
    def reranker_hf_tokenizer_init_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_tokenizer_params", {})

    @property
    def reranker_hf_forward_params(self):
        return self.config.get("retriever", {}).get("reranker_hf_forward_params", {})

    @property
    def retriever_num_gpus(self):
        return self.config.get("retriever", {}).get("num_gpus", None)

    @property
    def generator_model_name(self):
        return self.config.get("generator", {}).get(
            "generator_model_name", self.generator_defaults.get("GENERATOR_MODEL")
        )

    @property
    def generator_num_gpus(self):
        return self.config.get("generator", {}).get("num_gpus", 0)

    @property
    def generator_hf_model_params(self):
        return self.config.get("generator", {}).get("generator_hf_model_params", {})

    @property
    def generator_hf_tokenizer_params(self):
        return self.config.get("generator", {}).get("generator_hf_tokenizer_params", {})

    @property
    def generator_hf_tokenizer_init_params(self):
        return self.config.get("generator", {}).get("generator_hf_tokenizer_params", {})

    @property
    def generator_hf_forward_params(self):
        return self.config.get("generator", {}).get("generator_hf_forward_params", {})

    @property
    def generator_hf_generate_params(self):
        return self.config.get("generator", {}).get("generator_hf_generate_params", {})

    @property
    def generator_query_prefix(self):
        return self.config.get("generator", {}).get("generator_query_prefix", "")

    @property
    def gpt_generate_params(self):
        return self.config.get("generator", {}).get("gpt_generate_params", {})

    @property
    def use_vllm(self):
        return self.config.get("generator", {}).get("use_vllm", self.generator_defaults.get("USE_VLLM"))

    @property
    def vllm_sampling_params(self):
        return self.config.get("generator", {}).get("vllm_sampling_params", {})

    @property
    def openai_key_file(self):
        return self.config.get("generator", {}).get("openai_key_file", "")

    @property
    def use_bedrock(self):
        return self.config.get("generator", {}).get("use_bedrock", self.generator_defaults.get("USE_BEDROCK"))

    @property
    def bedrock_generate_params(self):
        return self.config.get("generator", {}).get("bedrock_generate_params", {})

    @property
    def generator_local_model_path(self):
        return self.config.get("generator", {}).get("local_model_path", None)
