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

    def __init__(self):
        self.args = self._parse_args()
        self.config = self._load_config(self.args.config_file)
        self.data_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/data_processing/default.yaml"))
        self.embedding_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/embedding/default.yaml"))
        self.vector_db_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/vector_db/default.yaml"))

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
    def data_s3_bucket(self):
        return self.config.get("data", {}).get("s3_bucket", None)

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
    def vector_db_index_path(self):
        return self.config.get("vector_db", {}).get("vector_db_index_path", self.vector_db_defaults.get("INDEX_PATH"))

    @property
    def use_existing_vector_db_index(self):
        return self.config.get("vector_db", {}).get(
            "use_existing_vector_db", self.vector_db_defaults.get("USE_EXISTING_INDEX")
        )

    @property
    def vector_db_s3_bucket(self):
        return self.config.get("vector_db", {}).get("s3_bucket", None)

    @property
    def vector_db_num_gpus(self):
        return self.config.get("vector_db", {}).get("num_gpus", None)
