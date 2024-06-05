import argparse
import logging
import os
import yaml

CURRENT_DIR = os.path.dirname(__file__)

logger = logging.getLogger("rag-logger")

class Arguments:
    def __init__(self):
        self.args = self._parse_args()
        self.config = self._load_config(self.args.config_file)
        self.data_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/data_processing/default.yaml"))
        self.embedding_defaults = self._load_defaults(os.path.join(CURRENT_DIR, "configs/embedding/default.yaml"))

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
    def s3_bucket(self):
        return self.config.get("data", {}).get("s3_bucket", None)

    @property
    def hf_embedding_model(self):
        return self.config.get("embedding", {}).get("embedding_model", self.embedding_defaults.get("DEFAULT_EMBEDDING_MODEL"))

    @property
    def pooling_strategy(self):
        return self.config.get("embedding", {}).get("pooling_strategy", None)

    @property
    def normalize_embeddings(self):
        return self.config.get("embedding", {}).get("normalize_embeddings", False)

    @property
    def hf_model_params(self):
        return self.config.get("embedding", {}).get("hf_model_params", {})

    @property
    def hf_tokenizer_params(self):
        return self.config.get("embedding", {}).get("hf_tokenizer_params", {"truncation": True, "padding": True})

    @property
    def hf_forward_params(self):
        return self.config.get("embedding", {}).get("hf_forward_params", {})

    @property
    def normalization_params(self):
        return self.config.get("embedding", {}).get("normalization_params", {})
