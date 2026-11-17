import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm

from agrag.constants import LOGGER_NAME
from agrag.modules.vector_db.faiss.faiss_db import load_faiss_index, save_faiss_index
from agrag.modules.vector_db.milvus.milvus_db import load_milvus_index, save_milvus_index
from agrag.utils import parse_path

logger = logging.getLogger(LOGGER_NAME)


def cosine_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)


def euclidean_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return -euclidean_distances(embeddings)


def manhattan_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return -manhattan_distances(embeddings)


SUPPORTED_SIMILARITY_FUNCTIONS = {
    "cosine": cosine_similarity_fn,
    "euclidean": euclidean_similarity_fn,
    "manhattan": manhattan_similarity_fn,
}


def remove_duplicates(
    embeddings: List[torch.Tensor], similarity_threshold: float, similarity_fn: str
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Removes duplicate embeddings based on cosine similarity.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be deduplicated.
    similarity_threshold : float
        The threshold for considering embeddings as duplicates based on cosine similarity.
    similarity_fn : str
        The name of the similarity function to use. Must be one of the supported functions.

    Returns:
    -------
    Tuple[List[torch.Tensor], List[int]]
        A list of deduplicated embeddings and their indices.
    """
    if len(embeddings) <= 1:
        return embeddings, list(range(len(embeddings)))

    # A threshold of >= 1.0 means "nothing is similar enough to be a duplicate", i.e.
    # keep everything. Short-circuit before building the O(n^2) similarity matrix --
    # for a large merged corpus (e.g. ~15-20k paragraphs) that matrix is a multi-GB
    # memory spike and a wasted compute pass. This makes >= 1.0 an explicit "dedup off"
    # switch; the default (0.95) path below is unchanged.
    if similarity_threshold >= 1.0:
        return embeddings, list(range(len(embeddings)))

    if similarity_fn not in SUPPORTED_SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"Unsupported similarity function: {similarity_fn}. Please choose from: {list(SUPPORTED_SIMILARITY_FUNCTIONS.keys())}"
        )

    embeddings_array = np.array([emb.numpy().flatten() for emb in embeddings])
    sim_fn = SUPPORTED_SIMILARITY_FUNCTIONS[similarity_fn]
    similarity_matrix = sim_fn(embeddings_array)

    remove = set()
    indices_to_keep = []
    for i in range(len(similarity_matrix)):
        if i in remove:
            continue
        indices_to_keep.append(i)
        duplicates = np.where(similarity_matrix[i, i + 1 :] > similarity_threshold)[0] + (
            i + 1
        )  # Only consider the upper triangle of the similarity matrix.
        remove.update(duplicates)

    deduplicated_embeddings = [embedding for i, embedding in enumerate(embeddings) if i not in remove]
    logger.info(f"Removed {len(remove)} duplicate embeddings")
    return deduplicated_embeddings, indices_to_keep


def pad_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """
    Pads embeddings to ensure they have the same length.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be padded.

    Returns:
    -------
    torch.Tensor
        A tensor containing the padded embeddings.
    """
    max_len = max(embedding.shape[1] for embedding in embeddings)
    padded_embeddings = [
        torch.nn.functional.pad(embedding, (0, 0, 0, max_len - embedding.shape[1])) for embedding in embeddings
    ]
    return torch.cat(padded_embeddings, dim=0)


def save_index(db_type: str, index: Union[faiss.IndexFlatL2], index_path: str) -> None:
    """
    Saves the Vector DB index to disk.

    Parameters:
    ----------
    db_type: str
        The type of Vector DB being used
    index: Union[faiss.IndexFlatL2]
        The Vector DB index to store
    index_path : str
        The path where the index will be saved.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources
    """
    s3_bucket, index_path = parse_path(index_path)
    s3_client = boto3.client("s3") if s3_bucket else None
    if not index:
        raise ValueError("No index to save. Please construct the index first.")
    if not index_path:
        logger.warning(f"Cannot save index. Invalid path {index_path}.")
        return
    basedir = os.path.dirname(index_path)
    if not os.path.exists(basedir):
        logger.info(f"Creating directory for Vector Index save at {basedir}")
        os.makedirs(basedir)
    if not os.path.isfile(index_path):
        with open(index_path, "w") as fp:
            pass
    if db_type == "faiss":
        if not isinstance(
            index,
            (
                faiss.IndexFlatL2,
                faiss.IndexFlatIP,
                faiss.IndexHNSWFlat,
                faiss.IndexLSH,
                faiss.IndexPQ,
                faiss.IndexIVFFlat,
                faiss.IndexScalarQuantizer,
                faiss.IndexIVFPQ,
            ),
        ):
            raise TypeError("Index for FAISS incorrectly created. Not of a valid FAISS index type.")
        success = save_faiss_index(index, index_path)
        if s3_bucket and success:
            save_index_s3(index_path, s3_bucket, s3_client)
        else:
            logger.warning(f"Failed to save index")
    elif db_type == "milvus":
        save_milvus_index()
    else:
        logger.warning(f"Cannot save index. Unsupported Vector DB {db_type}.")


def load_index(
    db_type: str,
    index_path: str,
    pbar: tqdm = None,
) -> Union[faiss.IndexFlatL2]:
    """
    Loads the Vector DB index from disk.

    Parameters:
    ----------
    db_type: str
        The type of Vector DB being used
    index_path : str
        The path from where the index will be loaded.

    Returns:
    -------
    Union[faiss.IndexFlatL2]
        Vector DB Index
    """
    if not index_path:
        logger.warning(f"Cannot load index. Invalid path {index_path}.")
        return False
    index = None
    s3_bucket, index_path = parse_path(index_path)
    basedir = os.path.dirname(index_path)
    if not os.path.exists(basedir):
        logger.info(f"Creating directory for Vector Index load at {basedir}")
        os.makedirs(basedir)
    s3_client = boto3.client("s3") if s3_bucket else None
    if db_type == "faiss":
        if s3_bucket:
            load_index_s3(index_path, s3_bucket, s3_client)
        index = load_faiss_index(index_path)
    elif db_type == "milvus":
        index = load_milvus_index(index_path)
    else:
        raise ValueError("Cannot load index. Unsupported Vector DB {db_type}.")
    if pbar:
        pbar.n = pbar.total
        pbar.refresh()
    return index


def save_metadata(
    metadata: pd.DataFrame,
    metadata_path: str,
):
    """
    Saves metadata to file.

    Parameters:
    ----------
    metadata: pd.DataFrame
        Metadata to store
    metadata_path : str
        The path to the metadata file.

    Returns:
    -------
    bool:
        True, if metadata saved successfully to file
        False, else
    """
    if metadata is None or metadata.empty:
        raise ValueError(
            "No metadata to save. Please construct metadata first using the Data Processing and Embedding Module."
        )
    if not metadata_path:
        logger.warning(f"Cannot save metadata. Invalid path {metadata_path}.")
        return False
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata not a pandas DataFrame.")

    s3_bucket, metadata_path = parse_path(metadata_path)
    s3_client = boto3.client("s3") if s3_bucket else None

    metadata_dir = os.path.dirname(metadata_path)
    if not os.path.exists(metadata_dir):
        logger.info(f"Creating directory for metadata save at {metadata_dir}")
        os.makedirs(metadata_dir)
    if not os.path.isfile(metadata_path):
        with open(metadata_path, "w") as fp:
            pass

    try:
        metadata.to_json(metadata_path, orient="records", lines=True)
        logger.info(f"Metadata saved to {metadata_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        return False

    if s3_bucket:
        try:
            s3_client.upload_file(Filename=metadata_path, Bucket=s3_bucket, Key=metadata_path)
            logger.info(f"Metadata saved to S3 Bucket {s3_bucket} at {metadata_path}.")
            return True
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return False
        except ClientError as e:
            logger.error(f"Failed to upload metadata to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving metadata to S3: {e}")
            return False
    return True


def load_metadata(
    metadata_path: str,
) -> pd.DataFrame:
    """
    Loads metadata from file.

    Parameters:
    ----------
    metadata_path : str
        The path to the metadata file.

    Returns:
    -------
    pd.DataFrame
        Metadata for Vector DB
    """
    if not metadata_path:
        logger.warning(f"Cannot load metadata. Invalid path {metadata_path}.")
        return None
    s3_bucket, metadata_path = parse_path(metadata_path)
    s3_client = boto3.client("s3") if s3_bucket else None
    basedir = os.path.dirname(metadata_path)
    if not os.path.exists(basedir):
        logger.info(f"Creating directory for metadata load at {basedir}")
        os.makedirs(basedir)
    if s3_bucket:
        try:
            s3_client.download_file(Filename=metadata_path, Bucket=s3_bucket, Key=metadata_path)
            logger.info(f"Metadata loaded from S3 Bucket {s3_bucket} at {metadata_path}.")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return pd.DataFrame()
        except ClientError as e:
            logger.error(f"Failed to download metadata from S3: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading metadata from S3: {e}")
            return pd.DataFrame()

    try:
        metadata = pd.read_json(metadata_path, orient="records", lines=True)
        logger.info(f"Metadata loaded from {metadata_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return pd.DataFrame()

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata not a pandas DataFrame.")
    return metadata


def _parent_store_path(metadata_path: str) -> str:
    """Derive the parent-store path that sits next to the metadata file."""
    root, ext = os.path.splitext(metadata_path)
    return f"{root}.parents{ext or '.jsonl'}"


def save_parent_store(parent_store: pd.DataFrame, metadata_path: str) -> bool:
    """Save the parent-chunk store next to the metadata file.

    The parent store is optional (only present with parent-child chunking). It is
    persisted in the same JSON-lines convention as metadata, at a sibling path
    derived from ``metadata_path``. A None/empty store is a no-op.

    S3 paths follow the same ``parse_path``/boto3 convention as ``save_metadata``:
    the store is written locally to the parsed (bucket-relative) path and then
    uploaded, so an ``s3://`` path never creates a literal local ``s3:`` directory.
    """
    if parent_store is None or parent_store.empty:
        return False
    # Derive the sibling parent path from the raw (possibly s3://) metadata path,
    # then parse it, mirroring save_metadata so S3 vs local behave consistently.
    parent_path = _parent_store_path(metadata_path)
    s3_bucket, parent_path = parse_path(parent_path)
    s3_client = boto3.client("s3") if s3_bucket else None

    parent_dir = os.path.dirname(parent_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    try:
        parent_store.to_json(parent_path, orient="records", lines=True)
        logger.info(f"Parent store saved to {parent_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to save parent store to {parent_path}: {e}")
        return False

    if s3_bucket:
        try:
            s3_client.upload_file(Filename=parent_path, Bucket=s3_bucket, Key=parent_path)
            logger.info(f"Parent store saved to S3 Bucket {s3_bucket} at {parent_path}.")
            return True
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return False
        except ClientError as e:
            logger.error(f"Failed to upload parent store to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving parent store to S3: {e}")
            return False
    return True


def load_parent_store(metadata_path: str) -> pd.DataFrame:
    """Load the parent-chunk store that sits next to the metadata file.

    Returns None when no parent store exists (e.g. an index built before
    parent-child chunking, or with flat chunking), so older indexes still load.

    S3 paths follow the same ``parse_path``/boto3 convention as ``load_metadata``:
    the store is downloaded to the parsed (bucket-relative) path before reading,
    so an ``s3://`` path never creates a literal local ``s3:`` directory.
    """
    if not metadata_path:
        return None
    parent_path = _parent_store_path(metadata_path)
    s3_bucket, parent_path = parse_path(parent_path)
    s3_client = boto3.client("s3") if s3_bucket else None

    if s3_bucket:
        try:
            s3_client.download_file(Filename=parent_path, Bucket=s3_bucket, Key=parent_path)
            logger.info(f"Parent store loaded from S3 Bucket {s3_bucket} at {parent_path}.")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return None
        except ClientError as e:
            # A missing object is expected for indexes without a parent store.
            logger.info(f"No parent store in S3 ({e}); skipping.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading parent store from S3: {e}")
            return None
    elif not os.path.isfile(parent_path):
        logger.info(f"No parent store found at {parent_path}; skipping.")
        return None

    try:
        parent_store = pd.read_json(parent_path, orient="records", lines=True)
        logger.info(f"Parent store loaded from {parent_path}")
        return parent_store
    except (IOError, Exception) as e:
        logger.error(f"Failed to load parent store from {parent_path}: {e}")
        return None


# Fields that define whether a persisted index is compatible with the current
# config. A mismatch on any of these means the stored vectors were built under
# different embedding/metric assumptions and must not be silently reused (e.g. an
# IndexFlatIP built from normalized vectors, then loaded under an L2 config, would
# rank results backwards). ``embedding_dim`` guards a hard FAISS dimension error;
# the rest guard silent semantic corruption.
INDEX_FINGERPRINT_FIELDS = (
    "embedding_model",
    "embedding_dim",
    "normalize_embeddings",
    "similarity_fn",
    "faiss_index_type",
)


def _fingerprint_path(metadata_path: str) -> str:
    """Derive the fingerprint sidecar path that sits next to the metadata file."""
    root, ext = os.path.splitext(metadata_path)
    return f"{root}.fingerprint{ext or '.json'}"


def save_index_fingerprint(fingerprint: Dict, metadata_path: str) -> bool:
    """Persist a small index fingerprint next to the metadata file.

    The fingerprint records the embedding model, embedding dimension, and metric
    configuration the index was built under (see ``INDEX_FINGERPRINT_FIELDS``) so a
    later load can refuse a mismatched/stale index instead of blindly reusing
    vectors built under different assumptions. Mirrors the parent-store convention:
    a sibling path derived from ``metadata_path`` with the same ``parse_path``/S3
    handling. A falsy path is a no-op.
    """
    if not metadata_path:
        return False
    fp_path = _fingerprint_path(metadata_path)
    s3_bucket, fp_path = parse_path(fp_path)
    s3_client = boto3.client("s3") if s3_bucket else None

    fp_dir = os.path.dirname(fp_path)
    if fp_dir and not os.path.exists(fp_dir):
        os.makedirs(fp_dir)
    try:
        with open(fp_path, "w") as fp:
            json.dump(fingerprint, fp)
        logger.info(f"Index fingerprint saved to {fp_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to save index fingerprint to {fp_path}: {e}")
        return False

    if s3_bucket:
        try:
            s3_client.upload_file(Filename=fp_path, Bucket=s3_bucket, Key=fp_path)
            logger.info(f"Index fingerprint saved to S3 Bucket {s3_bucket} at {fp_path}.")
            return True
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return False
        except ClientError as e:
            logger.error(f"Failed to upload index fingerprint to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving index fingerprint to S3: {e}")
            return False
    return True


def load_index_fingerprint(metadata_path: str) -> Optional[Dict]:
    """Load the index fingerprint next to the metadata file, or ``None``.

    Returns ``None`` when no fingerprint exists (e.g. an index built before this
    field was added), so older indexes still load -- the caller decides whether a
    missing fingerprint is tolerable or should be treated as a mismatch.
    """
    if not metadata_path:
        return None
    fp_path = _fingerprint_path(metadata_path)
    s3_bucket, fp_path = parse_path(fp_path)
    s3_client = boto3.client("s3") if s3_bucket else None

    if s3_bucket:
        try:
            s3_client.download_file(Filename=fp_path, Bucket=s3_bucket, Key=fp_path)
            logger.info(f"Index fingerprint loaded from S3 Bucket {s3_bucket} at {fp_path}.")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return None
        except ClientError as e:
            logger.info(f"No index fingerprint in S3 ({e}); skipping.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading index fingerprint from S3: {e}")
            return None
    elif not os.path.isfile(fp_path):
        logger.info(f"No index fingerprint found at {fp_path}; skipping.")
        return None

    try:
        with open(fp_path, "r") as fp:
            return json.load(fp)
    except (IOError, ValueError, Exception) as e:
        logger.error(f"Failed to load index fingerprint from {fp_path}: {e}")
        return None


def compare_index_fingerprint(stored: Optional[Dict], current: Dict) -> List[str]:
    """Return a list of human-readable mismatch descriptions (empty when compatible).

    Only ``INDEX_FINGERPRINT_FIELDS`` are compared. A ``None`` stored fingerprint
    (legacy index with no sidecar) yields a single "no fingerprint" notice rather
    than a per-field diff, so the caller can treat it distinctly (warn vs. refuse).
    """
    if stored is None:
        return ["no index fingerprint found (index predates fingerprinting or was built externally)"]
    mismatches = []
    for field in INDEX_FINGERPRINT_FIELDS:
        want = current.get(field)
        got = stored.get(field)
        if got != want:
            mismatches.append(f"{field}: index built with {got!r}, current config expects {want!r}")
    return mismatches


def save_index_s3(
    index_path: str,
    s3_bucket: str,
    s3_client: boto3.session.Session.client,
):
    """
    Saves the index to S3.

    Parameters:
    ----------
    index_path : str
        The path where the index will be saved.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources

    Returns:
    -------
    bool:
        True, if Vector DB Index saved successfully to S3
        False, else
    """
    try:
        s3_client.upload_file(Filename=index_path, Bucket=s3_bucket, Key=index_path)
        logger.info(f"Index saved to S3 Bucket {s3_bucket} at {index_path}")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not found or incomplete.")
        return False
    except ClientError as e:
        logger.error(f"Failed to upload the index to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the index to S3: {e}")
        return False
    finally:
        s3_client.close()


def load_index_s3(
    index_path: str,
    s3_bucket: str,
    s3_client: boto3.session.Session.client,
):
    """
    Loads the index from S3.

    Parameters:
    ----------
    index_path : str
        The path from where the index will be loaded.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources

    Returns:
    -------
    bool:
        True, if Vector DB Index loaded successfully from S3
        False, else
    """
    try:
        s3_client.download_file(Filename=index_path, Bucket=s3_bucket, Key=index_path)
        logger.info(f"Index loaded from S3 Bucket {s3_bucket} and stored at {index_path}")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not found or incomplete.")
        return False
    except ClientError as e:
        logger.error(f"Failed to download the index from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the index from S3: {e}")
        return False
    finally:
        s3_client.close()
