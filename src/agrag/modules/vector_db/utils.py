import logging
import os
from typing import List, Tuple, Union

from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
from agrag.constants import LOGGER_NAME
from agrag.modules.vector_db.faiss.faiss_db import load_faiss_index, save_faiss_index
from agrag.modules.vector_db.milvus.milvus_db import load_milvus_index, save_milvus_index
from agrag.utils import parse_path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm

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