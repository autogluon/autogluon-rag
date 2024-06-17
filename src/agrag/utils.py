from typing import Optional, Tuple


def parse_path(path: str) -> Tuple[Optional[str], str]:
    """
    Parses a given path to determine if it is an S3 path or a local path.

    Parameters:
    ----------
    path : str
        The full path to be parsed.

    Returns:
    -------
    Tuple[Optional[str], str]
        A tuple containing:
        - The S3 bucket name if the path is an S3 path, otherwise None.
        - The actual path within the S3 bucket or the local path.
    """
    if path.startswith("s3://"):
        parts = path.split("/", 3)
        s3_bucket = parts[2]
        s3_path = parts[3] if len(parts) > 3 else ""
        return s3_bucket, s3_path
    else:
        return None, path
