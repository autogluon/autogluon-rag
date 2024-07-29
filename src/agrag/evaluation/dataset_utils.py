from typing import List


def get_document_content(row: dict) -> str:
    """
    Extracts document content from the dataset.

    Parameters:
    ----------
    row : dict
        A row from the dataset containing the document content that will be passed into RAG.

    Returns:
    -------
    str
        The extracted text content.
    """
    return row["content"]


def get_query(row: dict) -> str:
    """
    Extracts the query from a row in the dataset.

    Parameters:
    ----------
    row : dict
        A row from the dataset.

    Returns:
    -------
    str
        The query.
    """
    return row["query"]


def get_expected_responses(row: dict) -> List[str]:
    """
    Extracts the expected responses from a row in the dataset.

    Parameters:
    ----------
    row : dict
        A row from the dataset.

    Returns:
    -------
    List[str]
        A list of expected responses.
    """
    return row["expected_responses"]
