from typing import List


def format_query(model_name: str, query: str, context: List[str]) -> str:
    """
    Formats the query and context based on the model requirements.

    Parameters:
    ----------
    model_name : str
        The name of the model to format the query for.
    query : str
        The user's query.
    context : List[str]
        The context related to the query.

    Returns:
    -------
    str
        The formatted query.
    """
    final_query = f"{query}\n\nHere is some useful context:\n{context}"

    if model_name.lower() == "mistral":
        formatted_query = f"[INST] {final_query} [/INST]"
    elif "anthropic" in model_name.lower():
        formatted_query = f"\n\nHuman: {query}\n\nAssistant: Here is some useful context:\n{context}\n\nAssistant:"
    elif "gpt-" in model_name.lower():
        formatted_query = final_query
    else:
        # Default formatting for other HuggingFace models
        formatted_query = f"User: {query}\n\nContext:\n{context}\n\nResponse:"

    return formatted_query
