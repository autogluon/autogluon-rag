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
    final_query = f"{query}\n"
    if context:
        final_query += f"\nHere is some useful context:\n{context}"

    model_name = model_name.lower()

    if "mistral" in model_name:
        formatted_query = f"[INST] {final_query} [/INST]"
    elif "llama" in model_name:
        if context:
            formatted_query = f"<s>[INST] <<SYS>> {context} <</SYS>> \n\n {query} [/INST]"
        else:
            formatted_query = f"<s>[INST] {query} [/INST]"
    elif "anthropic" in model_name or "claude" in model_name:
        # Anthropic/Claude models are served via Bedrock's messages API
        # (see BedrockGenerator.generate_response), which wraps the text in a user
        # role itself. So the content is plain text (query + context); the legacy
        # "\n\nHuman:/\n\nAssistant:" text-completion scaffolding must not be used.
        formatted_query = final_query
    elif "gpt-" in model_name:
        formatted_query = final_query
    else:
        # Default formatting for other HuggingFace models
        if context:
            formatted_query = f"User: {query}\n\nContext:\n{context}\n\nResponse:"
        else:
            formatted_query = f"User: {query}\n\nResponse:"

    return formatted_query
