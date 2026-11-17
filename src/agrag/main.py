import argparse

from agrag.agrag import AutoGluonRAG


def ag_rag():
    # Only the answering mode is parsed here; the demo pipeline config is
    # hardcoded below. Pass --mode agentic to route queries through the agentic
    # RAG path (otherwise the agent.* config values decide, defaulting to
    # standard single-pass RAG).
    parser = argparse.ArgumentParser(description="AutoGluon-RAG interactive demo")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "agentic"],
        default=None,
        help="Answering mode: 'standard' (single-pass RAG) or 'agentic' (multi-step "
        "planning/verification/abstention). If omitted, the config decides.",
    )
    cli_args, _ = parser.parse_known_args()

    agrag = AutoGluonRAG(
        preset_quality="medium_quality",  # or path to config file
        web_urls=["https://auto.gluon.ai/stable/index.html"],  # List of URLs to use for RAG
        base_urls=["https://auto.gluon.ai/stable/"],  # List of base URLs to use when processing web
        # URLs. Only Web URLs that stem from a base URL
        # will be processed.
        parse_urls_recursive=True,  # Whether to recursively parse all URLs from the provided web url list
        data_dir="s3://autogluon-rag-github-dev/autogluon_docs/",  # Directory containing files to use for RAG
    )

    agrag.initialize_rag_pipeline()
    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            break

        response = agrag.generate_response(query_text, mode=cli_args.mode)
        print(response)


if __name__ == "__main__":
    ag_rag()
