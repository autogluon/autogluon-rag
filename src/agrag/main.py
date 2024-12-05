from agrag.agrag import AutoGluonRAG


def ag_rag():
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

        agrag.generate_response(query_text)


if __name__ == "__main__":
    ag_rag()
