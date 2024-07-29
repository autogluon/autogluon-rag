from ragify.ragify import RAGify


def ag_rag():
    ragify = RAGify(preset_quality="medium_quality", data_dir="s3://autogluon-rag-github-dev/autogluon_docs/")
    ragify.initialize_rag_pipeline()
    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            break

        ragify.generate_response(query_text)


if __name__ == "__main__":
    ag_rag()
