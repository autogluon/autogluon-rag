from agrag.agrag import AutoGluonRAG


def ag_rag():
    agrag = AutoGluonRAG(preset_quality="medium", data_dir="s3://autogluon-rag-github-dev/autogluon_docs/")
    agrag.initialize_rag_pipeline()
    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            break

        response = agrag.generate_response(query=query_text)


if __name__ == "__main__":
    ag_rag()
