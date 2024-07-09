from agrag.agrag import AutoGluonRAG


def ag_rag():
    agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir="s3://autogluon-rag-github-dev/autogluon_docs/")
    agrag.initialize_rag_pipeline()
    agrag.generate_response("What is AutoGluon?")


if __name__ == "__main__":
    ag_rag()
