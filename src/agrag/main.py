from agrag.agrag import AutoGluonRAG


def ag_rag():
    agrag = AutoGluonRAG(preset_quality="medium", data_dir="s3://autogluon-rag-github-dev/autogluon_docs/")
    agrag.initialize_rag_pipeline()
    response = agrag.generate_responses()


if __name__ == "__main__":
    ag_rag()
