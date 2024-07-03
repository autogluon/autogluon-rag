import logging

from agrag.agrag import AutoGluonRAG

logger = logging.getLogger("rag-logger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def ag_rag():
    agrag = AutoGluonRAG(preset_quality="medium")
    agrag.initialize_rag_pipeline()
    while True:
        query_text = input(
            "Please enter a query for your RAG pipeline, based on the documents you provided (type 'q' to quit): "
        )
        if query_text == "q":
            break

        response = agrag.generate_response(query=query_text)

        logger.info(f"\nResponse: {response}\n")


if __name__ == "__main__":
    ag_rag()
