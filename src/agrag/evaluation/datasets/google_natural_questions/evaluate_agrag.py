from agrag.evaluation.datasets.google_natural_questions.google_nq import (
    get_google_nq_query,
    get_google_nq_responses,
    preprocess_google_nq,
)


def evaluate_rag_google_nq(evaluator):
    """
    Evaluates AutoGluon-RAG using the Google Natural Questions dataset.
    https://huggingface.co/datasets/google-research-datasets/natural_questions/
    https://github.com/google-research-datasets/natural-questions/

    Parameters:
    ----------
    evaluator : EvaluationModule
        The EvaluationModule instance to use for evaluation.
    """
    if evaluator.save_evaluation_data:
        evaluator.save_documents_to_files(
            evaluation_dir=evaluator.evaluation_dir, preprocessing_fn=preprocess_google_nq
        )

    evaluator.agrag.args.generator_query_prefix = (
        "You will be answering questions based on Wikipedia articles. "
        "Your response must contain only the actual answer itself and no extra information or words. Provide just the answer instead of forming full sentences. "
        "Your answer may even be a singular word or number. For example, if the query is \"what percentage of the earth's surface is water?\", your response must simply be 'roughly 78%', or '78%'. "
    )
    evaluator.agrag.initialize_rag_pipeline()

    queries, expected_repsonses, generated_responses = evaluator.get_queries_and_responses(
        query_fn=get_google_nq_query, response_fn=get_google_nq_responses
    )
    evaluator.evaluate_responses(predictions=generated_responses, references=expected_repsonses, queries=queries)
    if evaluator.save_csv_path:
        evaluator.save_evaluation_results(
            output_csv=evaluator.save_csv_path, references=expected_repsonses, queries=queries
        )
