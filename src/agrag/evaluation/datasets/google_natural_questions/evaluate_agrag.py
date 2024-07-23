from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.google_natural_questions.google_nq import (
    get_google_nq_query,
    get_google_nq_responses,
    preprocess_google_nq,
)
from agrag.evaluation.evaluator import EvaluationModule


def evaluate_rag_google_nq():
    """
    Evaluates AutoGluon-RAG using the Google Natural Questions dataset.
    https://huggingface.co/datasets/google-research-datasets/natural_questions/
    https://github.com/google-research-datasets/natural-questions/

    Parameters:
    ----------
    evaluator : EvaluationModule
        The EvaluationModule instance to use for evaluation.
    """
    evaluation_dir = "evaluation_data_google_nq"
    agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
    evaluator = EvaluationModule(
        agrag=agrag,
        dataset_name="google-research-datasets/natural_questions",
        metrics=["exact_match", "transformer_matcher"],
        preprocessing_fn=preprocess_google_nq,
        query_fn=get_google_nq_query,
        response_fn=get_google_nq_responses,
        hf_dataset_params={"name": "dev"},
    )

    evaluator.agrag.args.generator_query_prefix = (
        "You will be answering questions based on Wikipedia articles. "
        "Your response must contain only the actual answer itself and no extra information or words. Provide just the answer instead of forming full sentences. "
        "Your answer may even be a singular word or number. For example, if the query is \"what percentage of the earth's surface is water?\", your response must simply be 'roughly 78%', or '78%'. "
    )
    evaluator.run_evaluation()
