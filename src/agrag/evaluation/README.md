# AutoGluon-RAG Evaluation Module

## Overview

The `EvaluationModule` in AutoGluon-RAG is designed to facilitate the evaluation of the Retrieval-Augmented Generation (RAG) pipeline. This module allows users to easily assess the performance of the RAG pipeline using various datasets and evaluation metrics. The primary purpose of this module is to provide a flexible and extensible framework for evaluating the quality and effectiveness of the generated responses by the RAG pipeline.


The `EvaluationModule` is created to:

1. Simplify the evaluation process for the AutoGluon-RAG pipeline.
2. Support multiple evaluation metrics, including custom metrics.
3. Provide functionality to save evaluation data and results.
4. Allow users to preprocess data, extract queries, and expected responses from various datasets.

## Usage

### Initialization

To initialize the `EvaluationModule`, you need to provide an instance of `AutoGluonRAG`, the name of the dataset, a list of metrics, and additional optional parameters such as preprocessing functions and paths for saving evaluation data.

```python
from agrag.agrag import AutoGluonRAG
from agrag.evaluation.evaluator import EvaluationModule
from module import preprocessing_fn, query_fn, response_fn

evaluation_dir = "evaluation_data"
agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
evaluator = EvaluationModule(rag_instance=agrag)
evaluator.run_evaluation(
    dataset_name="huggingface_dataset/dataset_name",
    metrics=["exact_match", "transformer_matcher"],
    save_evaluation_data=True,
    evaluation_dir=evaluation_dir,
    preprocessing_fn=preprocessing_fn,
    query_fn=query_fn,
    response_fn=response_fn,
    hf_dataset_params={"name": "dev"},
)
```

Refer to `src/agrag/evaluation/datasets/google_natural_questions/evaluate_agrag.py` for a detailed example of how to evaluate AutoGluon-RAG on the Google Natural Questions dataset from HuggingFace.

## Arguments to Evaluation Module

### agrag
**Type**: `AutoGluonRAG`  
**Description**: The AutoGluonRAG instance to be evaluated.
```python
from agrag.agrag import AutoGluonRAG

agrag = AutoGluonRAG(preset_quality="medium_quality", data_dir=evaluation_dir)
# Calling agrag.initialize_rag_pipeline() is optional since the EvaluationModule will initialize the pipeline if it has not been done already.
```

## AutoGluon-RAG for Large Datasets
For large datasets, a naive version of AutoGluon-RAG may not be sufficient. Here are some steps you can take when working with a large corpus for RAG:
1. Using optimized indices for Vector DB. Refer to the documentation for the supported vector databases on how you can use optimized indices such as clustered and quantized databases. Set the parameters appropriately in your configuration file. 

    For example, here is an optimized FAISS setup (in the configuration file) using quantization that runs correctly for the Google Natural Questions dataset:
    ```python
        vector_db:
        db_type: faiss
        faiss_index_type: IndexIVFPQ
        faiss_quantized_index_params:
            nlist: 50
            m: 8
            bits: 8
        faiss_index_nprobe: 15
    ```
2. Use GPUs: Make sure to use GPUs appropriately in each module (wherever applicable). You can set the `num_gpus` parameter in the configuration file under each module.

3. System Memory: Make sure your system has enough RAM for at least the size of the dataset. You will require more memory fir the documents that will be generated from the datasets, the embeddings, the metadata, and the vector db index. We recommend running evaluation on a remote instance (such as AWS EC2) instead of running locally. If you would like to run locally, you can choose to run a subset of the evaluation data by setting `max_eval_size` when calling the `run_evaluation` function (see the next section).


## Arguments to `run_evaluation` function

### NOTE
Every time you run the `run_evaluation` function, you may need to set the `agrag.data_dir` parameter if you change the dataset being used. In that case, you will have to reinitialize the RAG pipeline. 

Alternatively, you can index all your evaluation datasets at once, or create multiple instances of `AutoGluonRAG`.

--------------

### dataset_name
**Type**: `str`  
**Description**: The name of the dataset to use for evaluation.

### metrics
**Type**: `List[Union[str, Callable]]`  
**Description**: The list of metrics to use for evaluation. Supported metrics include:
- `"bertscore"`: Uses the [BERTScore metric](https://huggingface.co/spaces/evaluate-metric/bertscore) from HuggingFace.
- `"bleu"`: Uses the [BLEU metric](https://huggingface.co/spaces/evaluate-metric/bleu) from HuggingFace
- `"exact_match"`: Uses the Inclusive Exact Match metric. This is a custom metric defined in this module since it is a bit more lenient compared to the HuggingFace `exact_match` metric. It also counts events where the expected response is contained within the generated response as a success.
- `"pedant"`: Uses the PEDANT metric from [QA Metrics](https://github.com/zli12321/qa_metrics).
- `"transformer_matcher"`: Uses the Transformer Matcher metric from [QA Metrics](https://github.com/zli12321/qa_metrics).
- `<callable_custom_metric>`: Any callable Python function or a function from a Python package.

### preprocessing_fn
**Type**: `Callable`  
**Description**: A function to preprocess the content before saving. This must be a function that returns the relevant content from the dataset row. For example, to extract text from the Google Natural Questions dataset, you can pass in this function as `preprocessing_fn`:
```python
def preprocess_google_nq(row):
    """
    Extracts text from HTML content for the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset containing HTML content.

    Returns:
    -------
    str
        The extracted text content.
    """
    html_content = row["document"]["html"]
    return extract_text_from_html(html_content) # Function to extract text from HTML
```

### query_fn
**Type**: `Callable`  
**Description**: A function to extract the query from the dataset row.  For example, to extract the query from the Google Natural Questions dataset, you can pass in this function as `query_fn`:
```python
def get_google_nq_query(row):
    """
    Extracts the query from a row in the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset.

    Returns:
    -------
    str
        The query.
    """
    return row["question"]["text"]
```

### response_fn
**Type**: `Callable`  
**Description**: A function to extract the expected responses from the dataset row. For example, to extract the expected responses from the Google Natural Questions dataset, you can pass in this function as `query_fn`:
```python
def get_google_nq_responses(row):
    """
    Extracts the expected responses from a row in the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset.

    Returns:
    -------
    List[str]
        A list of expected responses.
    """
    short_answers = row["annotations"]["short_answers"]
    return [answer["text"][0] for answer in short_answers if answer["text"]]
```

### hf_dataset_params
**Type**: `dict`  
**Description**: Additional parameters to pass into the HuggingFace load_dataset function.

### split
**Type**: `str`  
**Description**: The dataset split to use (default is "validation").

### save_evaluation_data
**Type**: `bool`  
**Description**: Whether to save evaluation data to files (default is True). Set this to False if you already have a directory of evaluation files to pass into AutoGluon RAG.

### evaluation_dir
**Type**: `str`  
**Description**: The directory for evaluation data (default is "./evaluation_data").

### save_csv_path
**Type**: `str`  
**Description**: The path to save the evaluation results as a CSV file (default is None). If no path is provided, the evaluation results will not be saved.

### max_eval_size
**Type**: `int`, optional  
**Description**: The maximum number of datapoints to process for evaluation (default is None). If this value is not less than the total number of datapoints (rows), the entire dataset will be used.
