# This is a tutorial on the optional parameters that can be passed into the `AutoGluonRAG` class when using the package through code. 

These are the <b>optional</b> parameters that can be passed into the `AutoGluonRAG` class:
```python
config_file : str
    Path to a configuration file that will be used to set specific parameters in the RAG pipeline.

preset_quality : str
    If you do not wish to use your own configuration file, you can use a preset configuration file which contains pre-defined arguments.
    You must provide the preset quality setting ("good_quality", "medium_quality", or, "best_quality"). Note that if both config_file and preset_quality are provided, config_file will be prioritized.  

model_ids : dict
    Dictionary of model IDs to use for specific modules.
    Example: {"generator_model_id": "mistral.mistral-7b-instruct-v0:2", "retriever_model_id": "BAAI/bge-large-en", "reranker_model_id": "nv_embed"}

data_dir : str
    The directory containing the data files that will be used for the RAG pipeline. If this value is not provided when initializing the object, it must be provided in the config file. If both are provided, the value in the class instantiation will be prioritized. 

web_urls : List[str] 
    List of website URLs to be ingested and processed. Each URL will processed recursively based on the base URL to include the content of URLs that exist within this URL.
    If this value is not provided when initializing the object, it must be provided in the config file. If both are provided, the value in the class instantiation will be prioritized.

base_urls : List[str]
    List of optional base URLs to check for links recursively. The base URL controls which URLs will be processed during recursion. The base_url does not need to be the same as the web_url. For example. the web_url can be "https://auto.gluon.ai/stable/index.html", and the base_urls will be "https://auto.gluon.ai/stable/".
    If this value is not provided when initializing the object, it must be provided in the config file. If both are provided, the value in the class instantiation will be prioritized.

login_info: dict
    A dictionary containing login credentials for each URL. Required if the target URL requires authentication.
    Must be structured as {target_url: {"login_url": <login_url>, "credentials": {"username": "your_username", "password": "your_password"}}}
    The target_url is a url that is present in the list of web_urls
    
parse_urls_recursive: bool
    Whether to parse each URL in the provided recursively. Setting this to True means that the child links present in each parent webpage will also be processed.

pipeline_batch_size: int
    Batch size to use for pre-processing stage (Data Processing, Embedding, Vector DB Module). This represents the number of files in each batch.
    The default value is 20.
```

**Note**: You may provide both `data_dir` and `web_urls`.