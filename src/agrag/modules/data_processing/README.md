## Data Processing Module

This module is responsible for ingesting and processing documents provided by the user

Here are the configurable parameters for this module:

```
data:
  data_dir : The directory containing the data files to be ingested. This can be either a local directory or an S3 URI to a directory in an S3 bucket.

  web_urls: List of website URLs to be ingested and processed.

  base_urls: List of base URLs to check for links recursively. The base URL controls which URLs will be processed during recursion. The base_url does not need to be the same as the web_url. For example. the web_url can be "https://auto.gluon.ai/stable/index.html", and the base_urls will be "https://auto.gluon.ai/stable/"/

  parse_urls_recursive: Whether to parse each URL in the provided recursively. Setting this to True means that the child links present in each parent webpage will also be processed.

  chunk_size : The size of each chunk of text (default is 512).

  chunk_overlap : The overlap between consecutive chunks of text (default is 128).

  file_exts: List of file extensions to support. Default is [".pdf", ".txt", ".docx", ".doc", ".rtf", ".csv", ".md", ".py", ".log"]
  
  html_tags_to_extract: List of HTML tags to extract text from. Default is ["p", "table"].

```