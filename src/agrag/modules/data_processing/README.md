## Data Processing Module

This module is responsible for ingesting and processing documents provided by the user

Here are the configurable parameters for this module:

```
data:
  data_dir : The directory containing the data files to be ingested. This can be either a local directory or an S3 URI to a directory in an S3 bucket.

  web_urls: List of website URLs to be ingested and processed. Each URL will not be recursively processed. Only the content that exists in each URL will be ingested into the pipeline.

  chunk_size : The size of each chunk of text (default is 512).

  chunk_overlap : The overlap between consecutive chunks of text (default is 128).

  file_exts: List of file extensions to support. Default is [".pdf", ".txt", ".docx", ".doc", ".rtf", ".csv", ".md", ".py", ".log"]
  
  html_tags_to_extract: List of HTML tags to extract text from. Default is ["p", "table"].

```