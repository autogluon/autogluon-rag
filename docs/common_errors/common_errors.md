# List of common errors in AutoGluon-RAG

## 1. Shape Mismatch / Shape Errors
A common error you may encounter when setting up the RAG pipeline end-to-end is shape errors when creating embeddings or storing them in the database. Here are some quick fixes to this problem:
1. Pooling for embedding model: Some models may require pooling the output to obtain the actual embeddings. Make sure you check the documentation of the model you are using to decide what pooling is required.
2. Document/Website access: Make sure the documents and/or websites you provide are accessible to the package. The provided files may be blocked at a system-level and provided websites may not be accessible by web scrapers. This could lead to no data being processed and empty embedding arrays.

## 2. Model Access Issues
Refer to the tutorial [Accessing models through different services](../tutorials/general/model_access.md) for more information about model access setup. These refer to both embedding models and generative models.

## 3. AWS Resource Issues
Refer to the tutorial [Using AWS resources and services](../tutorials/general/aws_resources.md) for more information about AWS access setup.
