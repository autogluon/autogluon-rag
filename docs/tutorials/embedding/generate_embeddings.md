# This is a tutorial on using AutoGluon-RAG to generate embeddings.

```python
agrag = AutoGluonRAG(
            data_dir="path/to/data", 
            preset_quality="medium_quality", # or path to config file
        )
agrag.initialize_data_module() 
agrag.initialize_embedding_module() 

processed_data = self.process_data()
embeddings = agrag.generate_embeddings(processed_data=processed_data)
```

Here, instead of calling `initialize_rag_pipeline` to initialize the entire pipeline, we simply initialize the data and embedding modules to generate the embeddings.
`generate_embeddings` returns a `pandas DataFrame` with the following columns: `"doc_id", "chunk_id", "text", "embedding", "all_embeddings_hidden_dim"`.

You can obtain the actual embeddings by:

```python
embeddings_list = embeddings["embedding"].tolist()
embeddings_array = np.array(embeddings_list)
```
