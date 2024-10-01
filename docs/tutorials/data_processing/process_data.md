# Using AutoGluon-RAG to process data from documents/websites.

```python
agrag = AutoGluonRAG(
            data_dir="path/to/data", 
            preset_quality="medium_quality", # or path to config file
        ) 
agrag.initialize_data_module() 

processed_data = self.process_data()
```

Here, instead of calling `initialize_rag_pipeline` to initialize the entire pipeline, we simply initialize the data module to process the data.
`process_data` returns a `pandas DataFrame` with the following columns: `"doc_id", "chunk_id", "text"`.

You can obtain the actual text by:

```python
text_list = processed_data["text"].tolist()
text_array = np.array(text_list)
```