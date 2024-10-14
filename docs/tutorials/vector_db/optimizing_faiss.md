# Optimizing FAISS Vector Database for large scale data.

By default, AutoGluon-RAG uses the brute-force version of FAISS - `IndexFlatL2`. `IndexFlatL2` measures the L2 (or Euclidean) distance between the given query vector and all the vectors stored in the database. While this is highly accurate, it is extremely time-consuming since it has a time-complexity of $O(n)$ (where $n$ is the number of vectors stored in the database). If you have a 1M (million) vectors stored, this would result in 1M comparisons with the query vector. If you have multiple queries, this process would have to be repeated for each one.
Thus, it is not the best option for scaling with the size of the data. To solve this, FAISS makes use of two optimized indices:

## Partitioned/Clustered Index
Partitioned indices, also known as clustered indices, use clustering algorithms like k-means to partition the data into multiple clusters. This way, only a subset of clusters needs to be searched, significantly reducing the number of comparisons.

### Steps to create a Partitioned Index:
```python
from faiss import IndexFlatL2, IndexIVFFlat
import faiss

d = 128 # Dimension of the vectors
nlist = 100 # Number of clusters

# Initialize the quantizer
quantizer = IndexFlatL2(d)
# Initialize the IVF index
index = IndexIVFFlat(quantizer, d, nlist)

# Train the index with your data
index.train(vectors) # Assuming 'vectors' is a numpy array of your data

# Add vectors to the index
index.add(vectors)

# Search the index
D, I = index.search(query_vectors, k)
```

## Quantized Index
Quantized indices reduce the precision of the vectors to lower the memory footprint and improve search speed. Product Quantization (PQ) is a common technique used in FAISS for this purpose.

### Steps to create a Quantized Index:
```python
from faiss import IndexFlatL2, IndexIVFPQ
import faiss

d = 128 # Dimension of the vectors
nlist = 100 # Number of clusters
m = 8  # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid

# Initialize the quantizer
quantizer = IndexFlatL2(d)
# Initialize the IVFPQ index
index = IndexIVFPQ(quantizer, d, nlist, m, bits)

# Train the index with your data
index.train(vectors) # Assuming 'vectors' is a numpy array of your data

# Add vectors to the index
index.add(vectors)

# Search the index
D, I = index.search(query_vectors, k)
```

## Choosing the Right Index
* `IndexFlatL2`: Use when accuracy is the primary concern, and the dataset is relatively small.
* `IndexIVFFlat`: Use when dealing with large datasets and you need to speed up the search process while maintaining reasonable accuracy.
* `IndexPQ`: Use when you need to optimize for memory usage and speed at the cost of some precision.
You can also use `IndexFlatIP`, `IndexHNSWFlat`, `IndexLSH`, `IndexScalarQuantizer`, or `IndexIVFPQ`. For more information, refer to [Faiss indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## Integration into AutoGluon-RAG
You must specify the values in your configuration file or after instantiating your `AutoGluonRAG` object. Refer to [this](https://github.com/autogluon/autogluon-rag/tree/main/documentation/tutorials/general/setting_parameters.md) tutorial on how to modify arguments through code after instantiating an  `AutoGluonRAG` object.

```
vector_db:
    db_type: faiss
    faiss_index_type: IndexFlatL2, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat, IndexLSH, IndexPQ, IndexScalarQuantizer, or IndexIVFPQ
    faiss_quantized_index_params: Parameters to pass into IndexIVFPQ (d, nlist, m, bits)
    faiss_clustered_index_params: Parameters to pass into IndexIVFFlat (d, nlist)
    faiss_index_nprobe: Set nprobe value. This defines how many nearby cells to search. It is applicable for both IndexIVFFlat and IndexIVFPQ
```