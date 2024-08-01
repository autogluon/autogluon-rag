## This is a tutorial on optimizing FAISS Vector Database for large scale data.

By default, AutoGluon-RAG uses the brute-force version of FAISS - `IndexFlatL2`. `IndexFlatL2` measures the L2 (or Euclidean) distance between the given query vector and all the vectors stored in the database. While this is highly accurate, it is extremely time-consuming since it has a time-complexity of $O(n)$ (where $n$ is the number of vectors stored in the database). If you have a 1M (million) vectors stored, this would result in 1M comparisons with the query vector. If you have multiple queries, this process would have to be repeated for each one.
Thus, it is not the best option for scaling with the size of the data. To solve this, FAISS makes use of two optimized indices:

### Partitioned/Clustered Index

### Quantized Index
