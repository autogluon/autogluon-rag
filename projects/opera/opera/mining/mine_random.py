import json
import os.path
import random

import numpy as np
import pandas as pd


def find_random_neg(query_df, corpus_df, negative_number):
    assert query_df.index.name == "query_id"
    assert corpus_df.index.name == "doc_id"
    query_ids = query_df.index
    doc_ids = corpus_df.index
    num_query = len(query_ids)

    neg_query_ids = np.repeat(query_ids, negative_number)
    neg_doc_ids = np.random.choice(doc_ids, size=num_query * negative_number, replace=True)

    neg_df = pd.DataFrame({"query_id": neg_query_ids, "doc_id": neg_doc_ids})

    return neg_df
