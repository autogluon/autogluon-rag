"""
Defining constants in this project
"""

import os

try:
    OPERA_RAW_DATA_DIR = os.environ["OPERA_RAW_DATA_DIR"]
    OPERA_WORK_ROOT = os.environ["OPERA_WORK_ROOT"]
except KeyError as e:
    raise KeyError(f"Please export root directories for data and checkpoints to system before running OPERA: {e}")


# Dataframe Type (df_type)
DIR = "dir"  # returning the directory
CORPUS = "corpus"
QUERY = "query"
DOC = "doc"
POSPAIR = "pospair"
NEGPAIR = "negpair"

DF_TYPES = [DIR, CORPUS, QUERY, DOC, POSPAIR, NEGPAIR]


# Splits
TRAIN = "train"
VAL = "val"
TEST = "test"

SPLITS = [TRAIN, VAL, TEST]


# Numerical constants for pruning/scoring
SCORE_PADDING_VALUE = -999999.0  # Padding value for invalid scores
LOSS_OFFSET = 100.0  # Offset to ensure loss values are positive
NUMERICAL_EPSILON = 1e-6  # Small value to prevent division by zero
