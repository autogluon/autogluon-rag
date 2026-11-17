"""BM25 sparse retrieval over the ingested corpus.

The dense retriever answers by embedding similarity; a lexical signal is a
complementary strength (exact term matches, rare tokens, out-of-embedding
vocabulary). ``BM25Retriever`` builds a self-contained BM25Okapi index (numpy
only, no new dependency) over the same corpus the dense index was built from —
the ``text`` column of ``VectorDatabaseModule.metadata`` — and returns hits as
``(row_index, score)`` pairs. Those ``row_index`` values are positional indices
into ``metadata``, exactly matching what ``search_vector_database`` returns, so
dense and sparse hits resolve to the same rows and fuse cleanly.

The index is built lazily on the first search (or explicitly via ``build``) and
cached. Only construct this retriever when hybrid retrieval is enabled.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from agrag.constants import LOGGER_NAME
from agrag.modules.retriever.tokenization import tokenize

logger = logging.getLogger(LOGGER_NAME)


class BM25Retriever:
    """A minimal BM25Okapi sparse retriever.

    Attributes:
    ----------
    k1 : float
        Term-frequency saturation parameter (higher = TF matters more).
    b : float
        Document-length normalization in ``[0, 1]`` (1 = full normalization).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus_size: int = 0
        self._avg_doc_len: float = 0.0
        self._doc_lengths: Optional[np.ndarray] = None
        self._doc_freqs: List[dict] = []  # per-doc term -> count
        self._idf: dict = {}
        self._built = False

    def build(self, documents: Sequence[str]) -> "BM25Retriever":
        """Build the BM25 index from an ordered list of document texts.

        ``documents[i]`` must correspond to metadata row ``i`` so returned row
        indices align with the dense index.
        """
        tokenized = [tokenize(doc) for doc in documents]
        self._corpus_size = len(tokenized)
        self._doc_lengths = np.array([len(doc) for doc in tokenized], dtype=float)
        self._avg_doc_len = float(self._doc_lengths.mean()) if self._corpus_size else 0.0

        self._doc_freqs = []
        term_doc_count: dict = {}
        for tokens in tokenized:
            freqs: dict = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            self._doc_freqs.append(freqs)
            for tok in freqs:
                term_doc_count[tok] = term_doc_count.get(tok, 0) + 1

        # Okapi BM25 IDF with the standard +1 smoothing so it is always positive.
        self._idf = {
            term: np.log(1 + (self._corpus_size - count + 0.5) / (count + 0.5))
            for term, count in term_doc_count.items()
        }
        self._built = True
        logger.debug("BM25 index built over %d documents", self._corpus_size)
        return self

    def _score_document(self, query_tokens: Sequence[str], doc_index: int) -> float:
        freqs = self._doc_freqs[doc_index]
        doc_len = self._doc_lengths[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self._idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1.0))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        return score

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return the top-``k`` documents as ``(row_index, score)``, best-first.

        Documents with a zero score (no query term present) are excluded. If the
        index has not been built, returns an empty list.
        """
        if not self._built or self._corpus_size == 0:
            logger.warning("BM25Retriever.search called before build (or on empty corpus).")
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = np.array(
            [self._score_document(query_tokens, i) for i in range(self._corpus_size)],
            dtype=float,
        )
        # Only rank documents that matched at least one query term.
        nonzero = np.nonzero(scores)[0]
        if nonzero.size == 0:
            return []
        order = nonzero[np.argsort(scores[nonzero])[::-1]][:top_k]
        return [(int(idx), float(scores[idx])) for idx in order]
