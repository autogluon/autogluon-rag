"""Lightweight text tokenization shared by sparse retrieval and metrics.

Both BM25 sparse retrieval and the retrieval-quality metrics need the same kind
of cheap, dependency-free tokenization (lowercase, alphanumeric words, optional
stopword filtering). Keeping one implementation here avoids two subtly different
tokenizers drifting apart.
"""

import re
from typing import List, Optional, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword set. Dropping these focuses matching on the content
# words that actually distinguish one passage from another.
STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "by",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "s",
    "t",
}


def tokenize(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    """Lowercase ``text`` and split into alphanumeric tokens.

    Parameters
    ----------
    text : str
        The text to tokenize. Non-string / empty input yields an empty list.
    stopwords : set of str, optional
        If provided, tokens in this set are dropped. Pass ``STOPWORDS`` to filter
        common English stopwords; omit to keep all tokens (BM25's default, since
        it already down-weights frequent terms via IDF).

    Returns
    -------
    list of str
        The token list.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if stopwords:
        tokens = [tok for tok in tokens if tok not in stopwords]
    return tokens
