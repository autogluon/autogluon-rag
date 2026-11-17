import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, EMBEDDING_KEY, LOGGER_NAME, PARENT_ID_KEY
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.fusion import dedup_records, default_dedup_key, mmr, reciprocal_rank_fusion
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.sparse_retriever import BM25Retriever
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger(LOGGER_NAME)


class RetrieverModule:
    """
    Initializes the RetrieverModule with the VectorDatabaseModule.

    Attributes:
    ----------
    vector_database_module : VectorDatabaseModule
        The module containing the vector database and metadata.
    embedding_model: EmbeddingModule,
        The module for generating embeddings.
    top_k: int
        The top-k documents to retrieve (default is 50).
        If this is set to 0, no documents will be retrieved and the generator will be used without providing additional context.
    reranker: Reranker
        Optional Reranker object to use for reranking
    use_reranker: bool
        Whether or not to use a reranker.
    sparse_retriever : Optional[BM25Retriever]
        Optional BM25 sparse retriever for hybrid retrieval. When provided and
        ``use_hybrid`` is True, its lexical hits are fused with the dense hits.
    use_hybrid : bool
        Whether to combine dense and sparse (BM25) retrieval. Default False keeps
        the legacy dense-only behavior.
    use_rrf : bool
        Whether to fuse ranked lists with Reciprocal Rank Fusion. Implied by
        ``use_hybrid``; kept separate so RRF can also be toggled independently.
    rrf_k : int
        The RRF constant (default 60).
    dense_weight, sparse_weight : float
        Per-signal weights applied during RRF fusion.
    use_mmr : bool
        Whether to apply Maximal Marginal Relevance for diversity after reranking.
    mmr_lambda : float
        MMR relevance/diversity trade-off in ``[0, 1]``.
    chunk_read : int
        Neighbor window: after retrieval, expand each chunk with +/- this many
        adjacent chunks (same document) and its parent chunk when a parent store
        is present. 0 disables expansion (default).
    parent_store : Optional[pd.DataFrame]
        Parent-chunk store (columns ``parent_id``, ``text``) produced by
        parent-child chunking. Used by ``chunk_read`` to expand a child into its
        parent context. None when parent-child chunking was not used.
    **kwargs : dict
        Additional parameters for `RetrieverModule`.

    Methods:
    -------
    encode_query(query: str) -> np.ndarray:
        Encodes the query into an embedding.

    retrieve(query: str, return_metadata: bool = False) -> List[Any]:
        Retrieves the top_k most similar embeddings to the query. Returns text
        chunks by default, or structured records when ``return_metadata`` is True.
    """

    def __init__(
        self,
        vector_database_module: VectorDatabaseModule,
        embedding_module: EmbeddingModule,
        top_k: int = 50,
        reranker: Reranker = None,
        use_reranker: bool = True,
        sparse_retriever: Optional[BM25Retriever] = None,
        use_hybrid: bool = False,
        use_rrf: bool = False,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        chunk_read: int = 0,
        parent_store: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        self.embedding_module = embedding_module
        self.vector_database_module = vector_database_module
        self.top_k = top_k
        self.reranker = None
        if use_reranker:
            assert isinstance(reranker, Reranker), "reranker must be of type <class> Reranker"
            self.reranker = reranker

        self.sparse_retriever = sparse_retriever
        self.use_hybrid = use_hybrid
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.chunk_read = chunk_read
        self.parent_store = parent_store
        # Cache: parent_id -> parent text, built lazily from the parent store.
        self._parent_text_cache: Optional[Dict[Any, str]] = None

    @property
    def _uses_advanced_path(self) -> bool:
        """True when any hybrid/fusion/diversity/expansion feature is enabled.

        When False, ``retrieve`` runs the original dense-only code path unchanged,
        preserving legacy behavior byte-for-byte. A present ``parent_store`` also
        opts in, so parent-child (small-to-big) expansion takes effect whenever an
        index built with parent-child chunking is loaded.
        """
        return bool(
            self.use_hybrid or self.use_rrf or self.use_mmr or self.chunk_read or self.parent_store is not None
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes the query into an embedding.

        Parameters:
        ----------
        query : str
            The query to be encoded.

        Returns:
        -------
        np.ndarray
            The embedding of the query.
        """
        if self.embedding_module.model_platform == "bedrock" and "cohere" in self.embedding_module.model_name:
            self.embedding_module.bedrock_embedding_params["input_type"] = "search_query"
        query_embedding = self.embedding_module.encode(data=pd.DataFrame([{DOC_TEXT_KEY: query}]))
        query_embedding = query_embedding[EMBEDDING_KEY][0]
        return query_embedding

    def retrieve(self, query: str, return_metadata: bool = False, top_k: int = None) -> List[Any]:
        """
        Retrieves the top_k most similar embeddings to the query.

        Parameters:
        ----------
        query : str
            The query to retrieve documents for.
        return_metadata : bool
            If False (default), returns a list of text chunks, preserving the
            original behavior. If True, returns a list of structured records
            (dicts) that keep provenance metadata: ``doc_id`` and ``chunk_id``
            from ingest, plus ``rank`` (position in the returned order, updated
            after reranking). It also carries ``source`` provenance (file path
            or URL) when the ingest metadata has it, ``retrieval_score`` (the raw
            vector-DB similarity/distance), and ``rerank_score`` when a reranker
            ran. The agentic RAG path uses the structured form to build
            ``Evidence`` objects.
        top_k : int, optional
            Overrides the module's ``top_k`` for this call only. If None (default),
            the module-level ``self.top_k`` is used, preserving existing behavior.
            The agentic path uses this to retrieve a per-query number of chunks.

        Returns:
        -------
        List[str] or List[Dict[str, Any]]
            Text chunks (default) or structured records when ``return_metadata``
            is True. Returns None if the vector database search yields no valid
            indices.
        """
        effective_top_k = self.top_k if top_k is None else top_k

        # Hybrid / fusion / MMR / chunk_read path. Only taken when a feature is
        # enabled; otherwise the original dense-only logic below runs unchanged.
        if self._uses_advanced_path:
            return self._retrieve_advanced(query, return_metadata, effective_top_k)

        logger.info(f"\nRetrieving top {effective_top_k} most similar embeddings")
        query_embedding = self.encode_query(query)

        if not return_metadata:
            # Original behavior: indices only, return text chunks.
            indices = self.vector_database_module.search_vector_database(
                embedding=query_embedding, top_k=effective_top_k
            )
            n_rows = self.vector_database_module.metadata.shape[0]
            # FAISS returns -1 for empty slots (fewer than top_k hits); such
            # sentinels must be dropped, not passed to iloc (iloc[-1] would
            # silently return the last metadata row).
            valid_indices = [idx for idx in indices if 0 <= idx < n_rows]
            if not valid_indices:
                logger.warning("No valid indices returned from the vector database search.")
                return None
            records = self.vector_database_module.metadata.iloc[valid_indices].to_dict(orient="records")
            retrieved_content = [chunk["text"] for chunk in records]
            if self.reranker:
                retrieved_content = self.reranker.rerank(query, retrieved_content)
            return retrieved_content

        # Structured behavior: also pull the raw similarity scores so evidence
        # can carry provenance and scoring.
        indices, scores = self.vector_database_module.search_vector_database(
            embedding=query_embedding, top_k=effective_top_k, return_scores=True
        )
        # Keep scores aligned with indices while filtering invalid indices.
        # Only in-range rows (including guarding against the FAISS -1 sentinel)
        # are kept; index/score alignment and ranking order are preserved.
        n_rows = self.vector_database_module.metadata.shape[0]
        valid_pairs = [(idx, score) for idx, score in zip(indices, scores) if 0 <= idx < n_rows]
        if not valid_pairs:
            logger.warning("No valid indices returned from the vector database search.")
            return None
        valid_indices = [idx for idx, _ in valid_pairs]
        valid_scores = [score for _, score in valid_pairs]

        records = self.vector_database_module.metadata.iloc[valid_indices].to_dict(orient="records")

        # Keep metadata aligned with each chunk, tagging pre-rerank rank + score.
        text_to_record = {}
        for rank, (record, score) in enumerate(zip(records, valid_scores)):
            record["rank"] = rank
            record["retrieval_score"] = score
            # Map text -> record so we can realign after reranking. If the same
            # text appears more than once, the first occurrence is kept.
            text_to_record.setdefault(record["text"], record)

        if self.reranker:
            reranked = self.reranker.rerank(query, [record["text"] for record in records], return_scores=True)
            ordered_records = []
            for new_rank, (text, rerank_score) in enumerate(reranked):
                record = dict(text_to_record.get(text, {"text": text}))
                record["rank"] = new_rank
                record["rerank_score"] = rerank_score
                ordered_records.append(record)
            return ordered_records

        return records

    # ------------------------------------------------------------------
    # Global multi-query fused retrieval (agentic path)
    # ------------------------------------------------------------------
    def retrieve_fused(
        self,
        subqueries: List[str],
        original_query: Optional[str] = None,
        return_metadata: bool = False,
        top_k: int = None,
        rrf_k: int = None,
    ) -> Optional[List[Any]]:
        """Globally fuse retrieval across several subqueries.

        This is the genuinely-global multi-query pipeline used by the agentic
        ``MultiQueryRetrieveTool``. It deliberately does NOT call ``retrieve``
        per subquery (which would rerank, MMR, expand, and truncate *per query*
        before any global fusion). Instead it collects raw candidates and runs
        each expensive stage exactly once, globally:

        1. For every subquery, collect *raw* dense (and, when ``use_hybrid``,
           sparse) ranked candidate lists — no per-query rerank, MMR, expansion,
           or truncation.
        2. Run one Reciprocal Rank Fusion across all ``(subquery x signal)``
           ranked lists.
        3. Deduplicate by chunk identity while preserving every
           ``retrieval_queries`` and ``source_signals`` that surfaced the chunk.
        4. Run the cross-encoder exactly once, globally, against the *original
           user query* (not a subquery).
        5. Apply MMR once, globally (when ``use_mmr``).
        6. Apply final top-k selection.
        7. Expand parent/neighbor context only after final selection, avoiding
           duplicate parent contexts while preserving child provenance.

        Parameters
        ----------
        subqueries : list of str
            The retrieval subqueries (plan) to fuse. Each contributes one dense
            ranked list and, under hybrid retrieval, one sparse ranked list.
        original_query : str, optional
            The immutable user question. The single global cross-encoder rerank
            scores candidates against *this*, so ranking reflects what the user
            actually asked rather than any one subquery. Defaults to the first
            subquery when not provided.
        return_metadata : bool
            If False, returns expanded text chunks; if True, structured records.
        top_k : int, optional
            Final number of chunks to keep. Defaults to the module ``top_k``.
        rrf_k : int, optional
            RRF constant override. Defaults to the module ``rrf_k``.

        Returns
        -------
        list or None
            Fused, globally-ranked results (text or records), or ``None`` when no
            subquery surfaced any valid candidate.
        """
        effective_top_k = self.top_k if top_k is None else top_k
        effective_rrf_k = self.rrf_k if rrf_k is None else rrf_k
        if not subqueries:
            return None
        if original_query is None:
            original_query = subqueries[0]

        # 1. Raw per-(subquery x signal) candidates, with NO post-processing.
        ranked_lists, weights, all_hits = self._fused_candidates(subqueries, effective_top_k)
        if not all_hits:
            logger.warning("No valid indices returned from fused multi-query retrieval.")
            return None

        # 3. Dedup on chunk identity (the metadata row is a unique chunk, so
        # identical text from different documents stays distinct), merging every
        # retrieval_query / signal that surfaced each chunk.
        deduped = dedup_records(all_hits, key_fn=self._fused_key)
        by_key = {self._fused_key(record): record for record in deduped}

        # 2. One global RRF across all (subquery x signal) ranked lists.
        fused = reciprocal_rank_fusion(ranked_lists, k=effective_rrf_k, weights=weights)
        ordered: List[Dict[str, Any]] = []
        for fusion_rank, (key, rrf_score) in enumerate(fused):
            record = by_key.get(key)
            if record is None:
                continue
            record["rrf_score"] = rrf_score
            record["fusion_rank"] = fusion_rank
            ordered.append(record)

        # 4. One global cross-encoder rerank against the ORIGINAL user query.
        if self.reranker and ordered:
            ordered = self._rerank_records(original_query, ordered)

        # 5. One global MMR diversity pass.
        if self.use_mmr and len(ordered) > 1:
            ordered = self._apply_mmr(original_query, ordered)

        # 6. Final top-k selection (a distinct step, after rerank + MMR).
        ordered = ordered[:effective_top_k]

        # 7. Context expansion only after final selection.
        if self.chunk_read or self.parent_store is not None:
            ordered = self._expand_context(ordered)

        for rank, record in enumerate(ordered):
            record["rank"] = rank
            record.pop("row_index", None)
            record.pop("signal", None)

        if not return_metadata:
            return [record.get(DOC_TEXT_KEY, "") for record in ordered]
        return ordered

    def _fused_key(self, record: Dict[str, Any]) -> Any:
        """Global identity for fusion/dedup in the multi-query path.

        Uses the metadata ``row_index`` when present: each row is a unique chunk,
        so this both dedups the same chunk seen by several subqueries and keeps
        identical text from *different* rows (documents) distinct. Falls back to
        ``(doc_id, chunk_id)`` / text so callers that inject records without a row
        index still behave sensibly.
        """
        if "row_index" in record:
            return ("row", record["row_index"])
        return default_dedup_key(record)

    def _fused_candidates(self, subqueries: List[str], top_k: int):
        """Collect raw dense (+ optional sparse) candidates per subquery.

        Returns ``(ranked_lists, weights, all_hits)`` where each element of
        ``ranked_lists`` is one ``(subquery x signal)`` ranked list of fusion
        keys, ``weights`` is the aligned RRF weight per list, and ``all_hits`` is
        every raw hit (untouched by rerank/MMR/expansion/truncation) for dedup.
        """
        ranked_lists: List[List[Any]] = []
        weights: List[float] = []
        all_hits: List[Dict[str, Any]] = []
        for subquery in subqueries:
            dense_hits = self._dense_hits(subquery, top_k)
            if dense_hits:
                ranked_lists.append([self._fused_key(h) for h in dense_hits])
                weights.append(self.dense_weight)
                all_hits.extend(dense_hits)
            if self.use_hybrid:
                sparse_hits = self._sparse_hits(subquery, top_k)
                if sparse_hits:
                    ranked_lists.append([self._fused_key(h) for h in sparse_hits])
                    weights.append(self.sparse_weight)
                    all_hits.extend(sparse_hits)
        return ranked_lists, weights, all_hits

    # ------------------------------------------------------------------
    # Advanced retrieval (hybrid fusion + global rerank + MMR + chunk_read)
    # ------------------------------------------------------------------
    def _dense_hits(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run dense retrieval, returning per-hit records tagged with signal/rank."""
        query_embedding = self.encode_query(query)
        indices, scores = self.vector_database_module.search_vector_database(
            embedding=query_embedding, top_k=top_k, return_scores=True
        )
        n_rows = self.vector_database_module.metadata.shape[0]
        hits = []
        rank = 0
        for idx, score in zip(indices, scores):
            # Skip invalid indices, including the FAISS -1 sentinel; iloc[-1]
            # would otherwise map to the last metadata row.
            if not 0 <= idx < n_rows:
                continue
            record = self.vector_database_module.metadata.iloc[idx].to_dict()
            record["row_index"] = int(idx)
            record["rank"] = rank
            record["retrieval_score"] = score
            record["signal"] = "dense"
            record["retrieval_query"] = query
            hits.append(record)
            rank += 1
        return hits

    def _sparse_hits(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run BM25 retrieval, returning per-hit records tagged with signal/rank."""
        if self.sparse_retriever is None:
            return []
        # Lazily build the BM25 index the first time it is used. The corpus (the
        # metadata ``text`` column) is only populated after the vector DB is
        # constructed or loaded, which happens after retriever init.
        if not getattr(self.sparse_retriever, "_built", False):
            metadata = self.vector_database_module.metadata
            if metadata is None or metadata.empty:
                logger.warning("Cannot build BM25 index: vector DB metadata is empty.")
                return []
            self.sparse_retriever.build(metadata[DOC_TEXT_KEY].tolist())
        n_rows = self.vector_database_module.metadata.shape[0]
        hits = []
        rank = 0
        for idx, score in self.sparse_retriever.search(query, top_k):
            if not 0 <= idx < n_rows:
                continue
            record = self.vector_database_module.metadata.iloc[idx].to_dict()
            record["row_index"] = int(idx)
            record["rank"] = rank
            record["sparse_score"] = score
            record["signal"] = "sparse"
            record["retrieval_query"] = query
            hits.append(record)
            rank += 1
        return hits

    def _retrieve_advanced(self, query: str, return_metadata: bool, top_k: int):
        """Dense + sparse hybrid retrieval with RRF fusion, global rerank, MMR,
        and chunk_read expansion. Shared by standard and agentic callers."""
        logger.info(f"\nHybrid retrieval: top {top_k} (hybrid={self.use_hybrid}, rrf={self.use_rrf})")

        dense_hits = self._dense_hits(query, top_k)
        sparse_hits = self._sparse_hits(query, top_k) if self.use_hybrid else []

        if not dense_hits and not sparse_hits:
            logger.warning("No valid indices returned from hybrid retrieval.")
            return None

        # Fuse the ranked lists on row index (RRF). A single list fuses to its own
        # order, so dense-only + RRF is a no-op ordering change.
        records = self._fuse(dense_hits, sparse_hits)

        # One global cross-encoder rerank over the fused candidate pool.
        if self.reranker:
            records = self._rerank_records(query, records)

        # Optional diversity re-ordering.
        if self.use_mmr and len(records) > 1:
            records = self._apply_mmr(query, records)

        # Optional context expansion (neighbor window + parent chunk).
        if self.chunk_read or self.parent_store is not None:
            records = self._expand_context(records)

        # Finalize ranks and strip internal bookkeeping keys.
        for new_rank, record in enumerate(records):
            record["rank"] = new_rank
            record.pop("row_index", None)
            record.pop("signal", None)

        if not return_metadata:
            return [record.get("text", "") for record in records]
        return records

    def _fuse(self, dense_hits: List[Dict[str, Any]], sparse_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse dense + sparse hits into one provenance-preserving record list."""
        # Merge duplicate hits (same chunk from dense and sparse) first so each
        # chunk keeps every signal/query that surfaced it.
        merged = dedup_records(dense_hits + sparse_hits)
        by_index = {record["row_index"]: record for record in merged}

        if self.use_rrf or self.use_hybrid:
            ranked_lists = [[h["row_index"] for h in dense_hits]]
            weights = [self.dense_weight]
            if sparse_hits:
                ranked_lists.append([h["row_index"] for h in sparse_hits])
                weights.append(self.sparse_weight)
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)
            ordered = []
            for row_index, rrf_score in fused:
                record = by_index.get(row_index)
                if record is None:
                    continue
                record["rrf_score"] = rrf_score
                ordered.append(record)
            return ordered

        # No RRF: keep dedup order (dense first, then sparse extras).
        return merged

    def _rerank_records(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply the cross-encoder globally, keying results back to records by
        identity (not text) so identical text from different docs is not merged."""
        texts = [record.get("text", "") for record in records]
        # Queue of records per text preserves identity when texts repeat.
        text_to_records: Dict[str, deque] = defaultdict(deque)
        for record in records:
            text_to_records[record.get("text", "")].append(record)

        reranked = self.reranker.rerank(query, texts, return_scores=True)
        ordered = []
        for text, rerank_score in reranked:
            queue = text_to_records.get(text)
            if not queue:
                continue
            record = queue.popleft()
            record["rerank_score"] = rerank_score
            ordered.append(record)
        return ordered

    def _apply_mmr(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-order records for diversity via MMR over freshly-encoded embeddings."""
        texts = [record.get("text", "") for record in records]
        try:
            encoded = self.embedding_module.encode(data=pd.DataFrame({DOC_TEXT_KEY: texts}))
            candidate_embeddings = list(encoded[EMBEDDING_KEY].values)
            query_embedding = self.encode_query(query)
        except Exception as exc:  # pragma: no cover - defensive; MMR is optional
            logger.warning("MMR embedding failed (%s); skipping diversity re-ordering.", exc)
            return records
        keys = list(range(len(records)))
        selected = mmr(
            query_embedding=np.asarray(query_embedding),
            candidate_keys=keys,
            candidate_embeddings=candidate_embeddings,
            lambda_mult=self.mmr_lambda,
        )
        return [records[i] for i in selected]

    def _build_parent_cache(self) -> Dict[Any, str]:
        if self._parent_text_cache is None:
            cache: Dict[Any, str] = {}
            if self.parent_store is not None and not self.parent_store.empty:
                for _, row in self.parent_store.iterrows():
                    cache[row[PARENT_ID_KEY]] = row.get(DOC_TEXT_KEY, "")
            self._parent_text_cache = cache
        return self._parent_text_cache

    def _expand_context(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand each retrieved child chunk into larger context.

        Precise retrieval stays keyed on the child chunk (``doc_id``/``chunk_id``
        are preserved and the child text is kept as ``child_text``), but ``text``
        is replaced with an expanded context: a +/-``chunk_read`` neighbor window
        of adjacent chunks in the same document, and/or the parent chunk when a
        parent store is available.

        Deduplication of parent context: when several selected child chunks share
        the same ``parent_id``, expanding each one independently would emit the
        identical parent text several times, bloating (and over-weighting) the
        context. Instead the first (best-ranked) child carrying a given parent
        keeps the expanded record; later siblings are folded into it, and their
        provenance (``retrieval_queries``, ``source_signals``) and child identity
        are preserved on the surviving record. Records without a parent are never
        collapsed, so distinct chunks are kept.
        """
        metadata = self.vector_database_module.metadata
        has_doc = DOC_ID_KEY in metadata.columns
        has_chunk = CHUNK_ID_KEY in metadata.columns
        parent_cache = self._build_parent_cache()

        # Carrier record per already-expanded parent_id, so siblings collapse.
        parent_carrier: Dict[Any, Dict[str, Any]] = {}
        expanded: List[Dict[str, Any]] = []

        for record in records:
            child_text = record.get(DOC_TEXT_KEY, "")
            record["child_text"] = child_text
            parent_id = record.get(PARENT_ID_KEY)
            has_parent = parent_id is not None and parent_id in parent_cache

            # A sibling of an already-expanded parent: fold provenance in and drop
            # the duplicate parent context rather than emitting it twice.
            if has_parent and parent_id in parent_carrier:
                self._merge_child_provenance(parent_carrier[parent_id], record)
                continue

            pieces: List[str] = []
            if has_parent:
                pieces.append(parent_cache[parent_id])

            # Neighbor window within the same document.
            if self.chunk_read and has_doc and has_chunk and record.get(CHUNK_ID_KEY) is not None:
                doc_id = record.get(DOC_ID_KEY)
                chunk_id = record.get(CHUNK_ID_KEY)
                lo, hi = chunk_id - self.chunk_read, chunk_id + self.chunk_read
                window = metadata[
                    (metadata[DOC_ID_KEY] == doc_id) & (metadata[CHUNK_ID_KEY] >= lo) & (metadata[CHUNK_ID_KEY] <= hi)
                ].sort_values(CHUNK_ID_KEY)
                neighbor_text = "\n".join(str(t) for t in window[DOC_TEXT_KEY].tolist())
                record["neighbor_chunk_ids"] = [int(c) for c in window[CHUNK_ID_KEY].tolist()]
                if neighbor_text:
                    pieces.append(neighbor_text)

            if pieces:
                record[DOC_TEXT_KEY] = "\n\n".join(pieces)

            # Track child chunk ids merged into this expanded record (starts with
            # its own), so folded siblings can be recorded without losing them.
            own_chunk = record.get(CHUNK_ID_KEY)
            record["expanded_child_chunk_ids"] = [own_chunk] if own_chunk is not None else []
            if has_parent:
                parent_carrier[parent_id] = record
            expanded.append(record)

        return expanded

    @staticmethod
    def _merge_child_provenance(carrier: Dict[str, Any], sibling: Dict[str, Any]) -> None:
        """Fold a dropped sibling's provenance into the surviving parent carrier.

        Preserves which subqueries/signals surfaced the sibling and records its
        child chunk id, so collapsing duplicate parent contexts never loses child
        provenance.
        """
        for field_name in ("retrieval_queries", "source_signals"):
            merged = carrier.setdefault(field_name, [])
            for value in sibling.get(field_name, []) or []:
                if value not in merged:
                    merged.append(value)
        sibling_chunk = sibling.get(CHUNK_ID_KEY)
        if sibling_chunk is not None:
            child_ids = carrier.setdefault("expanded_child_chunk_ids", [])
            if sibling_chunk not in child_ids:
                child_ids.append(sibling_chunk)
