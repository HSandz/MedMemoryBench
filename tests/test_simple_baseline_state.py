"""Offline regression coverage for simple baseline state transitions."""

from __future__ import annotations

import numpy as np

from benchmarks.snapshot_artifacts import (
    load_memory_state_embedding_artifacts,
    publish_memory_state_embedding_artifacts,
)
from methods.bm25_rag import BM25RAGAgent
from methods.embedding_rag import EmbeddingRAGAgent
from methods.long_context import LongContextAgent


class _Tokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class _Embeddings:
    """Small deterministic embedding implementation for FAISS state tests."""

    @staticmethod
    def _vector(text):
        return [float(len(text)), float(sum(map(ord, text)) % 97)]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def __call__(self, text):
        return self._vector(text)


def _long_context():
    agent = LongContextAgent.__new__(LongContextAgent)
    agent._tokenizer = _Tokenizer()
    agent.max_context_tokens = 3
    agent.truncation_strategy = "oldest_first"
    agent._memory_chunks = []
    agent._context = ""
    agent._is_initialized = False
    agent._context_id = None
    return agent


def _embedding_rag():
    agent = EmbeddingRAGAgent.__new__(EmbeddingRAGAgent)
    agent._tokenizer = _Tokenizer()
    agent.chunk_size = 100
    agent.chunk_overlap = 0
    agent.top_k = 5
    agent._memory_chunks = []
    agent._chunks = []
    agent._vectorstore = None
    agent._embedding_model_instance = _Embeddings()
    agent._get_embedding_model = lambda: agent._embedding_model_instance
    agent.embedding_model = "test_embed"
    agent.embedding_provider = "test_provider"
    agent._is_initialized = False
    agent._context_id = None
    return agent


def _bm25_rag():
    agent = BM25RAGAgent.__new__(BM25RAGAgent)
    agent._tokenizer = _Tokenizer()
    agent.language = "en"
    agent.chunk_size = 100
    agent.chunk_overlap = 0
    agent.k1 = 1.5
    agent.b = 0.75
    agent.top_k = 5
    agent._memory_chunks = []
    agent._chunks = []
    agent._tokenized_corpus = []
    agent._bm25 = None
    agent._is_initialized = False
    agent._context_id = None
    return agent


def test_long_context_retains_oversized_newest_session_and_round_trips():
    agent = _long_context()
    agent.memorize("one two three four")

    assert agent.context == "two three four"
    assert agent.get_memory_chunks() == ["two three four"]

    restored = _long_context()
    restored.import_memory_state(agent.export_memory_state(context_id="unit-1"))
    assert restored.context == agent.context
    assert restored._context_id == "unit-1"


def test_embedding_rag_adds_only_new_documents_and_restores_exact_vectors(tmp_path):
    agent = _embedding_rag()
    agent.memorize("alpha")
    first_index = agent._vectorstore.index
    agent.memorize("beta")

    assert agent._vectorstore.index is first_index
    assert agent._vectorstore.index.ntotal == 2
    state = agent.export_memory_state(context_id="unit-2")
    snapshot_path = tmp_path / "embedding.json"
    publish_memory_state_embedding_artifacts(state, snapshot_path)
    load_memory_state_embedding_artifacts(state, snapshot_path)

    restored = _embedding_rag()
    restored.import_memory_state(state)
    assert restored._chunks == ["alpha", "beta"]
    assert restored._vectorstore.index.ntotal == 2
    assert restored._retrieve("beta") == agent._retrieve("beta")
    np.testing.assert_allclose(
        [restored._vectorstore.index.reconstruct(i) for i in range(2)],
        [agent._vectorstore.index.reconstruct(i) for i in range(2)],
    )


def test_bm25_rag_does_not_retokenize_prior_chunks_and_round_trips():
    agent = _bm25_rag()
    calls = []
    original_tokenize = agent._tokenize

    def tokenize(text):
        calls.append(text)
        return original_tokenize(text)

    agent._tokenize = tokenize
    agent.memorize("alpha beta")
    agent.memorize("gamma delta")
    agent.memorize("epsilon zeta")

    assert calls == ["alpha beta", "gamma delta", "epsilon zeta"]
    state = agent.export_memory_state(context_id="unit-3")

    restored = _bm25_rag()
    restored.import_memory_state(state)
    assert restored._tokenized_corpus == agent._tokenized_corpus
    assert restored._retrieve("gamma") == ["gamma delta"]


def test_bm25_and_embedding_rag_use_raw_question_when_provided():
    bm25_agent = _bm25_rag()
    bm25_agent.memorize("the patient took aspirin daily")

    retrieval_queries = []
    bm25_agent._retrieve = lambda q: retrieval_queries.append(q) or ["the patient took aspirin daily"]
    bm25_agent._truncate_to_tokens = lambda text, max_tok: text
    bm25_agent.count_tokens = lambda text: len(text.split())
    bm25_agent.max_tokens = 50
    bm25_agent.max_context_tokens = 500
    bm25_agent.max_question_tokens = 200

    full_prompt = "Few-shot example: dancing\nQuestion: What medication did the patient take?"
    raw_q = "What medication did the patient take?"

    bm25_agent.prepare_batch_query(full_prompt, raw_question=raw_q)
    assert retrieval_queries == [raw_q]

    emb_agent = _embedding_rag()
    emb_agent.memorize("the patient took aspirin daily")
    emb_retrievals = []
    emb_agent._retrieve = lambda q: emb_retrievals.append(q) or ["the patient took aspirin daily"]
    emb_agent._truncate_to_tokens = lambda text, max_tok: text
    emb_agent.count_tokens = lambda text: len(text.split())
    emb_agent.max_tokens = 50
    emb_agent.max_context_tokens = 500
    emb_agent.max_question_tokens = 200

    emb_agent.prepare_batch_query(full_prompt, raw_question=raw_q)
    assert emb_retrievals == [raw_q]


def test_graph_rag_query_engine_extracts_locomo_question():
    from methods.graph_rag import QueryEngine
    engine = QueryEngine.__new__(QueryEngine)

    prompt = "Instruction: Answer based on context.\n\nQuestion: What date did they meet?\n\nAnswer:"
    assert engine._extract_retrieval_query(prompt) == "What date did they meet?"


def test_amem_supports_string_context_id():
    import unittest.mock as mock
    from methods.amem_agent import AMemAgent
    agent = AMemAgent.__new__(AMemAgent)
    agent._context_id = None
    agent._memory_chunks = []
    agent._is_initialized = True
    agent.MEMORY_STATE_VERSION = 1
    agent._amem_systems = {}

    agent.set_context_id("conv-30")
    assert agent._get_context_id() == "conv-30"

    mock_sys = mock.MagicMock()
    mock_sys.evo_cnt = 0
    mock_sys.evo_threshold = 10
    mock_sys.max_context_chars = 1000
    mock_sys.memories = {}
    mock_sys.retriever.corpus = []
    mock_sys.retriever.document_ids = []
    mock_sys.retriever.embeddings = None
    agent._amem_systems["conv-30"] = mock_sys
    agent._memory_state_config = lambda: {}

    state = agent.export_memory_state(context_id="conv-30")
    assert state["context_id"] == "conv-30"

