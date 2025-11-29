"""
Unit tests for Phase 2: Advanced Retrieval components.

Tests BM25 search, reranking, hybrid search, and response synthesis.
"""

import pytest

from src.rag.retrieval import BM25Search, Reranker
from src.rag.generation import ResponseSynthesizer


class TestBM25Search:
    """Test cases for BM25 keyword search."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "1",
                "text": "Employees receive 15 days of vacation per year.",
                "metadata": {"filename": "vacation_policy.md"}
            },
            {
                "chunk_id": "2",
                "text": "Remote work requires manager approval.",
                "metadata": {"filename": "remote_work.md"}
            },
            {
                "chunk_id": "3",
                "text": "Senior employees get 25 vacation days annually.",
                "metadata": {"filename": "vacation_policy.md"}
            },
        ]
    
    def test_index_and_search(self, sample_chunks):
        """Test basic indexing and searching."""
        bm25 = BM25Search()
        bm25.index_chunks(sample_chunks)
        
        results = bm25.search("vacation days", top_k=2)
        
        assert len(results) <= 2
        assert all("score" in r for r in results)
        assert all("chunk_id" in r for r in results)
    
    def test_empty_index_raises_error(self):
        """Test that searching empty index raises error."""
        bm25 = BM25Search()
        
        with pytest.raises(ValueError, match="No chunks indexed"):
            bm25.search("test query")
    
    def test_corpus_stats(self, sample_chunks):
        """Test corpus statistics."""
        bm25 = BM25Search()
        bm25.index_chunks(sample_chunks)
        
        stats = bm25.get_corpus_stats()
        
        assert stats["document_count"] == 3
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_doc"] > 0
    
    def test_score_threshold(self, sample_chunks):
        """Test score threshold filtering."""
        bm25 = BM25Search()
        bm25.index_chunks(sample_chunks)
        
        # High threshold should return fewer results
        results_high = bm25.search("vacation", top_k=10, score_threshold=10.0)
        results_low = bm25.search("vacation", top_k=10, score_threshold=0.1)
        
        assert len(results_high) <= len(results_low)


class TestReranker:
    """Test cases for cross-encoder reranker."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample search results for reranking."""
        return [
            {
                "chunk_id": "1",
                "text": "Python is a programming language.",
                "score": 0.8,
            },
            {
                "chunk_id": "2",
                "text": "Python is also a type of snake.",
                "score": 0.7,
            },
        ]
    
    @pytest.mark.slow
    def test_rerank_basic(self, sample_results):
        """Test basic reranking functionality."""
        reranker = Reranker()
        
        reranked = reranker.rerank(
            query="programming language",
            results=sample_results
        )
        
        assert len(reranked) == 2
        assert all("score" in r for r in reranked)
        assert all("rank" in r for r in reranked)
        # First result should be about programming
        assert "programming" in reranked[0]["text"]
    
    def test_rerank_empty_results(self):
        """Test reranking with empty results."""
        reranker = Reranker()
        
        reranked = reranker.rerank("test query", [])
        
        assert len(reranked) == 0
    
    def test_rerank_top_k(self, sample_results):
        """Test top-k filtering in reranking."""
        reranker = Reranker()
        
        reranked = reranker.rerank(
            query="python",
            results=sample_results,
            top_k=1
        )
        
        assert len(reranked) == 1
    
    @pytest.mark.slow
    def test_get_model_info(self):
        """Test getting model information."""
        reranker = Reranker()
        
        info = reranker.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "max_length" in info


class TestResponseSynthesizer:
    """Test cases for response synthesis."""
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results."""
        return [
            {
                "chunk_id": "1",
                "text": "New employees receive 15 vacation days per year.",
                "score": 0.9,
                "metadata": {"filename": "vacation_policy.md"}
            },
            {
                "chunk_id": "2",
                "text": "Employees must request vacation 2 weeks in advance.",
                "score": 0.8,
                "metadata": {"filename": "vacation_policy.md"}
            },
        ]
    
    def test_synthesize_with_results(self, sample_search_results):
        """Test synthesis with good results."""
        synthesizer = ResponseSynthesizer(min_confidence=0.5)
        
        response = synthesizer.synthesize(
            query="vacation policy",
            search_results=sample_search_results
        )
        
        assert response.has_answer
        assert response.confidence > 0.5
        assert len(response.sources) > 0
        assert "vacation" in response.answer.lower()
    
    def test_synthesize_empty_results(self):
        """Test synthesis with no results."""
        synthesizer = ResponseSynthesizer()
        
        response = synthesizer.synthesize(
            query="test query",
            search_results=[]
        )
        
        assert not response.has_answer
        assert response.confidence == 0.0
        assert len(response.sources) == 0
    
    def test_synthesize_low_confidence(self, sample_search_results):
        """Test synthesis with low confidence results."""
        # Lower the scores
        low_score_results = [
            {**r, "score": 0.3} for r in sample_search_results
        ]
        
        synthesizer = ResponseSynthesizer(min_confidence=0.5)
        
        response = synthesizer.synthesize(
            query="test query",
            search_results=low_score_results
        )
        
        assert not response.has_answer
        assert response.confidence < 0.5
    
    def test_extract_citations(self, sample_search_results):
        """Test citation extraction."""
        synthesizer = ResponseSynthesizer()
        
        response = synthesizer.synthesize(
            query="vacation",
            search_results=sample_search_results
        )
        
        citations = synthesizer.extract_citations(response)
        
        assert len(citations) == len(response.sources)
        assert all("id" in c for c in citations)
        assert all("source" in c for c in citations)
    
    def test_max_sources_limit(self):
        """Test that max_sources is respected."""
        many_results = [
            {
                "chunk_id": str(i),
                "text": f"Result {i}",
                "score": 0.8,
                "metadata": {"filename": "test.md"}
            }
            for i in range(10)
        ]
        
        synthesizer = ResponseSynthesizer(max_sources=3)
        
        response = synthesizer.synthesize(
            query="test",
            search_results=many_results
        )
        
        assert len(response.sources) == 3