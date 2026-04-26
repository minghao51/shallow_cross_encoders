"""Tests for FlashRank adapter."""



def test_flashrank_wrapper_init():
    """Test FlashRankWrapper initialization."""
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    # Test with default model
    wrapper = FlashRankWrapper()
    assert wrapper.model_name == "ms-marco-TinyBERT-L-2-v2"

    # Test with custom model
    wrapper = FlashRankWrapper("ms-marco-MiniLM-L-12-v2")
    assert wrapper.model_name == "ms-marco-MiniLM-L-12-v2"


def test_flashrank_wrapper_rerank_empty():
    """Test FlashRankWrapper with empty docs."""
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    wrapper = FlashRankWrapper()
    result = wrapper.rerank("test query", [])
    assert result == []


def test_flashrank_wrapper_basic():
    """Test FlashRankWrapper basic functionality exists."""
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    wrapper = FlashRankWrapper()
    # Just verify the class and methods exist without importing flashrank
    assert hasattr(wrapper, "_load_ranker")
    assert hasattr(wrapper, "rerank")
    assert callable(wrapper.rerank)
