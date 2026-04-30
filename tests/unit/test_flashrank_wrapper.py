"""Tests for FlashRank adapter."""


def test_flashrank_wrapper_init():
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    wrapper = FlashRankWrapper()
    assert wrapper.model_name == "ms-marco-TinyBERT-L-2-v2"

    wrapper = FlashRankWrapper("ms-marco-MiniLM-L-12-v2")
    assert wrapper.model_name == "ms-marco-MiniLM-L-12-v2"


def test_flashrank_wrapper_rerank_empty():
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    wrapper = FlashRankWrapper()
    result = wrapper.rerank("test query", [])
    assert result == []


def test_flashrank_wrapper_return_type():
    from reranker.adapters.flashrank_wrapper import FlashRankWrapper

    wrapper = FlashRankWrapper()
    assert hasattr(wrapper, "_load_ranker")
    assert hasattr(wrapper, "rerank")
    assert callable(wrapper.rerank)
