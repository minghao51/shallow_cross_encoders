import numpy as np

from reranker.heuristics.keyword import KeywordMatchAdapter
from reranker.heuristics.lsh import (
    LSHAdapter,
    _char_ngrams,
    _jaccard_from_signatures,
    _minhash_signature,
)


class TestLSHAdapter:
    def test_lsh_basic(self) -> None:
        adapter = LSHAdapter(ngram_size=3, num_perm=128)
        result = adapter.compute("hello world", "hello there world")

        assert "lsh_score" in result
        assert "lsh_jaccard" in result
        assert 0.0 <= result["lsh_score"] <= 1.0
        assert 0.0 <= result["lsh_jaccard"] <= 1.0

    def test_lsh_identical_texts(self) -> None:
        adapter = LSHAdapter(ngram_size=3, num_perm=128)
        text = "the quick brown fox"
        result = adapter.compute(text, text)

        assert result["lsh_score"] == 1.0
        assert result["lsh_jaccard"] == 1.0

    def test_lsh_completely_different(self) -> None:
        adapter = LSHAdapter(ngram_size=3, num_perm=128)
        result = adapter.compute("hello world", "xyz abc def")

        assert result["lsh_score"] <= 1.0
        assert result["lsh_jaccard"] <= 1.0

    def test_lsh_empty_query(self) -> None:
        adapter = LSHAdapter()
        result = adapter.compute("", "some document")

        assert result["lsh_score"] == 0.0
        assert result["lsh_jaccard"] == 0.0

    def test_lsh_empty_doc(self) -> None:
        adapter = LSHAdapter()
        result = adapter.compute("some query", "")

        assert result["lsh_score"] == 0.0
        assert result["lsh_jaccard"] == 0.0

    def test_lsh_very_short_texts(self) -> None:
        adapter = LSHAdapter(ngram_size=3, num_perm=64)
        result = adapter.compute("ab", "abc")

        assert "lsh_score" in result
        assert "lsh_jaccard" in result

    def test_lsh_uses_settings_defaults(self) -> None:
        adapter = LSHAdapter()
        assert adapter.ngram_size == 3
        assert adapter.num_perm == 128


class TestKeywordMatchAdapter:
    def test_keyword_basic(self) -> None:
        adapter = KeywordMatchAdapter()
        result = adapter.compute("hello world", "hello world, how are you?")

        assert "keyword_hit_rate" in result
        assert result["keyword_hit_rate"] == 1.0

    def test_keyword_partial_hit(self) -> None:
        adapter = KeywordMatchAdapter()
        result = adapter.compute("hello world", "hello everyone!")

        assert 0.0 < result["keyword_hit_rate"] < 1.0

    def test_keyword_no_hit(self) -> None:
        adapter = KeywordMatchAdapter()
        result = adapter.compute("hello world", "greetings planet")

        assert result["keyword_hit_rate"] == 0.0

    def test_keyword_empty_query(self) -> None:
        adapter = KeywordMatchAdapter()
        result = adapter.compute("", "some document")

        assert result["keyword_hit_rate"] == 0.0

    def test_keyword_case_insensitive(self) -> None:
        adapter = KeywordMatchAdapter()
        result = adapter.compute("Hello WORLD", "hello world!")

        assert result["keyword_hit_rate"] == 1.0

    def test_keyword_custom_tokenize(self) -> None:
        def tokenize(text: str) -> list[str]:
            return text.split("-")

        adapter = KeywordMatchAdapter(tokenize_fn=tokenize)
        result = adapter.compute("one-two three-four", "one two three four")

        assert result["keyword_hit_rate"] == 1.0


class TestLSHHelpers:
    def test_char_ngrams_basic(self) -> None:
        ngrams = _char_ngrams("hello", n=3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_char_ngrams_short_string(self) -> None:
        ngrams = _char_ngrams("ab", n=3)
        assert ngrams == {"ab"}

    def test_char_ngrams_empty_string(self) -> None:
        ngrams = _char_ngrams("", n=3)
        assert ngrams == set()

    def test_minhash_signature(self) -> None:
        ngrams = {"abc", "def"}
        sig = _minhash_signature(ngrams, num_perm=128)

        assert sig.shape == (128,)
        assert sig.dtype == np.int64
        assert np.all(sig >= 0)

    def test_minhash_empty_ngrams(self) -> None:
        sig = _minhash_signature(set(), num_perm=64)
        assert sig.shape == (64,)
        assert np.all(sig == 0)

    def test_jaccard_from_signatures(self) -> None:
        sig = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = _jaccard_from_signatures(sig, sig)
        assert result == 1.0

    def test_jaccard_from_signatures_no_overlap(self) -> None:
        sig1 = np.array([1, 2, 3], dtype=np.int64)
        sig2 = np.array([4, 5, 6], dtype=np.int64)
        result = _jaccard_from_signatures(sig1, sig2)
        assert result == 0.0

    def test_jaccard_from_signatures_partial(self) -> None:
        sig1 = np.array([1, 2, 3], dtype=np.int64)
        sig2 = np.array([1, 5, 3], dtype=np.int64)
        result = _jaccard_from_signatures(sig1, sig2)
        assert 0.0 < result < 1.0

    def test_jaccard_from_signatures_empty(self) -> None:
        result = _jaccard_from_signatures(np.array([]), np.array([]))
        assert result == 0.0
