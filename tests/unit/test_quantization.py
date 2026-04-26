import numpy as np
import pytest

from reranker.quantization import (
    QuantizationResult,
    compression_ratio,
    dequantize,
    dequantize_4bit,
    dequantize_ternary,
    memory_bytes,
    quantize,
    quantize_4bit,
    quantize_ternary,
)


class TestQuantize4Bit:
    def test_quantize_basic(self) -> None:
        vectors = np.array([[0.0, 5.0, 10.0, 15.0]], dtype=np.float32)
        result = quantize_4bit(vectors)

        assert result.mode == "4bit"
        assert result.original_shape == (1, 4)
        assert result.codes.shape == (1, 2)
        assert result.scale is not None
        assert result.min_val is not None

    def test_quantize_and_dequantize_recovers_approximate_values(self) -> None:
        original = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        result = quantize_4bit(original)
        recovered = dequantize_4bit(result)

        np.testing.assert_allclose(original, recovered, atol=0.5)

    def test_quantize_preserves_shape_info(self) -> None:
        vectors = np.random.randn(10, 16).astype(np.float32)
        result = quantize_4bit(vectors)

        assert result.original_shape == (10, 16)
        assert result.codes.shape == (10, 8)


class TestQuantizeTernary:
    def test_quantize_ternary_basic(self) -> None:
        vectors = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32)
        result = quantize_ternary(vectors)

        assert result.mode == "ternary"
        assert result.codes.dtype == np.int8
        unique_codes = np.unique(result.codes)
        assert all(c in (-1, 0, 1) for c in unique_codes)

    def test_dequantize_ternary(self) -> None:
        original = np.array([[3.0, -3.0, 1.0, -1.0]], dtype=np.float32)
        result = quantize_ternary(original)
        recovered = dequantize_ternary(result)

        np.testing.assert_allclose(recovered, original * 0.7, rtol=0.15)

    def test_ternary_scale_shape(self) -> None:
        vectors = np.random.randn(5, 8).astype(np.float32)
        result = quantize_ternary(vectors)

        assert result.scale is not None
        assert result.scale.ndim == 1 or result.scale.ndim == 2


class TestQuantize:
    @pytest.mark.parametrize("mode", ["4bit", "ternary", "none"])
    def test_quantize_dispatch(self, mode: str) -> None:
        vectors = np.random.randn(3, 8).astype(np.float32)
        result = quantize(vectors, mode=mode)

        assert isinstance(result, QuantizationResult)
        assert result.mode == mode

    def test_quantize_unknown_mode_returns_original(self) -> None:
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = quantize(vectors, mode="unknown")

        assert result.mode == "none"
        np.testing.assert_allclose(result.codes, vectors)


class TestDequantize:
    @pytest.mark.parametrize("mode", ["4bit", "ternary", "none"])
    def test_dequantize_roundtrip(self, mode: str) -> None:
        vectors = np.random.randn(4, 8).astype(np.float32) * 10
        quantized = quantize(vectors, mode=mode)
        recovered = dequantize(quantized)

        assert recovered.dtype == np.float32
        if mode == "none":
            np.testing.assert_allclose(vectors, recovered)
        else:
            assert recovered.shape == vectors.shape


class TestMemoryAndCompression:
    def test_memory_bytes_4bit(self) -> None:
        vectors = np.random.randn(100, 64).astype(np.float32)
        result = quantize(vectors, mode="4bit")

        assert memory_bytes(result) < vectors.nbytes

    def test_memory_bytes_ternary(self) -> None:
        vectors = np.random.randn(100, 64).astype(np.float32)
        result = quantize(vectors, mode="ternary")

        assert memory_bytes(result) < vectors.nbytes

    def test_compression_ratio_4bit(self) -> None:
        vectors = np.random.randn(50, 128).astype(np.float32)
        result = quantize(vectors, mode="4bit")

        ratio = compression_ratio(result)
        assert ratio > 7.0

    def test_compression_ratio_ternary(self) -> None:
        vectors = np.random.randn(50, 128).astype(np.float32)
        result = quantize(vectors, mode="ternary")

        ratio = compression_ratio(result)
        assert 3.0 < ratio < 10.0

    def test_compression_ratio_none(self) -> None:
        vectors = np.random.randn(10, 10).astype(np.float32)
        result = quantize(vectors, mode="none")

        ratio = compression_ratio(result)
        assert ratio == 1.0


class TestQuantizationResult:
    def test_result_dataclass_fields(self) -> None:
        result = QuantizationResult(
            codes=np.array([[1, 2]]),
            codebook=None,
            scale=np.array([1.0]),
            min_val=np.array([0.0]),
            mode="4bit",
            original_shape=(1, 2),
        )

        assert result.codes.shape == (1, 2)
        assert result.mode == "4bit"
        assert result.original_shape == (1, 2)

    def test_result_with_codebook(self) -> None:
        codebook = [np.array([0.0, 1.0]), np.array([2.0, 3.0])]
        result = QuantizationResult(
            codes=np.array([[0, 1]]),
            codebook=codebook,
            scale=None,
            min_val=None,
            mode="4bit",
            original_shape=(1, 2),
        )

        assert result.codebook is not None
        assert len(result.codebook) == 2