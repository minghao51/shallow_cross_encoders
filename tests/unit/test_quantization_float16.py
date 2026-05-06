import numpy as np

from reranker.quantization import (
    compression_ratio,
    dequantize,
    dequantize_float16,
    memory_bytes,
    quantize,
    quantize_float16,
)


class TestQuantizeFloat16:
    def test_roundtrip(self) -> None:
        original = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, 1.0]], dtype=np.float32)
        result = quantize_float16(original)
        recovered = dequantize_float16(result)
        np.testing.assert_allclose(original, recovered, rtol=1e-3)

    def test_mode(self) -> None:
        vectors = np.random.randn(3, 8).astype(np.float32)
        result = quantize_float16(vectors)
        assert result.mode == "float16"
        assert result.codes.dtype == np.float16

    def test_compression_ratio(self) -> None:
        vectors = np.random.randn(100, 256).astype(np.float32)
        result = quantize_float16(vectors)
        ratio = compression_ratio(result)
        assert 1.8 < ratio < 2.2

    def test_dispatch_via_quantize(self) -> None:
        vectors = np.random.randn(4, 8).astype(np.float32)
        result = quantize(vectors, mode="float16")
        recovered = dequantize(result)
        assert recovered.shape == vectors.shape
        assert recovered.dtype == np.float32

    def test_memory_smaller(self) -> None:
        vectors = np.random.randn(100, 256).astype(np.float32)
        result = quantize_float16(vectors)
        assert memory_bytes(result) < vectors.nbytes

    def test_preserves_shape(self) -> None:
        vectors = np.random.randn(10, 128).astype(np.float32)
        result = quantize_float16(vectors)
        assert result.original_shape == (10, 128)
        assert result.codes.shape == (10, 128)
