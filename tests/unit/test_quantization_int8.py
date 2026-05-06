import numpy as np

from reranker.quantization import (
    compression_ratio,
    dequantize,
    dequantize_int8,
    memory_bytes,
    quantize,
    quantize_int8,
)


class TestQuantizeInt8:
    def test_roundtrip(self) -> None:
        original = np.array([[1.0, -2.0, 3.0, -4.0], [0.5, 0.0, -0.5, 1.0]], dtype=np.float32)
        result = quantize_int8(original)
        recovered = dequantize_int8(result)
        np.testing.assert_allclose(original, recovered, atol=0.05)

    def test_mode(self) -> None:
        vectors = np.random.randn(3, 8).astype(np.float32)
        result = quantize_int8(vectors)
        assert result.mode == "int8"
        assert result.codes.dtype == np.int8

    def test_compression_ratio(self) -> None:
        vectors = np.random.randn(100, 256).astype(np.float32)
        result = quantize_int8(vectors)
        ratio = compression_ratio(result)
        assert 3.5 < ratio < 4.5

    def test_dispatch_via_quantize(self) -> None:
        vectors = np.random.randn(4, 8).astype(np.float32)
        result = quantize(vectors, mode="int8")
        recovered = dequantize(result)
        assert recovered.shape == vectors.shape
        assert recovered.dtype == np.float32

    def test_memory_smaller(self) -> None:
        vectors = np.random.randn(100, 256).astype(np.float32)
        result = quantize_int8(vectors)
        assert memory_bytes(result) < vectors.nbytes

    def test_zero_vectors(self) -> None:
        vectors = np.zeros((3, 8), dtype=np.float32)
        result = quantize_int8(vectors)
        recovered = dequantize_int8(result)
        np.testing.assert_allclose(recovered, vectors, atol=0.01)
