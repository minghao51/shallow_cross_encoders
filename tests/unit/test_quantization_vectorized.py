import numpy as np

from reranker.quantization import dequantize_4bit, quantize_4bit


class TestQuantize4BitVectorized:
    def test_pack_unpack_roundtrip(self) -> None:
        original = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        result = quantize_4bit(original)
        recovered = dequantize_4bit(result)
        np.testing.assert_allclose(original, recovered, atol=0.5)

    def test_roundtrip_large_matrix(self) -> None:
        original = np.random.randn(50, 256).astype(np.float32)
        result = quantize_4bit(original)
        recovered = dequantize_4bit(result)
        np.testing.assert_allclose(original, recovered, atol=0.3)
        assert recovered.shape == original.shape

    def test_packed_shape(self) -> None:
        vectors = np.random.randn(10, 16).astype(np.float32)
        result = quantize_4bit(vectors)
        assert result.codes.shape == (10, 8)

    def test_odd_dimension(self) -> None:
        vectors = np.random.randn(5, 7).astype(np.float32)
        result = quantize_4bit(vectors)
        assert result.codes.shape == (5, 4)
        recovered = dequantize_4bit(result)
        assert recovered.shape == (5, 7)

    def test_no_python_loop(self) -> None:
        import inspect

        pack_source = inspect.getsource(quantize_4bit)
        unpack_source = inspect.getsource(dequantize_4bit)
        assert "for i in range(d)" not in pack_source
        assert "for i in range(d)" not in unpack_source
