"""Embedding quantization utilities.

Supports 4-bit and ternary quantization modes for reducing memory
footprint of embedding vectors at the cost of minor precision loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class QuantizationResult:
    """Container for quantized embedding data and reconstruction parameters.

    Attributes:
        codes: Packed quantized codes.
        codebook: Optional codebook for vector quantization.
        scale: Per-dimension scale factors for dequantization.
        min_val: Per-dimension minimum values for dequantization.
        mode: Quantization mode ("4bit", "ternary", "none").
        original_shape: Shape of the original float32 matrix.
    """

    codes: np.ndarray
    codebook: list[np.ndarray] | None = None
    scale: np.ndarray | None = None
    min_val: np.ndarray | None = None
    mode: str = "none"
    original_shape: tuple[int, ...] = (0,)


def quantize_4bit(
    vectors: np.ndarray,
) -> QuantizationResult:
    """Quantize float32 vectors to packed 4-bit representations.

    Each dimension is independently scaled to a 4-bit range (0-15),
    then two values are packed per byte.

    Args:
        vectors: Float32 matrix of shape (n, d).

    Returns:
        QuantizationResult with packed uint8 codes.
    """
    n, d = vectors.shape

    min_val = vectors.min(axis=0)
    scale = (vectors.max(axis=0) - min_val) / 15.0
    scale = np.where(scale == 0, 1.0, scale)
    quantized = np.round((vectors - min_val) / scale).astype(np.uint8)
    quantized = np.clip(quantized, 0, 15)

    packed = np.zeros((n, (d + 1) // 2), dtype=np.uint8)
    for i in range(d):
        col_idx = i // 2
        if i % 2 == 0:
            packed[:, col_idx] = (quantized[:, i] & 0x0F).astype(np.uint8)
        else:
            packed[:, col_idx] |= (quantized[:, i] << 4).astype(np.uint8)

    return QuantizationResult(
        codes=packed,
        codebook=None,
        scale=scale.astype(np.float32),
        min_val=min_val.astype(np.float32),
        mode="4bit",
        original_shape=vectors.shape,
    )


def dequantize_4bit(result: QuantizationResult) -> np.ndarray:
    """Dequantize packed 4-bit codes back to float32.

    Args:
        result: QuantizationResult from quantize_4bit().

    Returns:
        Reconstructed float32 matrix.
    """
    n, packed_d = result.codes.shape
    d = result.original_shape[1]
    if result.scale is None or result.min_val is None:
        raise ValueError("Invalid 4bit quantization result: missing scale or min_val")
    unpacked = np.zeros((n, d), dtype=np.uint8)
    for i in range(d):
        col_idx = i // 2
        if i % 2 == 0:
            unpacked[:, i] = result.codes[:, col_idx] & 0x0F
        else:
            unpacked[:, i] = (result.codes[:, col_idx] >> 4) & 0x0F

    dequantized = unpacked.astype(np.float32) * result.scale + result.min_val
    return dequantized


def quantize_ternary(vectors: np.ndarray, delta: float = 0.7) -> QuantizationResult:
    """Quantize vectors to ternary ( -1, 0, +1 ) values.

    Values above delta * max_abs become +1, below -delta * max_abs become -1.

    Args:
        vectors: Float32 matrix of shape (n, d).
        delta: Threshold ratio. Default 0.7.

    Returns:
        QuantizationResult with int8 codes in {-1, 0, 1}.
    """
    max_abs = np.max(np.abs(vectors), axis=0, keepdims=True)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)
    threshold = max_abs * delta
    codes = np.zeros(vectors.shape, dtype=np.int8)
    codes[vectors > threshold] = 1
    codes[vectors < -threshold] = -1
    return QuantizationResult(
        codes=codes,
        scale=max_abs.astype(np.float32).squeeze(0),
        min_val=None,
        mode="ternary",
        original_shape=vectors.shape,
    )


def dequantize_ternary(result: QuantizationResult) -> np.ndarray:
    """Dequantize ternary codes back to float32.

    Args:
        result: QuantizationResult from quantize_ternary().

    Returns:
        Reconstructed float32 matrix.
    """
    scale = result.scale
    if scale is None:
        raise ValueError("Invalid ternary quantization result: missing scale")
    if scale.ndim == 1:
        scale = scale.reshape(1, -1)
    return result.codes.astype(np.float32) * scale * 0.7


def quantize(
    vectors: np.ndarray,
    mode: str = "none",
) -> QuantizationResult:
    """Quantize vectors using the specified mode.

    Args:
        vectors: Float32 matrix of shape (n, d).
        mode: One of "4bit", "ternary", or "none" (passthrough).

    Returns:
        QuantizationResult.
    """
    if mode == "4bit":
        return quantize_4bit(vectors)
    if mode == "ternary":
        return quantize_ternary(vectors)
    return QuantizationResult(
        codes=vectors.astype(np.float32),
        codebook=None,
        scale=None,
        min_val=None,
        mode="none",
        original_shape=vectors.shape,
    )


def dequantize(result: QuantizationResult) -> np.ndarray:
    """Dequantize vectors back to float32.

    Dispatches to the appropriate dequantizer based on result.mode.

    Args:
        result: QuantizationResult from quantize().

    Returns:
        Reconstructed float32 matrix.
    """
    if result.mode == "4bit":
        return dequantize_4bit(result)
    if result.mode == "ternary":
        return dequantize_ternary(result)
    return result.codes.astype(np.float32)


def memory_bytes(result: QuantizationResult) -> int:
    """Return the memory footprint of the quantized data in bytes.

    Args:
        result: QuantizationResult.

    Returns:
        Number of bytes used by the quantized codes.
    """
    return result.codes.nbytes


def compression_ratio(result: QuantizationResult) -> float:
    """Compute the compression ratio achieved by quantization.

    Args:
        result: QuantizationResult.

    Returns:
        Ratio of original float32 size to quantized size (higher = better).
    """
    original_bytes = int(np.prod(result.original_shape)) * 4
    return original_bytes / max(result.codes.nbytes, 1)
