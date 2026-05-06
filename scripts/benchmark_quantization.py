"""Benchmark compression ratio vs quality tradeoff for all quantization modes."""

from __future__ import annotations

import time

import numpy as np

from reranker.quantization import (
    compression_ratio,
    dequantize,
    memory_bytes,
    quantize,
)


def estimate_cache_hits(texts: list[str]) -> dict[str, int]:
    unique_count = len(set(texts))
    total = len(texts)
    return {
        "total_texts": total,
        "unique_texts": unique_count,
        "estimated_cache_hits": max(total - unique_count, 0),
    }


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def cosine_similarity_preservation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig_norms = np.linalg.norm(original, axis=1, keepdims=True)
    recon_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
    orig_norms = np.where(orig_norms == 0, 1.0, orig_norms)
    recon_norms = np.where(recon_norms == 0, 1.0, recon_norms)
    orig_norm = original / orig_norms
    recon_norm = reconstructed / recon_norms
    similarities = np.sum(orig_norm * recon_norm, axis=1)
    return float(np.mean(similarities))


def benchmark_mode(vectors: np.ndarray, mode: str) -> dict:
    start = time.perf_counter()
    result = quantize(vectors, mode=mode)
    encode_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    recovered = dequantize(result)
    decode_ms = (time.perf_counter() - start) * 1000

    error = rmse(vectors, recovered)
    cosine_sim = cosine_similarity_preservation(vectors, recovered)
    ratio = compression_ratio(result)
    mem = memory_bytes(result)

    return {
        "mode": mode,
        "compression_ratio": round(ratio, 2),
        "memory_bytes": mem,
        "rmse": round(error, 6),
        "cosine_similarity": round(cosine_sim, 6),
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
    }


def main() -> None:
    np.random.seed(42)
    shapes = [(100, 64), (100, 256), (1000, 256)]
    modes = ["4bit", "int8", "float16", "ternary", "none"]
    sample_texts = (
        ["python ranking"] * 40
        + ["semantic search"] * 30
        + [f"unique document {i}" for i in range(30)]
    )
    cache_stats = estimate_cache_hits(sample_texts)
    print("Cache Reuse Estimate:")
    print(
        f"  total={cache_stats['total_texts']} unique={cache_stats['unique_texts']} "
        f"estimated_hits={cache_stats['estimated_cache_hits']}"
    )

    for n, d in shapes:
        vectors = np.random.randn(n, d).astype(np.float32)
        print(f"\n{'=' * 70}")
        print(f"Shape: ({n}, {d}) — {vectors.nbytes / 1024:.1f} KB original")
        print(f"{'=' * 70}")
        header = (
            f"{'Mode':<10} {'Ratio':>8} {'RMSE':>12} {'CosSim':>10} {'Enc(ms)':>10} {'Dec(ms)':>10}"
        )
        print(header)
        print("-" * 70)
        for mode in modes:
            stats = benchmark_mode(vectors, mode)
            print(
                f"{stats['mode']:<10} "
                f"{stats['compression_ratio']:>8.2f} "
                f"{stats['rmse']:>12.6f} "
                f"{stats['cosine_similarity']:>10.6f} "
                f"{stats['encode_ms']:>10.3f} "
                f"{stats['decode_ms']:>10.3f}"
            )


if __name__ == "__main__":
    main()
