"""Download BEIR datasets directly without beir package.

Downloads TREC-COVID and NFCorpus for benchmarking.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import httpx

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def download_dataset(
    name: str,
    save_path: Path = Path("data/beir"),
) -> Path:
    """Download and extract BEIR dataset."""
    save_path.mkdir(parents=True, exist_ok=True)

    url = f"{BEIR_BASE_URL}/{name}.zip"
    zip_path = save_path / f"{name}.zip"
    extract_path = save_path / name

    if extract_path.exists():
        print(f"{name} already downloaded at {extract_path}")
        return extract_path

    print(f"Downloading {name} from {url}...")

    # Download with progress
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024 * 1024) == 0 or downloaded == total_size:
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"  Progress: {mb:.1f}/{total_mb:.1f} MB")
            else:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_path)

    # Clean up zip
    zip_path.unlink()

    print(f"Downloaded {name} to {extract_path}")
    return extract_path


def load_trec_covid(data_path: Path) -> dict:
    """Load TREC-COVID dataset."""

    # Load corpus
    corpus_path = data_path / "qrels" / "collectionandqueries"
    corpus_file = corpus_path / "docs.tsv" if (corpus_path / "docs.tsv").exists() else None

    corpus = {}
    if corpus_file and corpus_file.exists():
        with open(corpus_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    doc_id, title, text = parts[0], parts[1], parts[2]
                    corpus[doc_id] = {"_id": doc_id, "title": title, "text": f"{title} {text}"}
    else:
        # Try alternative location
        corpus_file = data_path / "collectionandqueries" / "collection.tsv"
        if corpus_file.exists():
            with open(corpus_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        doc_id, text = parts[0], parts[1]
                        corpus[doc_id] = {"_id": doc_id, "title": "", "text": text}

    # Load queries
    queries = {}
    queries_file = data_path / "collectionandqueries" / "queries.tsv"
    if queries_file.exists():
        with open(queries_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    q_id, query = parts[0], parts[1]
                    queries[q_id] = query

    # Load qrels
    from collections import defaultdict

    qrels = defaultdict(dict)
    qrels_file = data_path / "qrels" / "test.tsv"
    if qrels_file.exists():
        with open(qrels_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    q_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if rel > 0:
                        qrels[q_id][doc_id] = rel

    print(f"Loaded TREC-COVID: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": dict(qrels),
    }


def load_nfcorpus(data_path: Path) -> dict:
    """Load NFCorpus dataset."""
    import json

    corpus = {}
    queries = {}
    qrels = {}

    # Load corpus
    corpus_file = data_path / "nfcorpus" / "corpus.jsonl"
    if corpus_file.exists():
        with open(corpus_file) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {
                    "_id": doc["_id"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }

    # Load queries
    for split in ["test", "train"]:
        queries_file = data_path / "nfcorpus" / f"{split}.queries.jsonl"
        if queries_file.exists():
            with open(queries_file) as f:
                for line in f:
                    item = json.loads(line)
                    queries[item["_id"]] = item["text"]

    # Load qrels
    qrels_file = data_path / "nfcorpus" / "qrels" / "test.tsv"
    if qrels_file.exists():
        from collections import defaultdict

        qrels_dict = defaultdict(dict)
        with open(qrels_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    q_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if rel > 0:
                        qrels_dict[q_id][doc_id] = rel
        qrels = dict(qrels_dict)

    print(f"Loaded NFCorpus: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }


if __name__ == "__main__":
    import sys

    # Download datasets
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["trec-covid", "nfcorpus"]

    for name in datasets:
        print(f"\n=== Processing {name} ===")
        try:
            path = download_dataset(name)

            # Load and verify
            if "trec" in name.lower():
                data = load_trec_covid(path)
            else:
                data = load_nfcorpus(path)

            # Save summary
            summary_path = Path("data/beir") / f"{name}_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            # Save samples (don't save full corpus)
            summary = {
                "dataset": name,
                "num_queries": len(data["queries"]),
                "num_docs": len(data["corpus"]),
                "num_qrels": len(data["qrels"]),
                "sample_queries": dict(list(data["queries"].items())[:5]),
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Summary saved to {summary_path}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback

            traceback.print_exc()
