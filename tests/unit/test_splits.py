from reranker.data.splits import partition_rows


def test_partition_rows_is_deterministic_by_group() -> None:
    rows = [
        {"query": "q1", "doc": "a"},
        {"query": "q1", "doc": "b"},
        {"query": "q2", "doc": "c"},
        {"query": "q3", "doc": "d"},
        {"query": "q4", "doc": "e"},
    ]

    train_a = partition_rows(rows, key_fn=lambda row: row["query"], split="train", seed=5)
    train_b = partition_rows(rows, key_fn=lambda row: row["query"], split="train", seed=5)
    test_rows = partition_rows(rows, key_fn=lambda row: row["query"], split="test", seed=5)

    assert train_a == train_b
    assert {row["query"] for row in train_a}.isdisjoint({row["query"] for row in test_rows})
