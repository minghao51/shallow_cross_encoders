"""Unit tests for utils.py module."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from reranker.utils import (
    append_jsonl,
    dump_pickle,
    ensure_parent,
    load_pickle,
    read_json,
    read_jsonl,
    to_serializable,
    write_json,
    write_jsonl,
)


class TestEnsureParent:
    """Tests for ensure_parent function."""

    @pytest.mark.unit
    def test_ensure_parent_creates_directories(self, tmp_path: Path) -> None:
        """ensure_parent should create parent directories."""
        nested_path = tmp_path / "level1" / "level2" / "file.txt"
        result = ensure_parent(nested_path)

        assert result == nested_path
        assert result.parent.exists()
        assert result.parent.is_dir()

    @pytest.mark.unit
    def test_ensure_parent_with_existing_directories(self, tmp_path: Path) -> None:
        """ensure_parent should work with existing directories."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir(parents=True, exist_ok=True)

        file_path = existing_dir / "file.txt"
        result = ensure_parent(file_path)

        assert result == file_path

    @pytest.mark.unit
    def test_ensure_parent_with_string_path(self, tmp_path: Path) -> None:
        """ensure_parent should accept string paths."""
        result = ensure_parent(str(tmp_path / "test" / "file.txt"))

        assert isinstance(result, Path)
        assert result.parent.exists()


class TestWriteJson:
    """Tests for write_json function."""

    @pytest.mark.unit
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        """write_json should create a JSON file."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        write_json(test_file, data)

        assert test_file.exists()
        assert test_file.is_file()

    @pytest.mark.unit
    def test_write_json_content(self, tmp_path: Path) -> None:
        """write_json should write correct JSON content."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        write_json(test_file, data)

        content = test_file.read_text(encoding="utf-8")
        loaded = json.loads(content)
        assert loaded == data

    @pytest.mark.unit
    def test_write_json_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_json should create parent directories."""
        nested_file = tmp_path / "level1" / "level2" / "test.json"
        data = {"key": "value"}

        write_json(nested_file, data)

        assert nested_file.exists()
        assert nested_file.parent.exists()

    @pytest.mark.unit
    def test_write_json_sorts_keys(self, tmp_path: Path) -> None:
        """write_json should sort keys."""
        test_file = tmp_path / "test.json"
        data = {"z": 1, "a": 2, "m": 3}

        write_json(test_file, data)

        content = test_file.read_text(encoding="utf-8")
        # Keys should be sorted: a, m, z
        lines = content.strip().split("\n")
        # First key after opening brace should be "a"
        assert '"a"' in lines[1]

    @pytest.mark.unit
    def test_write_json_handles_special_characters(self, tmp_path: Path) -> None:
        """write_json should handle special characters."""
        test_file = tmp_path / "test.json"
        data = {"unicode": "こんにちは", "emoji": "😀", "quotes": 'test"quote'}

        write_json(test_file, data)

        loaded = read_json(test_file)
        assert loaded["unicode"] == "こんにちは"
        assert loaded["emoji"] == "😀"
        assert loaded["quotes"] == 'test"quote'


class TestReadJson:
    """Tests for read_json function."""

    @pytest.mark.unit
    def test_read_json_loads_file(self, tmp_path: Path) -> None:
        """read_json should load JSON file."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = read_json(test_file)

        assert loaded == data

    @pytest.mark.unit
    def test_read_json_with_string_path(self, tmp_path: Path) -> None:
        """read_json should accept string paths."""
        test_file = tmp_path / "test.json"
        data = {"key": "value"}
        test_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = read_json(str(test_file))

        assert loaded == data

    @pytest.mark.unit
    def test_read_json_file_not_found(self, tmp_path: Path) -> None:
        """read_json should raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "nonexistent.json")

    @pytest.mark.unit
    def test_read_json_malformed_json(self, tmp_path: Path) -> None:
        """read_json should raise json.JSONDecodeError for malformed JSON."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{invalid json}", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            read_json(test_file)


class TestAppendJsonl:
    """Tests for append_jsonl function."""

    @pytest.mark.unit
    def test_append_jsonl_adds_line(self, tmp_path: Path) -> None:
        """append_jsonl should append a JSON line to file."""
        test_file = tmp_path / "test.jsonl"
        record = {"key": "value", "number": 42}

        append_jsonl(test_file, record)

        assert test_file.exists()
        content = test_file.read_text(encoding="utf-8")
        assert json.loads(content.strip()) == record

    @pytest.mark.unit
    def test_append_jsonl_multiple_records(self, tmp_path: Path) -> None:
        """append_jsonl should append multiple records."""
        test_file = tmp_path / "test.jsonl"

        append_jsonl(test_file, {"id": 1})
        append_jsonl(test_file, {"id": 2})
        append_jsonl(test_file, {"id": 3})

        lines = test_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}
        assert json.loads(lines[2]) == {"id": 3}

    @pytest.mark.unit
    def test_append_jsonl_creates_parent_directories(self, tmp_path: Path) -> None:
        """append_jsonl should create parent directories."""
        nested_file = tmp_path / "level1" / "test.jsonl"

        append_jsonl(nested_file, {"key": "value"})

        assert nested_file.exists()
        assert nested_file.parent.exists()


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    @pytest.mark.unit
    def test_write_jsonl_creates_file(self, tmp_path: Path) -> None:
        """write_jsonl should create a JSONL file."""
        test_file = tmp_path / "test.jsonl"
        records = [{"id": 1}, {"id": 2}]

        write_jsonl(test_file, records)

        assert test_file.exists()

    @pytest.mark.unit
    def test_write_jsonl_content(self, tmp_path: Path) -> None:
        """write_jsonl should write correct JSONL content."""
        test_file = tmp_path / "test.jsonl"
        records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        write_jsonl(test_file, records)

        loaded = read_jsonl(test_file)
        assert loaded == records

    @pytest.mark.unit
    def test_write_jsonl_overwrites_existing(self, tmp_path: Path) -> None:
        """write_jsonl should overwrite existing file."""
        test_file = tmp_path / "test.jsonl"

        write_jsonl(test_file, [{"id": 1}])
        write_jsonl(test_file, [{"id": 2}])

        loaded = read_jsonl(test_file)
        assert len(loaded) == 1
        assert loaded[0]["id"] == 2

    @pytest.mark.unit
    def test_write_jsonl_with_iterable(self, tmp_path: Path) -> None:
        """write_jsonl should accept any iterable."""
        test_file = tmp_path / "test.jsonl"

        def record_generator():
            yield {"id": 1}
            yield {"id": 2}

        write_jsonl(test_file, record_generator())

        loaded = read_jsonl(test_file)
        assert len(loaded) == 2


class TestReadJsonl:
    """Tests for read_jsonl function."""

    @pytest.mark.unit
    def test_read_jsonl_loads_file(self, tmp_path: Path) -> None:
        """read_jsonl should load JSONL file."""
        test_file = tmp_path / "test.jsonl"
        records = [{"id": 1}, {"id": 2}]
        write_jsonl(test_file, records)

        loaded = read_jsonl(test_file)

        assert loaded == records

    @pytest.mark.unit
    def test_read_jsonl_empty_file(self, tmp_path: Path) -> None:
        """read_jsonl should return empty list for empty file."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text("", encoding="utf-8")

        loaded = read_jsonl(test_file)

        assert loaded == []

    @pytest.mark.unit
    def test_read_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        """read_jsonl should skip blank lines."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"id": 1}\n\n{"id": 2}\n  \n{"id": 3}', encoding="utf-8")

        loaded = read_jsonl(test_file)

        assert len(loaded) == 3
        assert loaded[0]["id"] == 1
        assert loaded[1]["id"] == 2
        assert loaded[2]["id"] == 3

    @pytest.mark.unit
    def test_read_jsonl_nonexistent_file(self, tmp_path: Path) -> None:
        """read_jsonl should return empty list for nonexistent file."""
        loaded = read_jsonl(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    @pytest.mark.unit
    def test_read_jsonl_malformed_line(self, tmp_path: Path) -> None:
        """read_jsonl should raise json.JSONDecodeError for malformed line."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"id": 1}\n{invalid}\n{"id": 2}', encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            read_jsonl(test_file)


class TestDumpPickle:
    """Tests for dump_pickle function."""

    @pytest.mark.unit
    def test_dump_pickle_creates_file(self, tmp_path: Path) -> None:
        """dump_pickle should create a pickle file."""
        test_file = tmp_path / "test.pkl"
        data = [1, 2, 3, "test"]

        dump_pickle(test_file, data)

        assert test_file.exists()

    @pytest.mark.unit
    def test_dump_pickle_content(self, tmp_path: Path) -> None:
        """dump_pickle should write correct pickle content."""
        test_file = tmp_path / "test.pkl"
        data = {"key": "value", "nested": [1, 2, 3]}

        dump_pickle(test_file, data)

        loaded = load_pickle(test_file)
        assert loaded == data

    @pytest.mark.unit
    def test_dump_pickle_creates_parent_directories(self, tmp_path: Path) -> None:
        """dump_pickle should create parent directories."""
        nested_file = tmp_path / "level1" / "test.pkl"

        dump_pickle(nested_file, [1, 2, 3])

        assert nested_file.exists()
        assert nested_file.parent.exists()

    @pytest.mark.unit
    def test_dump_pickle_handles_complex_objects(self, tmp_path: Path) -> None:
        """dump_pickle should handle complex Python objects."""
        test_file = tmp_path / "test.pkl"

        class CustomClass:
            def __init__(self, value: int) -> None:
                self.value = value

        data = CustomClass(42)

        dump_pickle(test_file, data)

        loaded = load_pickle(test_file)
        assert loaded.value == 42


class TestLoadPickle:
    """Tests for load_pickle function."""

    @pytest.mark.unit
    def test_load_pickle_loads_file(self, tmp_path: Path) -> None:
        """load_pickle should load pickle file."""
        test_file = tmp_path / "test.pkl"
        data = [1, 2, 3, "test"]
        dump_pickle(test_file, data)

        loaded = load_pickle(test_file)

        assert loaded == data

    @pytest.mark.unit
    def test_load_pickle_with_string_path(self, tmp_path: Path) -> None:
        """load_pickle should accept string paths."""
        test_file = tmp_path / "test.pkl"
        data = {"key": "value"}
        dump_pickle(test_file, data)

        loaded = load_pickle(str(test_file))

        assert loaded == data

    @pytest.mark.unit
    def test_load_pickle_file_not_found(self, tmp_path: Path) -> None:
        """load_pickle should raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "nonexistent.pkl")

    @pytest.mark.unit
    def test_load_pickle_corrupted_file(self, tmp_path: Path) -> None:
        """load_pickle should raise pickle.UnpicklingError for corrupted file."""
        test_file = tmp_path / "test.pkl"
        test_file.write_bytes(b"not a valid pickle")

        with pytest.raises(pickle.UnpicklingError):
            load_pickle(test_file)


class TestToSerializable:
    """Tests for to_serializable function."""

    @pytest.mark.unit
    def test_to_serializable_with_dict(self) -> None:
        """to_serializable should handle dictionaries."""
        data = {"key": "value", "number": 42}
        result = to_serializable(data)
        assert result == data

    @pytest.mark.unit
    def test_to_serializable_with_list(self) -> None:
        """to_serializable should handle lists."""
        data = [1, 2, 3, "test"]
        result = to_serializable(data)
        assert result == data

    @pytest.mark.unit
    def test_to_serializable_with_tuple(self) -> None:
        """to_serializable should convert tuples to lists."""
        data = (1, 2, 3)
        result = to_serializable(data)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_to_serializable_with_nested_structures(self) -> None:
        """to_serializable should handle nested structures."""
        data = {"list": [1, 2, 3], "dict": {"nested": "value"}}
        result = to_serializable(data)
        assert result == data

    @pytest.mark.unit
    def test_to_serializable_with_dataclass(self) -> None:
        """to_serializable should convert dataclasses to dicts."""

        @dataclass
        class TestDataclass:
            name: str
            value: int

        data = TestDataclass("test", 42)
        result = to_serializable(data)

        assert result == {"name": "test", "value": 42}
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_to_serializable_with_pydantic_model(self) -> None:
        """to_serializable should convert Pydantic models to dicts."""

        class TestModel(BaseModel):
            name: str
            value: int

        data = TestModel(name="test", value=42)
        result = to_serializable(data)

        assert result == {"name": "test", "value": 42}
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_to_serializable_with_mixed_types(self) -> None:
        """to_serializable should handle mixed types."""

        @dataclass
        class Inner:
            x: int

        class TestModel(BaseModel):
            name: str
            value: int

        data = {
            "dataclass": Inner(1),
            "pydantic": TestModel(name="test", value=42),  # type: ignore
            "list": [1, 2, 3],
            "plain": "string",
        }

        result = to_serializable(data)

        assert result["dataclass"] == {"x": 1}
        assert result["pydantic"] == {"name": "test", "value": 42}
        assert result["list"] == [1, 2, 3]
        assert result["plain"] == "string"

    @pytest.mark.unit
    def test_to_serializable_preserves_primitives(self) -> None:
        """to_serializable should preserve primitive types."""
        assert to_serializable("string") == "string"
        assert to_serializable(42) == 42
        assert to_serializable(3.14) == 3.14
        assert to_serializable(True) is True
        assert to_serializable(None) is None
