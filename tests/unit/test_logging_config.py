"""Unit tests for logging_config.py — structlog configuration."""

from __future__ import annotations

import logging

import pytest

from reranker.logging_config import configure_logging


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_runs_without_error(self) -> None:
        configure_logging()

    def test_default_level_is_info(self) -> None:
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_explicit_level_debug(self) -> None:
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_explicit_level_warning(self) -> None:
        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_env_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANKER_LOG_LEVEL", "ERROR")
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_env_override_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANKER_LOG_LEVEL", "debug")
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_explicit_level_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RERANKER_LOG_LEVEL", "ERROR")
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_env_defaults_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANKER_LOG_LEVEL", "INVALID_LEVEL")
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_json_format_configures_without_error(self) -> None:
        configure_logging(json_format=True)

    def test_replaces_existing_handlers(self) -> None:
        configure_logging()
        root = logging.getLogger()
        first_handlers = list(root.handlers)
        configure_logging()
        assert len(root.handlers) == len(first_handlers)

    def test_noisy_loggers_set_to_warning(self) -> None:
        configure_logging()
        for name in ("httpx", "urllib3", "transformers", "sentence_transformers"):
            assert logging.getLogger(name).level == logging.WARNING
