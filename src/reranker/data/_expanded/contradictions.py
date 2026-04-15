"""Expanded offline contradiction generation with streaming support."""
# ruff: noqa: E501

from __future__ import annotations

import random
from collections.abc import Iterator

from reranker.data._expanded.helpers import limited_shuffle
from reranker.data._expanded.types import ExpandedContradictionRecord

CONTRADICTION_TEMPLATES = [
    {
        "subject": "Python 3.12",
        "field": "release_date",
        "value_a": "October 2023",
        "value_b": "March 2024",
    },
    {
        "subject": "React 18",
        "field": "concurrent_features",
        "value_a": "enabled by default",
        "value_b": "opt-in only",
    },
    {
        "subject": "TypeScript 5.0",
        "field": "decorators",
        "value_a": "uses TC39 decorators",
        "value_b": "uses experimental decorators",
    },
    {
        "subject": "Node.js 20",
        "field": "default_module",
        "value_a": "CommonJS",
        "value_b": "ES Modules",
    },
    {
        "subject": "Django 5.0",
        "field": "python_requirement",
        "value_a": "Python 3.10+",
        "value_b": "Python 3.12+",
    },
    {
        "subject": "FastAPI",
        "field": "async_support",
        "value_a": "fully async native",
        "value_b": "sync only",
    },
    {
        "subject": "PyTorch 2.0",
        "field": "compilation",
        "value_a": "torch.compile default",
        "value_b": "manual tracing",
    },
    {
        "subject": "TensorFlow 2.x",
        "field": "execution_mode",
        "value_a": "eager execution",
        "value_b": "graph only",
    },
    {
        "subject": "GPT-4",
        "field": "context_window",
        "value_a": "128K tokens",
        "value_b": "8K tokens",
    },
    {
        "subject": "LLaMA 2",
        "field": "license",
        "value_a": "commercial use allowed",
        "value_b": "research only",
    },
    {
        "subject": "Stable Diffusion XL",
        "field": "resolution",
        "value_a": "1024x1024 native",
        "value_b": "512x512 native",
    },
    {
        "subject": "BERT",
        "field": "training_objective",
        "value_a": "masked language modeling",
        "value_b": "causal modeling",
    },
    {
        "subject": "ResNet-50",
        "field": "parameters",
        "value_a": "25 million",
        "value_b": "60 million",
    },
    {
        "subject": "ViT-Large",
        "field": "attention_heads",
        "value_a": "16 heads",
        "value_b": "12 heads",
    },
    {
        "subject": "Kubernetes 1.28",
        "field": "default_runtime",
        "value_a": "containerd",
        "value_b": "Docker",
    },
    {
        "subject": "Terraform 1.6",
        "field": "state_locking",
        "value_a": "DynamoDB built-in",
        "value_b": "manual locking",
    },
    {
        "subject": "PostgreSQL 16",
        "field": "json_support",
        "value_a": "native JSONB",
        "value_b": "text only",
    },
    {
        "subject": "Redis 7.2",
        "field": "persistence",
        "value_a": "RDB snapshots",
        "value_b": "AOF log",
    },
    {
        "subject": "Kafka 3.6",
        "field": "consumer_groups",
        "value_a": "KRaft mode",
        "value_b": "ZooKeeper required",
    },
    {
        "subject": "Elasticsearch 8.x",
        "field": "security",
        "value_a": "enabled by default",
        "value_b": "plugin required",
    },
    {
        "subject": "Project Atlas",
        "field": "launch_date",
        "value_a": "Q1 2025",
        "value_b": "Q3 2025",
    },
    {
        "subject": "Project Atlas",
        "field": "unit_count",
        "value_a": "450 units",
        "value_b": "600 units",
    },
    {"subject": "Project Atlas", "field": "expected_top", "value_a": "2028", "value_b": "2030"},
    {
        "subject": "HDB BTO",
        "field": "application_period",
        "value_a": "February 2025",
        "value_b": "May 2025",
    },
    {
        "subject": "Marina Bay Residences",
        "field": "psf_price",
        "value_a": "$3,200 psf",
        "value_b": "$4,100 psf",
    },
    {
        "subject": "Tengah New Town",
        "field": "completion_year",
        "value_a": "2026",
        "value_b": "2030",
    },
    {"subject": "S&P 500", "field": "return_2024", "value_a": "24.2%", "value_b": "18.7%"},
    {
        "subject": "Singapore CPF",
        "field": "oa_rate",
        "value_a": "2.5% per annum",
        "value_b": "4.0% per annum",
    },
    {"subject": "STI ETF", "field": "dividend_yield", "value_a": "4.2%", "value_b": "6.8%"},
    {
        "subject": "US Federal Reserve",
        "field": "current_rate",
        "value_a": "5.25-5.50%",
        "value_b": "3.00-3.25%",
    },
    {
        "subject": "WHO Guidelines",
        "field": "sleep_recommendation",
        "value_a": "7-9 hours",
        "value_b": "5-6 hours",
    },
    {
        "subject": "CRISPR Therapy",
        "field": "fda_approval",
        "value_a": "approved",
        "value_b": "in trials",
    },
    {
        "subject": "mRNA Vaccines",
        "field": "storage_temp",
        "value_a": "-70C required",
        "value_b": "2-8C fridge",
    },
]

CONTROL_TEMPLATES = [
    {"subject": "Python 3.12", "field": "type_syntax", "value": "uses square brackets"},
    {"subject": "React 18", "field": "rendering", "value": "uses virtual DOM"},
    {"subject": "Kubernetes", "field": "orchestration", "value": "manages containers"},
    {"subject": "PostgreSQL", "field": "license", "value": "open source"},
    {"subject": "Singapore CPF", "field": "mandatory", "value": "required for citizens"},
    {"subject": "HTTP/3", "field": "transport", "value": "uses QUIC"},
    {"subject": "BERT", "field": "architecture", "value": "encoder-only"},
    {"subject": "ResNet", "field": "innovation", "value": "skip connections"},
]


def iter_expanded_contradictions(
    contradiction_count: int = 1000,
    control_count: int = 400,
    seed: int = 42,
) -> Iterator[ExpandedContradictionRecord]:
    """Yield contradiction/control records without forcing eager consumers."""
    rng = random.Random(seed)
    records: list[ExpandedContradictionRecord] = []

    for template in CONTRADICTION_TEMPLATES:
        for _ in range(max(1, contradiction_count // len(CONTRADICTION_TEMPLATES))):
            records.append(
                {
                    "subject": template["subject"],
                    "doc_a": f"{template['subject']} states {template['field']} is {template['value_a']}.",
                    "doc_b": f"{template['subject']} specifies {template['field']} as {template['value_b']}.",
                    "contradicted_field": template["field"],
                    "value_a": template["value_a"],
                    "value_b": template["value_b"],
                    "is_contradiction": True,
                }
            )

    for template in CONTROL_TEMPLATES:
        for _ in range(max(1, control_count // len(CONTROL_TEMPLATES))):
            records.append(
                {
                    "subject": template["subject"],
                    "doc_a": f"{template['subject']} confirms {template['field']} {template['value']}.",
                    "doc_b": f"{template['subject']} states {template['field']} {template['value']}.",
                    "contradicted_field": template["field"],
                    "value_a": template["value"],
                    "value_b": template["value"],
                    "is_contradiction": False,
                }
            )

    yield from limited_shuffle(records, limit=contradiction_count + control_count, rng=rng)


def generate_expanded_contradictions(
    contradiction_count: int = 1000,
    control_count: int = 400,
    seed: int = 42,
) -> list[ExpandedContradictionRecord]:
    """Generate expanded contradiction/control examples."""
    return list(
        iter_expanded_contradictions(
            contradiction_count=contradiction_count,
            control_count=control_count,
            seed=seed,
        )
    )
