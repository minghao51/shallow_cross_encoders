"""Validate that all sweep config YAML files parse correctly."""

from __future__ import annotations

from pathlib import Path

from reranker.config import load_yaml_config

SWEEP_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "configs"


def _list_sweep_configs() -> list[Path]:
    return sorted(SWEEP_DIR.glob("*.yaml"))


class TestSweepConfigsExist:
    def test_all_expected_sweeps_present(self):
        configs = {p.stem for p in _list_sweep_configs()}
        expected = {
            "sweep_hybrid",
            "sweep_colbert",
            "sweep_lsh",
            "sweep_active_distill",
            "sweep_cascade",
            "sweep_binary",
            "sweep_pipeline",
            "sweep_distilled",
            "full_sweep",
        }
        missing = expected - configs
        assert not missing, f"Missing sweep configs: {missing}"

    def test_has_nine_configs(self):
        assert 9 <= len(_list_sweep_configs()) <= 9


class TestEachConfigParses:
    @staticmethod
    def _check_config(path: Path) -> None:
        data = load_yaml_config(path)
        assert "name" in data, f"{path.name}: missing 'name'"
        assert "variants" in data, f"{path.name}: missing 'variants'"
        assert len(data["variants"]) > 0, f"{path.name}: no variants"
        for vname, vconfig in data["variants"].items():
            assert isinstance(vconfig, dict), f"{path.name}: variant '{vname}' not a dict"

    def test_all_configs_parse(self):
        for path in _list_sweep_configs():
            self._check_config(path)

    def test_sweep_cascade(self):
        path = SWEEP_DIR / "sweep_cascade.yaml"
        data = load_yaml_config(path)
        for vname, vconfig in data["variants"].items():
            cascade = vconfig.get("cascade", {})
            assert "confidence_threshold" in cascade, f"{vname}: missing threshold"
            assert "confidence_metric" in cascade, f"{vname}: missing metric"

    def test_sweep_binary(self):
        path = SWEEP_DIR / "sweep_binary.yaml"
        data = load_yaml_config(path)
        for vname, vconfig in data["variants"].items():
            binary = vconfig.get("binary", {})
            assert "hamming_top_k" in binary, f"{vname}: missing hamming_top_k"
            assert "bilinear_top_k" in binary, f"{vname}: missing bilinear_top_k"
            assert "quantization" in binary, f"{vname}: missing quantization"

    def test_sweep_pipeline(self):
        path = SWEEP_DIR / "sweep_pipeline.yaml"
        data = load_yaml_config(path)
        for vname, vconfig in data["variants"].items():
            pipeline = vconfig.get("pipeline", {})
            assert "stages" in pipeline, f"{vname}: missing stages"
            assert "top_ks" in pipeline, f"{vname}: missing top_ks"
            assert len(pipeline["stages"]) == len(pipeline["top_ks"]), (
                f"{vname}: stages/top_ks length mismatch"
            )

    def test_sweep_distilled(self):
        path = SWEEP_DIR / "sweep_distilled.yaml"
        data = load_yaml_config(path)
        for vname, vconfig in data["variants"].items():
            distilled = vconfig.get("distilled", {})
            assert "loss_type" in distilled, f"{vname}: missing loss_type"
            assert "C" in distilled, f"{vname}: missing C (regularization)"
