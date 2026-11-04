"""Final stage: the reproduce plan covers every script, and the summary builds from results."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from src import reproduce, summary
from src.config import PROJECT_ROOT, load_config

CFG = load_config()


def test_every_step_is_an_importable_module_with_main() -> None:
    for step in reproduce.steps(from_scratch=True):
        module = importlib.import_module(step[0])
        assert hasattr(module, "main"), step[0]


def test_every_experiment_script_is_in_the_plan() -> None:
    planned = {s[0] for s in reproduce.steps(from_scratch=True)}
    scripts = {f"src.experiments.{p.stem}" for p in (PROJECT_ROOT / "src" / "experiments").glob("*.py")
               if p.stem != "__init__"}
    assert scripts <= planned


def test_md_table() -> None:
    t = summary.md_table(pd.DataFrame({"a": ["x"], "b": [0.12345]}), 2)
    assert t.splitlines() == ["| a | b |", "|---|---|", "| x | 0.12 |"]


@pytest.mark.skipif(not (PROJECT_ROOT / "results" / "shift_real.csv").exists(),
                    reason="needs generated results")
def test_summary_builds_from_results() -> None:
    text = summary.build(PROJECT_ROOT / CFG["paths"]["results"],
                         PROJECT_ROOT / CFG["paths"]["dataset"])
    for heading in ["## Dataset", "## Selectors vs baselines", "## Feature tiers",
                    "## Probing", "## Uncertainty", "## Explainability", "## Strength checks"]:
        assert heading in text
