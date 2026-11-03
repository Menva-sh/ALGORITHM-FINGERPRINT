"""Stage 0: config loads and has the required sections."""
from src.config import PROJECT_ROOT, load_config


def test_config_loads() -> None:
    cfg = load_config()
    for key in ["seed", "paths", "generation", "algorithms", "benchmark", "features", "model"]:
        assert key in cfg


def test_config_values_sane() -> None:
    cfg = load_config()
    assert isinstance(cfg["seed"], int)
    assert cfg["benchmark"]["repeats"] >= 1
    assert 0 < cfg["benchmark"]["tie_threshold_rel"] < 1
    assert set(cfg["features"]["tiers"]) == {0, 1, 2, 3}
    assert "dfs" not in cfg["algorithms"]["candidates"]


def test_repo_layout_exists() -> None:
    for d in ["data/raw_instances", "data/real_graphs", "src/experiments",
              "app", "notebooks", "results", "tests"]:
        assert (PROJECT_ROOT / d).is_dir(), d
