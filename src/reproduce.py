"""Regenerate every table and figure in results/ from the datasets in data/.

Default: analysis only (uses data/dataset*.csv and data/probes*.csv, ~10 minutes).
--from-scratch also regenerates graph instances and re-times everything first (hours;
needs AC power; timings will differ slightly from the committed ones on other hardware).

Run:  python -m src.reproduce [--from-scratch] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.config import PROJECT_ROOT

DATA_STEPS = [
    ["src.generators"], ["src.benchmark"], ["src.probe"],
    ["src.generators", "--profile", "large"], ["src.benchmark", "--profile", "large"],
    ["src.probe", "--profile", "large"],
    ["src.real_graphs"], ["src.benchmark", "--profile", "real"],
    ["src.probe", "--profile", "real"],
]

ANALYSIS_STEPS = [
    ["src.experiments.label_report"],          # Stage 4: label balance, ties, noise
    ["src.evaluate"],                          # Stage 5: selectors vs baselines
    ["src.experiments.tiers"],                 # Stage 6: cost-aware tiers
    ["src.experiments.probing"],               # Stage 7: probing
    ["src.experiments.uncertainty"],           # Stage 8: calibration, conformal, abstention
    ["src.explain"],                           # Stage 8: SHAP, permutation, rules
    ["src.experiments.size_shift"],            # Stage 9
    ["src.experiments.labels_ablation"],       # Stage 9
    ["src.experiments.real_graphs"],           # Stage 9
    ["src.experiments.algorithm_map"],         # Stage 10 (demo page 2)
    ["src.summary"],                           # results/summary.md
]

# The tier-cost report needs the graph instances (not in git): run it only if they exist.
FINGERPRINT_STEP = ["src.fingerprint"]


def steps(from_scratch: bool) -> list[list[str]]:
    out = list(DATA_STEPS) if from_scratch else []
    if from_scratch or (PROJECT_ROOT / "data" / "raw_instances" / "index.csv").exists():
        out.append(FINGERPRINT_STEP)
    return out + ANALYSIS_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list the steps only")
    args = parser.parse_args()
    plan = steps(args.from_scratch)
    for i, step in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] python -m {' '.join(step)}", flush=True)
        if args.dry_run:
            continue
        start = time.time()
        result = subprocess.run([sys.executable, "-W", "ignore", "-m", *step], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"step failed: {' '.join(step)} (exit {result.returncode})")
        print(f"      done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
