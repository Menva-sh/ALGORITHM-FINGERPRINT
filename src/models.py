"""Selector models: Random Forest (primary), RBF-SVM, a small MLP, and a shallow
decision tree evaluated in plain Python (cheap enough to pay off at query time).

Every model scores each candidate algorithm; the selector picks the best-scoring
algorithm that is APPLICABLE to the instance (inapplicable ones are masked out).
SVM and MLP see log1p-transformed, standardised features; the trees see raw features.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def feature_columns(df: pd.DataFrame, cfg: dict[str, Any], tiers: list[int]) -> list[str]:
    """Dataset columns holding the features of the given tiers."""
    names = [n for t in tiers for n in cfg["features"]["tiers"][t]]
    cols = [f"f_{n}" for n in names]
    missing = [c for c in cols if c not in df]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return cols


def feature_cost_ns(df: pd.DataFrame, tiers: list[int]) -> np.ndarray:
    """Per-row cost of computing the given tiers (ns)."""
    return df[[f"cost_tier{t}_ns" for t in tiers]].sum(axis=1).to_numpy(dtype=float)


def applicability_mask(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    """Boolean matrix (rows x candidates): algorithm applicable to the instance."""
    return df[[f"app_{a}" for a in candidates]].to_numpy(dtype=bool)


class PlainTreeSelector(ClassifierMixin, BaseEstimator):
    """Shallow decision tree trained with sklearn, then evaluated in PLAIN PYTHON.

    The candidate algorithms are plain Python, so the selector's inference is too:
    `choose()` walks lists (no numpy, no sklearn call overhead) and does the masked
    argmax itself. `predict_proba` gives the same scores for batch evaluation.
    """

    def __init__(self, max_depth: int = 6, candidates: tuple[str, ...] = (),
                 random_state: int | None = None) -> None:
        self.max_depth = max_depth
        self.candidates = candidates
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlainTreeSelector":
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, y)
        self.tree_ = tree  # kept for explanations (SHAP); choose() never uses it
        self.classes_ = tree.classes_
        t = tree.tree_
        self._left = t.children_left.tolist()
        self._right = t.children_right.tolist()
        self._feature = t.feature.tolist()
        self._threshold = t.threshold.tolist()
        cand = list(self.candidates)
        self._leaf_scores: list[list[float]] = []  # per node, scores in candidate order
        for node in range(t.node_count):
            dist = t.value[node][0] / t.value[node][0].sum()
            scores = [-math.inf] * len(cand)
            for c, p in zip(self.classes_, dist):
                scores[cand.index(c)] = float(p)
            self._leaf_scores.append(scores)
        return self

    def _leaf(self, x: list[float]) -> int:
        node, left, right = 0, self._left, self._right
        feature, threshold = self._feature, self._threshold
        while left[node] != -1:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        return node

    def choose(self, x: list[float], applicable: list[bool]) -> int:
        """Index (candidate order) of the best-scoring applicable algorithm."""
        scores = self._leaf_scores[self._leaf(x)]
        best, best_score = -1, -math.inf
        for j, ok in enumerate(applicable):
            if ok and scores[j] > best_score:
                best, best_score = j, scores[j]
        return best if best >= 0 else applicable.index(True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cand = list(self.candidates)
        cols = [cand.index(c) for c in self.classes_]
        return np.array([[self._leaf_scores[self._leaf(row)][j] for j in cols]
                         for row in np.asarray(X, dtype=float).tolist()])


def build_models(cfg: dict[str, Any]) -> dict[str, ClassifierMixin]:
    """Fresh, unfitted models with seeds and settings from config."""
    m, seed = cfg["model"], cfg["seed"]
    log_scale = [FunctionTransformer(np.log1p), StandardScaler()]
    return {
        "plain_tree": PlainTreeSelector(max_depth=m["plain_tree"]["max_depth"],
                                        candidates=tuple(cfg["algorithms"]["candidates"]),
                                        random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=m["rf"]["n_estimators"], min_samples_leaf=m["rf"]["min_samples_leaf"],
            random_state=seed, n_jobs=1),
        "svm": make_pipeline(*log_scale, SVC(kernel="rbf", C=m["svm"]["C"],
                                             gamma=m["svm"]["gamma"], random_state=seed)),
        "mlp": make_pipeline(*log_scale, MLPClassifier(
            hidden_layer_sizes=tuple(m["mlp"]["hidden_layer_sizes"]),
            max_iter=m["mlp"]["max_iter"], early_stopping=True, random_state=seed)),
    }


def class_scores(model: ClassifierMixin, X: np.ndarray, candidates: list[str]) -> np.ndarray:
    """Score per candidate (rows x candidates); classes never seen in training get -inf."""
    raw = model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)
    classes = list(model.classes_)
    if raw.ndim == 1:  # binary decision_function: positive -> classes[1]
        raw = np.column_stack([-raw, raw])
    scores = np.full((X.shape[0], len(candidates)), -np.inf)
    for j, c in enumerate(classes):
        scores[:, candidates.index(c)] = raw[:, j]
    return scores


def masked_choice(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Index of the best-scoring APPLICABLE candidate per row.

    If every applicable candidate scores -inf (unseen classes), the first applicable
    candidate in config order is used.
    """
    masked = np.where(mask, scores, -np.inf)
    choice = masked.argmax(axis=1)
    no_score = ~np.isfinite(masked.max(axis=1))
    choice[no_score] = mask[no_score].argmax(axis=1)
    return choice


def select(model: ClassifierMixin, X: np.ndarray, mask: np.ndarray,
           candidates: list[str]) -> np.ndarray:
    """Chosen candidate index per row."""
    return masked_choice(class_scores(model, X, candidates), mask)


def inference_ns(model: ClassifierMixin, X: np.ndarray, mask: np.ndarray,
                 candidates: list[str], n_samples: int, seed: int) -> float:
    """Median wall-clock time to select for ONE instance (scores + masking), in ns.

    Single-row calls, as the selector would be used before one query. Models with a
    plain-Python `choose()` are timed through it (features arrive as Python floats from
    the fingerprint, so the list conversion happens outside the timer).
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    if hasattr(model, "choose"):
        xs, ms = X.tolist(), mask.tolist()
        call = lambda r: model.choose(xs[r], ms[r])  # noqa: E731
    else:
        call = lambda r: select(model, X[r:r + 1], mask[r:r + 1], candidates)  # noqa: E731
    call(rows[0])  # warm-up
    times = []
    for r in rows:
        start = time.perf_counter_ns()
        call(r)
        times.append(time.perf_counter_ns() - start)
    return float(np.median(times))


# ------------------------------------------------------------------ uncertainty

def masked_probs(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scores -> probabilities over APPLICABLE candidates only (rows sum to 1).

    Unseen classes (-inf) and inapplicable algorithms get 0; a row with no mass on any
    applicable algorithm becomes uniform over the applicable ones.
    """
    p = np.where(mask & np.isfinite(scores), np.clip(scores, 0.0, None), 0.0)
    total = p.sum(axis=1, keepdims=True)
    uniform = mask / mask.sum(axis=1, keepdims=True)
    return np.where(total > 0, p / np.where(total > 0, total, 1.0), uniform)


def calibrated(model: ClassifierMixin, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               cfg: dict[str, Any]) -> ClassifierMixin:
    """Probability-calibrated copy of model (CalibratedClassifierCV with GROUP folds)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    c = cfg["model"]["calibration"]
    splits = list(GroupKFold(n_splits=c["folds"]).split(X, y, groups))
    return CalibratedClassifierCV(model, method=c["method"], cv=splits).fit(X, y)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int) -> float:
    """ECE of top-choice confidence: bin-weighted |accuracy - mean confidence|."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def conformal_threshold(probs: np.ndarray, label_idx: np.ndarray, coverage: float) -> float:
    """Split-conformal threshold on the nonconformity score 1 - p(true algorithm).

    Returns the k-th smallest calibration score with k = ceil((n + 1) * coverage) (the
    finite-sample conformal quantile); if k > n every algorithm is included (threshold 1).
    """
    scores = np.sort(1.0 - probs[np.arange(len(label_idx)), label_idx])
    k = int(np.ceil((len(scores) + 1) * coverage))
    return 1.0 if k > len(scores) else float(scores[k - 1])


def conformal_sets(probs: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (rows x candidates): applicable algorithms with 1 - p <= threshold.

    An empty set (nothing plausible enough) falls back to the top-probability algorithm.
    """
    sets = mask & (1.0 - probs <= threshold)
    empty = ~sets.any(axis=1)
    sets[empty, probs[empty].argmax(axis=1)] = True
    return sets


def abstaining_cost(sets: np.ndarray, probs: np.ndarray, T: np.ndarray, fallback: int,
                    policy: str) -> np.ndarray:
    """Search cost (ns) per row of the selector with abstention.

    Set of one algorithm -> run it. Larger set ("not sure"):
      'fallback'  run the safe default algorithm (index `fallback`)
      'top2'      run the two most probable algorithms interleaved on one core: both
                  advance at half speed until the first finishes -> 2 * min(t1, t2)
      'none'      ignore the set, run the top-probability algorithm (no abstention)
    """
    rows = np.arange(len(T))
    top = probs.argmax(axis=1)
    cost = T[rows, top].copy()
    if policy == "none":
        return cost
    unsure = sets.sum(axis=1) > 1
    if policy == "fallback":
        cost[unsure] = T[unsure, fallback]
    elif policy == "top2":
        second = np.argsort(-probs, axis=1)[:, 1]
        pair = np.minimum(T[rows, top], np.where(np.isnan(T[rows, second]), np.inf,
                                                 T[rows, second]))
        cost[unsure] = 2 * pair[unsure]
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return cost
