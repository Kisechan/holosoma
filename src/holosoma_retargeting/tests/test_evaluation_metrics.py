from __future__ import annotations

import numpy as np

from holosoma_retargeting.evaluation.metrics import (
    binary_contact_metrics,
    canonical_config_hash,
    foot_skating_metrics,
    summarize_penetration_depths,
)
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity


def test_penetration_thresholds_and_percentiles() -> None:
    result = summarize_penetration_depths([0.0, 0.001, 0.003, 0.006, 0.012])
    assert result["tol_0mm"] == 0.8
    assert result["tol_2mm"] == 0.6
    assert result["tol_5mm"] == 0.4
    assert result["tol_10mm"] == 0.2
    assert result["max"] == 0.012


def test_strict_foot_skating_uses_fps() -> None:
    left = np.array([[0.0, 0.0], [0.005, 0.0], [0.010, 0.0]])
    right = np.zeros_like(left)
    stance = np.ones((3, 2), dtype=bool)
    strict = foot_skating_metrics(left, right, stance, fps=30, threshold_mps=0.01)
    legacy = foot_skating_metrics(left, right, stance, fps=30, threshold_mps=0.01, legacy=True)
    assert np.isclose(strict["max_velocity"], 0.15)
    assert strict["duration"] == 2 / 3
    assert legacy["duration"] == 0.0


def test_contact_metrics_include_false_positives_and_switches() -> None:
    target = np.array([[0, 0], [1, 0], [1, 0]], dtype=bool)
    predicted = np.array([[0, 1], [1, 0], [0, 0]], dtype=bool)
    result = binary_contact_metrics(target, predicted)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["n_switches"] == 3


def test_stance_detector_has_explicit_legacy_ablation() -> None:
    joints = np.zeros((3, 2, 3), dtype=float)
    joints[:, 0, 0] = [0.0, 0.005, 0.010]
    legacy = extract_foot_sticking_sequence_velocity(joints, ["L", "R"], ["L", "R"])
    strict = extract_foot_sticking_sequence_velocity(joints, ["L", "R"], ["L", "R"], fps=30)
    assert legacy[1]["L_Toe"]
    assert not strict[1]["L_Toe"]


def test_config_hash_is_order_independent() -> None:
    assert canonical_config_hash({"a": 1, "b": 2}) == canonical_config_hash({"b": 2, "a": 1})
