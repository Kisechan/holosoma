from __future__ import annotations

from pathlib import Path

import numpy as np
from holosoma_retargeting.evaluation.eval_retargeting import _object_mesh_scale_from_scene
from holosoma_retargeting.evaluation.metrics import (
    LEGACY_FOOT_SKATING_THRESHOLD_M_PER_FRAME,
    STRICT_FOOT_SKATING_THRESHOLD_MPS,
    binary_contact_metrics,
    canonical_config_hash,
    foot_skating_metrics,
    resolve_foot_skating_threshold,
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


def test_foot_skating_threshold_defaults_have_explicit_units() -> None:
    assert resolve_foot_skating_threshold("strict") == (
        STRICT_FOOT_SKATING_THRESHOLD_MPS,
        "m_per_second",
    )
    assert resolve_foot_skating_threshold("legacy") == (
        LEGACY_FOOT_SKATING_THRESHOLD_M_PER_FRAME,
        "m_per_frame",
    )


def test_strict_default_ignores_velocity_below_point_three_mps() -> None:
    left = np.array([[0.0, 0.0], [0.005, 0.0], [0.010, 0.0]])
    right = np.zeros_like(left)
    stance = np.ones((3, 2), dtype=bool)
    result = foot_skating_metrics(
        left,
        right,
        stance,
        fps=30,
        threshold_mps=STRICT_FOOT_SKATING_THRESHOLD_MPS,
    )
    assert result["duration"] == 0.0
    assert result["max_velocity"] == 0.0


def test_foot_skating_threshold_override_is_mode_specific() -> None:
    assert resolve_foot_skating_threshold("strict", strict_threshold_mps=0.42) == (
        0.42,
        "m_per_second",
    )
    assert resolve_foot_skating_threshold("legacy", legacy_threshold_m_per_frame=0.02) == (
        0.02,
        "m_per_frame",
    )


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


def test_external_scene_object_scale_is_applied_to_contact_mesh(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><asset><mesh name="box_mesh" file="box.obj" '
        'scale="0.7 0.8 0.9"/></asset><worldbody><body>'
        '<geom name="box" type="mesh" mesh="box_mesh"/>'
        "</body></worldbody></mujoco>"
    )

    assert np.allclose(_object_mesh_scale_from_scene(str(scene), "box"), [0.7, 0.8, 0.9])
