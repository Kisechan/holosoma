from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from holosoma_retargeting.src.solver_diagnostics import (
    SUCCESS_STATUSES,
    RetargetingSolveError,
    build_collision_constraint,
    diagnose_constraint_groups,
    summarize_slack_variables,
    write_failure_artifacts,
)


def _is_success(status: str) -> bool:
    return status in SUCCESS_STATUSES


def test_joint_limit_and_trust_region_conflict_is_attributed() -> None:
    delta = cp.Variable(1)
    result = diagnose_constraint_groups(
        [],
        {
            "joint_limits": [delta >= 2.0],
            "trust_region": [cp.SOC(1.0, delta)],
        },
    )

    assert _is_success(result["ablation_status"]["joint_limits"])
    assert _is_success(result["ablation_status"]["trust_region"])
    assert _is_success(result["elastic_status"])
    total_slack = sum(item["sum_slack"] for item in result["group_slack"].values())
    assert total_slack >= 0.99


def test_foot_and_collision_conflict_is_attributed() -> None:
    delta = cp.Variable(1)
    result = diagnose_constraint_groups(
        [],
        {
            "foot_sticking": [delta <= 0.0],
            "object_ground_collision": [delta >= 1.0],
        },
    )

    assert _is_success(result["ablation_status"]["foot_sticking"])
    assert _is_success(result["ablation_status"]["object_ground_collision"])
    total_slack = sum(item["sum_slack"] for item in result["group_slack"].values())
    assert total_slack >= 0.99


def test_solve_error_accepts_frame_context() -> None:
    error = RetargetingSolveError("infeasible", frame_idx=12, sqp_iteration=3)
    error.add_context(completed_frames=12)

    assert error.frame_idx == 12
    assert error.sqp_iteration == 3
    assert error.diagnostics["completed_frames"] == 12
    assert "frame 12" in str(error)


def test_soft_collision_slack_restores_feasibility() -> None:
    delta = cp.Variable(1)
    constraints, objective, slack = build_collision_constraint(
        delta, 1.0, mode="soft", slack_weight=100.0, name="collision_slack"
    )
    problem = cp.Problem(cp.Minimize(objective), [delta == 0.0, *constraints])
    problem.solve(solver=cp.CLARABEL)

    assert _is_success(problem.status)
    assert slack is not None
    assert float(np.asarray(slack.value)) >= 0.99
    summary = summarize_slack_variables({"robot_object_collision": [slack]})
    assert summary["robot_object_collision"]["max_slack"] >= 0.99


def test_hard_collision_keeps_infeasible_conflict() -> None:
    delta = cp.Variable(1)
    constraints, objective, slack = build_collision_constraint(delta, 1.0, mode="hard", slack_weight=100.0)
    problem = cp.Problem(cp.Minimize(0), [delta == 0.0, *constraints])
    problem.solve(solver=cp.CLARABEL)

    assert problem.status == cp.INFEASIBLE
    assert objective is None
    assert slack is None


def test_collision_constraint_mode_validation() -> None:
    delta = cp.Variable(1)
    with pytest.raises(ValueError, match="collision_constraint_mode"):
        build_collision_constraint(delta, 1.0, mode="elastic", slack_weight=100.0)


def test_failure_artifacts_preserve_diagnostics_and_partial_trajectory(tmp_path) -> None:
    error = RetargetingSolveError(
        "infeasible",
        frame_idx=4,
        sqp_iteration=2,
        diagnostics={
            "sequence": "sub12_largebox_000",
            "augmentation": "original",
            "completed_frames": 4,
            "total_frames": 10,
            "partial_qpos": np.zeros((4, 43)),
            "constraint_group_counts": {"foot_sticking": 4, "joint_limits": 58},
            "group_slack": {"foot_sticking": {"max_slack": np.float64(0.003)}},
        },
    )
    failure_path = tmp_path / "sequence_original_failure.json"
    partial_path = tmp_path / "sequence_original_partial.npz"

    payload = write_failure_artifacts(error, failure_path, partial_path=partial_path)

    assert failure_path.exists()
    assert partial_path.exists()
    assert payload["first_infeasible_frame"] == 4
    assert payload["completed_frame_ratio"] == 0.4
    assert payload["diagnostics"]["group_slack"]["foot_sticking"]["max_slack"] == 0.003
    with np.load(partial_path) as partial:
        assert partial["qpos"].shape == (4, 43)
