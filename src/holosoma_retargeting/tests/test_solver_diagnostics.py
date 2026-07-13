from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from holosoma_retargeting.src.solver_diagnostics import (
    SUCCESS_STATUSES,
    RetargetingSolveError,
    build_collision_constraint,
    build_interval_constraint,
    diagnose_constraint_groups,
    summarize_slack_variables,
    update_trust_region,
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
    assert all("mean_slack" in item for item in result["group_slack"].values())


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


def test_soft_foot_interval_slack_is_independent_from_collision_slack() -> None:
    delta = cp.Variable(2)
    collision_constraints, collision_objective, collision_slack = build_collision_constraint(
        delta[0], 1.0, mode="soft", slack_weight=100.0, name="collision_slack"
    )
    foot_constraints, foot_objective, foot_slack = build_interval_constraint(
        delta[1],
        lower=0.5,
        upper=0.75,
        mode="soft",
        slack_weight=10.0,
        name="foot_sticking_slack",
    )
    problem = cp.Problem(
        cp.Minimize(collision_objective + foot_objective),
        [delta == 0.0, *collision_constraints, *foot_constraints],
    )
    problem.solve(solver=cp.CLARABEL)

    assert _is_success(problem.status)
    assert collision_slack is not None
    assert foot_slack is not None
    assert collision_slack is not foot_slack
    assert float(np.asarray(collision_slack.value)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.asarray(foot_slack.value)) == pytest.approx(0.5, abs=1e-5)

    summary = summarize_slack_variables(
        {
            "robot_object_collision": [collision_slack],
            "foot_sticking": [foot_slack],
        }
    )
    assert summary["robot_object_collision"]["max_slack"] == pytest.approx(1.0, abs=1e-5)
    assert summary["foot_sticking"]["max_slack"] == pytest.approx(0.5, abs=1e-5)


def test_hard_foot_interval_preserves_infeasible_conflict() -> None:
    delta = cp.Variable(1)
    constraints, objective, slack = build_interval_constraint(
        delta,
        lower=np.asarray([0.5]),
        upper=np.asarray([0.75]),
        mode="hard",
        slack_weight=10.0,
        name="foot_sticking_slack",
    )
    problem = cp.Problem(cp.Minimize(0), [delta == 0.0, *constraints])
    problem.solve(solver=cp.CLARABEL)

    assert problem.status == cp.INFEASIBLE
    assert objective is None
    assert slack is None


def test_soft_foot_interval_honors_max_slack() -> None:
    delta = cp.Variable(1)
    constraints, objective, slack = build_interval_constraint(
        delta,
        lower=np.asarray([0.5]),
        upper=np.asarray([0.75]),
        mode="soft",
        slack_weight=10.0,
        max_slack=0.25,
        name="foot_sticking_slack",
    )
    problem = cp.Problem(cp.Minimize(objective), [delta == 0.0, *constraints])
    problem.solve(solver=cp.CLARABEL)

    assert problem.status == cp.INFEASIBLE
    assert slack is not None


@pytest.mark.parametrize(
    ("predicted_residual", "actual_residual", "expected_radius"),
    [
        (0.02, 0.04, 0.1),
        (0.02, 0.021, 0.2),
        (0.02, 0.005, 0.4),
    ],
)
def test_adaptive_trust_region_scales_from_prediction_quality(
    predicted_residual: float,
    actual_residual: float,
    expected_radius: float,
) -> None:
    radius = update_trust_region(
        radius=0.2,
        predicted_residual=predicted_residual,
        actual_residual=actual_residual,
        min_radius=0.05,
        max_radius=0.5,
        shrink_factor=0.5,
        grow_factor=2.0,
        acceptance_ratio=0.25,
    )

    assert radius == pytest.approx(expected_radius)


def test_adaptive_trust_region_clamps_to_bounds() -> None:
    shrunk = update_trust_region(
        radius=0.06,
        predicted_residual=0.01,
        actual_residual=0.1,
        min_radius=0.05,
        max_radius=0.5,
        shrink_factor=0.5,
        grow_factor=2.0,
        acceptance_ratio=0.25,
    )
    grown = update_trust_region(
        radius=0.4,
        predicted_residual=0.1,
        actual_residual=0.0,
        min_radius=0.05,
        max_radius=0.5,
        shrink_factor=0.5,
        grow_factor=2.0,
        acceptance_ratio=0.25,
    )

    assert shrunk == pytest.approx(0.05)
    assert grown == pytest.approx(0.5)


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
