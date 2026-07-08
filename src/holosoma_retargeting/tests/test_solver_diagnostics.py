from __future__ import annotations

import cvxpy as cp

from holosoma_retargeting.src.solver_diagnostics import (
    SUCCESS_STATUSES,
    RetargetingSolveError,
    diagnose_constraint_groups,
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
