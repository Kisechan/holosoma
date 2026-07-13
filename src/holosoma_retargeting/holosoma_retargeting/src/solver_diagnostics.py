"""Structured diagnostics for infeasible retargeting subproblems."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cvxpy as cp  # type: ignore[import-not-found]
import numpy as np

SUCCESS_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
COLLISION_CONSTRAINT_MODES = {"hard", "soft"}
FEASIBILITY_RECOVERY_MODES = {"off", "fixed", "adaptive"}


class RetargetingSolveError(RuntimeError):
    """A solver failure with enough state for an experiment runner to persist it."""

    def __init__(
        self,
        status: str,
        frame_idx: int,
        sqp_iteration: int,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.status = str(status)
        self.frame_idx = int(frame_idx)
        self.sqp_iteration = int(sqp_iteration)
        self.diagnostics: dict[str, Any] = diagnostics or {}
        super().__init__(
            f"CVXPY solve failed: {self.status} at frame {self.frame_idx}, "
            f"SQP iteration {self.sqp_iteration}"
        )

    def add_context(self, **values: Any) -> None:
        """Attach frame-level state while preserving the original solver evidence."""
        self.diagnostics.update(values)


def _jsonable(value: Any) -> Any:
    """Convert diagnostic values to deterministic JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_failure_artifacts(
    error: RetargetingSolveError,
    failure_path: Path,
    *,
    partial_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one solver failure and an optional partial trajectory sidecar."""
    failure_path = Path(failure_path)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(error.diagnostics)
    partial_qpos = diagnostics.pop("partial_qpos", None)
    if partial_qpos is not None and partial_path is not None:
        partial = np.asarray(partial_qpos, dtype=float)
        partial_path = Path(partial_path)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_payload = {
            "qpos": partial,
            "completed_frames": np.asarray(len(partial), dtype=np.int32),
        }
        if "fps" in diagnostics:
            partial_payload["fps"] = np.asarray(diagnostics["fps"], dtype=np.float32)
        np.savez(partial_path, **partial_payload)

    repo_root = Path(__file__).resolve().parents[5]
    completed_frames = int(diagnostics.get("completed_frames", 0))
    total_frames = int(diagnostics.get("total_frames", 0))
    payload = {
        "schema_version": "1.0.0",
        "sequence": diagnostics.get("sequence"),
        "augmentation": diagnostics.get("augmentation", "original"),
        "retarget_success": False,
        "completed_frames": completed_frames,
        "total_frames": total_frames,
        "completed_frame_ratio": completed_frames / total_frames if total_frames else 0.0,
        "first_infeasible_frame": error.frame_idx,
        "first_infeasible_sqp_iteration": error.sqp_iteration,
        "clarabel_status": error.status,
        "error": str(error),
        "diagnostics": _jsonable(diagnostics),
        "artifacts": {
            "failure_json": str(failure_path),
            "partial_npz": str(partial_path) if partial_qpos is not None and partial_path is not None else None,
        },
        "provenance": {
            **_jsonable(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "root_git_commit": _git_commit(repo_root),
            "holosoma_git_commit": _git_commit(repo_root / "holosoma"),
        },
    }
    failure_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def constraint_size(constraint: cp.Constraint) -> int:
    """Return the number of scalar relations represented by a CVXPY constraint."""
    if isinstance(constraint, cp.constraints.second_order.SOC):
        return 1
    return int(np.prod(constraint.shape or (1,)))


def validate_collision_constraint_mode(mode: str) -> str:
    """Return a normalized collision constraint mode or raise a stable error."""
    normalized = str(mode).lower()
    if normalized not in COLLISION_CONSTRAINT_MODES:
        raise ValueError(f"collision_constraint_mode must be one of {sorted(COLLISION_CONSTRAINT_MODES)}")
    return normalized


def validate_feasibility_recovery_mode(mode: str) -> str:
    """Return a normalized recovery mode used by the three M3 ablations."""
    normalized = str(mode).lower()
    if normalized not in FEASIBILITY_RECOVERY_MODES:
        raise ValueError(f"feasibility_recovery_mode must be one of {sorted(FEASIBILITY_RECOVERY_MODES)}")
    return normalized


def build_collision_constraint(
    expr: cp.Expression,
    rhs: float,
    mode: str,
    slack_weight: float,
    max_slack: float | None = None,
    name: str = "collision_slack",
) -> tuple[list[cp.Constraint], cp.Expression | None, cp.Variable | None]:
    """Build a hard or elastic linearized non-penetration constraint."""
    normalized = validate_collision_constraint_mode(mode)
    if normalized == "hard":
        return [expr >= rhs], None, None

    if slack_weight <= 0:
        raise ValueError(f"collision_slack_weight must be positive, got {slack_weight}")
    if max_slack is not None and max_slack < 0:
        raise ValueError(f"collision_max_slack must be non-negative or None, got {max_slack}")

    slack = cp.Variable(nonneg=True, name=name)
    constraints: list[cp.Constraint] = [expr + slack >= rhs]
    if max_slack is not None:
        constraints.append(slack <= max_slack)
    return constraints, float(slack_weight) * cp.sum_squares(slack), slack


def build_interval_constraint(
    expr: cp.Expression,
    lower: np.ndarray,
    upper: np.ndarray,
    mode: str,
    slack_weight: float,
    max_slack: float | None = None,
    name: str = "interval_slack",
) -> tuple[list[cp.Constraint], cp.Expression | None, cp.Variable | None]:
    """Build a hard or elastic vector interval with one nonnegative slack per axis."""
    normalized = validate_collision_constraint_mode(mode)
    if normalized == "hard":
        return [expr >= lower, expr <= upper], None, None
    if slack_weight <= 0:
        raise ValueError(f"slack_weight must be positive, got {slack_weight}")
    if max_slack is not None and max_slack < 0:
        raise ValueError(f"max_slack must be non-negative or None, got {max_slack}")
    slack = cp.Variable(expr.shape, nonneg=True, name=name)
    constraints: list[cp.Constraint] = [expr + slack >= lower, expr - slack <= upper]
    if max_slack is not None:
        constraints.append(slack <= max_slack)
    return constraints, float(slack_weight) * cp.sum_squares(slack), slack


def summarize_slack_variables(slack_variables: dict[str, list[cp.Variable]]) -> dict[str, dict[str, float | int]]:
    """Summarize solved slack variables by constraint group."""
    summary: dict[str, dict[str, float | int]] = {}
    for group, variables in slack_variables.items():
        values: list[np.ndarray] = []
        for variable in variables:
            if variable.value is None:
                continue
            values.append(np.asarray(variable.value, dtype=float).reshape(-1))
        flat = np.concatenate(values) if values else np.zeros(0, dtype=float)
        summary[group] = {
            "count": len(variables),
            "max_slack": float(np.max(flat, initial=0.0)),
            "mean_slack": float(np.mean(flat)) if flat.size else 0.0,
            "sum_slack": float(np.sum(flat)),
        }
    return summary


def update_trust_region(
    radius: float,
    predicted_residual: float,
    actual_residual: float,
    *,
    min_radius: float,
    max_radius: float,
    shrink_factor: float = 0.5,
    grow_factor: float = 1.25,
    acceptance_ratio: float = 0.25,
) -> float:
    """Update a trust radius from predicted and measured nonlinear residuals."""
    values = (radius, predicted_residual, actual_residual, min_radius, max_radius)
    if not all(np.isfinite(value) for value in values):
        action = "shrink"
    else:
        lower = predicted_residual * (1.0 - acceptance_ratio)
        upper = predicted_residual * (1.0 + acceptance_ratio)
        if actual_residual > upper:
            action = "shrink"
        elif actual_residual < lower:
            action = "grow"
        else:
            action = "keep"
    if action == "shrink":
        new_radius = max(min_radius, radius * shrink_factor)
    elif action == "grow":
        new_radius = min(max_radius, radius * grow_factor)
    else:
        new_radius = radius
    return float(new_radius)


def _solve_status(constraints: list[cp.Constraint]) -> str:
    try:
        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=cp.CLARABEL)
        return str(problem.status)
    except Exception as exc:  # pragma: no cover - solver backend failures vary
        return f"error:{type(exc).__name__}:{exc}"


def _slack_variable(constraint: cp.Constraint, name: str) -> cp.Variable:
    if isinstance(constraint, cp.constraints.second_order.SOC) or constraint.shape == ():
        return cp.Variable(nonneg=True, name=name)
    return cp.Variable(constraint.shape, nonneg=True, name=name)


def _relax_constraint(constraint: cp.Constraint, slack: cp.Variable) -> list[cp.Constraint]:
    if isinstance(constraint, cp.constraints.nonpos.Inequality):
        return [constraint.expr <= slack]
    if isinstance(constraint, cp.constraints.zero.Equality):
        return [constraint.expr <= slack, -constraint.expr <= slack]
    if isinstance(constraint, cp.constraints.second_order.SOC):
        return [cp.SOC(constraint.args[0] + slack, constraint.args[1], axis=constraint.axis)]
    raise TypeError(f"unsupported diagnostic constraint: {type(constraint).__name__}")


def diagnose_constraint_groups(
    base_constraints: list[cp.Constraint],
    constraint_groups: dict[str, list[cp.Constraint]],
) -> dict[str, Any]:
    """Attribute infeasibility with leave-one-group-out and elastic feasibility solves."""
    active = {name: values for name, values in constraint_groups.items() if values}
    counts = {
        name: int(sum(constraint_size(constraint) for constraint in values))
        for name, values in constraint_groups.items()
    }
    ablation_status: dict[str, str] = {}
    for omitted in active:
        remaining = list(base_constraints)
        for name, values in active.items():
            if name != omitted:
                remaining.extend(values)
        ablation_status[omitted] = _solve_status(remaining)

    elastic_constraints = list(base_constraints)
    slacks: dict[str, list[cp.Variable]] = {name: [] for name in active}
    objective_terms: list[cp.Expression] = []
    for group_index, (name, values) in enumerate(active.items()):
        for constraint_index, constraint in enumerate(values):
            slack = _slack_variable(constraint, f"diag_s_{group_index}_{constraint_index}")
            elastic_constraints.extend(_relax_constraint(constraint, slack))
            slacks[name].append(slack)
            objective_terms.append(cp.sum(slack))

    elastic_status = "not_run"
    group_slack: dict[str, dict[str, float]] = {
        name: {
            "max_slack": float("nan"),
            "mean_slack": float("nan"),
            "sum_slack": float("nan"),
        }
        for name in active
    }
    if objective_terms:
        try:
            problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), elastic_constraints)
            problem.solve(solver=cp.CLARABEL)
            elastic_status = str(problem.status)
            if problem.status in SUCCESS_STATUSES:
                for name, variables in slacks.items():
                    values = np.concatenate(
                        [np.asarray(variable.value, dtype=float).reshape(-1) for variable in variables]
                    )
                    group_slack[name] = {
                        "max_slack": float(np.max(values, initial=0.0)),
                        "mean_slack": float(np.mean(values)) if values.size else 0.0,
                        "sum_slack": float(np.sum(values)),
                    }
        except Exception as exc:  # pragma: no cover - solver backend failures vary
            elastic_status = f"error:{type(exc).__name__}:{exc}"

    return {
        "constraint_group_counts": counts,
        "ablation_status": ablation_status,
        "elastic_status": elastic_status,
        "group_slack": group_slack,
    }
