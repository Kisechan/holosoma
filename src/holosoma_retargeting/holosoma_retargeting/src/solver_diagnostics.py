"""Structured diagnostics for infeasible retargeting subproblems."""

from __future__ import annotations

from typing import Any

import cvxpy as cp  # type: ignore[import-not-found]
import numpy as np


SUCCESS_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
COLLISION_CONSTRAINT_MODES = {"hard", "soft"}


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
            "count": int(len(variables)),
            "max_slack": float(np.max(flat, initial=0.0)),
            "sum_slack": float(np.sum(flat)),
        }
    return summary


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
        name: {"max_slack": float("nan"), "sum_slack": float("nan")} for name in active
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
