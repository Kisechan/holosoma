"""Receding-horizon temporal refinement for interaction-mesh retargeting outputs."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from holosoma_retargeting.config_types.retargeter import PhaseWindowConfig


SUCCESS_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
TEMPORAL_QPOS_INDICES = np.asarray([0, 1, 2, *range(7, 36)], dtype=int)
TEMPORAL_JOINT_SLICE = slice(3, 32)


@dataclass(frozen=True)
class PhaseWindowResult:
    qpos: np.ndarray
    window_id: np.ndarray
    window_iteration_count: np.ndarray
    velocity_cost: np.ndarray
    acceleration_cost: np.ndarray


def receding_windows(num_frames: int, window_frames: int, stride_frames: int) -> list[tuple[int, int, int]]:
    """Return (start, end, commit_end) windows covering every frame exactly once at commit time."""
    if num_frames < 1:
        return []
    windows: list[tuple[int, int, int]] = []
    start = 0
    while start < num_frames:
        end = min(num_frames, start + window_frames)
        commit_end = end if end == num_frames else min(end, start + stride_frames)
        windows.append((start, end, commit_end))
        start = commit_end
    return windows


def _scale(values: np.ndarray | None, default: float) -> np.ndarray:
    if values is None:
        return np.full(29, default, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (29,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("temporal scales must contain 29 finite positive values")
    return array


def refine_temporal_sequence(
    reference_qpos: np.ndarray,
    fps: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity_limits: np.ndarray,
    *,
    velocity_scale: np.ndarray | None = None,
    acceleration_scale: np.ndarray | None = None,
    config: PhaseWindowConfig | None = None,
) -> PhaseWindowResult:
    """Refine root translation and 29 joints while preserving quaternion/object trajectories."""
    cfg = config or PhaseWindowConfig()
    reference = np.asarray(reference_qpos, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 36 or len(reference) == 0:
        raise ValueError("reference_qpos must have shape (T, >=36) with at least one frame")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    lower = np.asarray(joint_lower, dtype=float).reshape(-1)
    upper = np.asarray(joint_upper, dtype=float).reshape(-1)
    velocity_limits = np.asarray(joint_velocity_limits, dtype=float).reshape(-1)
    if lower.shape != (29,) or upper.shape != (29,) or velocity_limits.shape != (29,):
        raise ValueError("joint bounds and velocity limits must contain 29 values")
    velocity_norm = _scale(velocity_scale, 1.0)
    acceleration_norm = _scale(acceleration_scale, 10.0)

    working = reference.copy()
    frame_window = np.full(len(reference), -1, dtype=np.int32)
    frame_iterations = np.zeros(len(reference), dtype=np.int32)
    velocity_cost = np.zeros(len(reference), dtype=float)
    acceleration_cost = np.zeros(len(reference), dtype=float)
    for window_index, (start, end, commit_end) in enumerate(
        receding_windows(len(reference), cfg.window_frames, cfg.stride_frames)
    ):
        window_reference = working[start:end, TEMPORAL_QPOS_INDICES]
        state = cp.Variable(window_reference.shape, name=f"phase_window_{window_index}")
        constraints: list[cp.Constraint] = [
            state[:, TEMPORAL_JOINT_SLICE] >= lower,
            state[:, TEMPORAL_JOINT_SLICE] <= upper,
            cp.abs(state - window_reference) <= cfg.trust_radius,
        ]
        if len(window_reference) > 1:
            joint_delta = state[1:, TEMPORAL_JOINT_SLICE] - state[:-1, TEMPORAL_JOINT_SLICE]
            constraints.append(cp.abs(joint_delta) <= velocity_limits / fps)
        if start > 0:
            previous = working[start - 1, TEMPORAL_QPOS_INDICES]
            constraints.append(
                cp.abs(state[0, TEMPORAL_JOINT_SLICE] - previous[TEMPORAL_JOINT_SLICE])
                <= velocity_limits / fps
            )

        terms: list[cp.Expression] = [cfg.reference_weight * cp.sum_squares(state - window_reference)]
        if len(window_reference) > 1:
            joint_velocity = (
                state[1:, TEMPORAL_JOINT_SLICE] - state[:-1, TEMPORAL_JOINT_SLICE]
            ) * fps
            terms.append(
                cfg.velocity_weight
                * cp.sum_squares(cp.multiply(1.0 / velocity_norm, joint_velocity))
            )
        if len(window_reference) > 2:
            joint_acceleration = (
                state[2:, TEMPORAL_JOINT_SLICE]
                - 2 * state[1:-1, TEMPORAL_JOINT_SLICE]
                + state[:-2, TEMPORAL_JOINT_SLICE]
            ) * fps * fps
            terms.append(
                cfg.acceleration_weight
                * cp.sum_squares(cp.multiply(1.0 / acceleration_norm, joint_acceleration))
            )
        problem = cp.Problem(cp.Minimize(cp.sum(terms)), constraints)
        problem.solve(solver=cp.CLARABEL)
        if problem.status not in SUCCESS_STATUSES or state.value is None:
            raise RuntimeError(f"phase window {window_index} solve failed: {problem.status}")
        solved = np.asarray(state.value, dtype=float)
        working[start:end, TEMPORAL_QPOS_INDICES] = solved
        committed = slice(start, commit_end)
        frame_window[committed] = window_index
        frame_iterations[committed] = 1
        if commit_end - start > 1:
            velocity = np.diff(solved[: commit_end - start, TEMPORAL_JOINT_SLICE], axis=0) * fps
            velocity_cost[start + 1:commit_end] = np.mean((velocity / velocity_norm) ** 2, axis=1)
        if commit_end - start > 2:
            acceleration = np.diff(solved[: commit_end - start, TEMPORAL_JOINT_SLICE], n=2, axis=0) * fps * fps
            acceleration_cost[start + 2:commit_end] = np.mean((acceleration / acceleration_norm) ** 2, axis=1)
    working[:, 3:7] /= np.maximum(np.linalg.norm(working[:, 3:7], axis=1, keepdims=True), 1e-12)
    return PhaseWindowResult(working, frame_window, frame_iterations, velocity_cost, acceleration_cost)
