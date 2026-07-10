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
    contact_cost: np.ndarray
    foot_anchor_cost: np.ndarray


CONTACT_INACTIVE = 0
CONTACT_APPROACH = 1
CONTACT_ACTIVE = 2
CONTACT_RELEASE = 3


def build_contact_phases(
    contact: np.ndarray,
    *,
    closing_frames: int = 2,
    minimum_active_frames: int = 3,
    ramp_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean binary contact and create inactive/approach/active/release phases and weights."""
    values = np.asarray(contact, dtype=bool)
    if values.ndim != 2:
        raise ValueError("contact must have shape (T, C)")
    cleaned = values.copy()
    for channel in range(values.shape[1]):
        active = np.flatnonzero(cleaned[:, channel])
        for left, right in zip(active[:-1], active[1:]):
            if 1 < right - left <= closing_frames + 1:
                cleaned[left:right + 1, channel] = True
        starts = np.flatnonzero(cleaned[:, channel] & ~np.r_[False, cleaned[:-1, channel]])
        ends = np.flatnonzero(cleaned[:, channel] & ~np.r_[cleaned[1:, channel], False])
        for start, end in zip(starts, ends):
            if end - start + 1 < minimum_active_frames:
                cleaned[start:end + 1, channel] = False

    phase = np.full(values.shape, CONTACT_INACTIVE, dtype=np.uint8)
    weights = np.zeros(values.shape, dtype=float)
    phase[cleaned] = CONTACT_ACTIVE
    weights[cleaned] = 1.0
    for channel in range(values.shape[1]):
        starts = np.flatnonzero(cleaned[:, channel] & ~np.r_[False, cleaned[:-1, channel]])
        ends = np.flatnonzero(cleaned[:, channel] & ~np.r_[cleaned[1:, channel], False])
        for start, end in zip(starts, ends):
            approach_start = max(0, start - ramp_frames)
            approach_indices = np.arange(approach_start, start)
            if len(approach_indices):
                phase[approach_indices, channel] = CONTACT_APPROACH
                u = np.arange(1, len(approach_indices) + 1) / (len(approach_indices) + 1)
                weights[approach_indices, channel] = 0.5 - 0.5 * np.cos(np.pi * u)
            release_end = min(len(cleaned), end + ramp_frames + 1)
            release_indices = np.arange(end + 1, release_end)
            if len(release_indices):
                phase[release_indices, channel] = CONTACT_RELEASE
                u = np.arange(len(release_indices), 0, -1) / (len(release_indices) + 1)
                weights[release_indices, channel] = 0.5 - 0.5 * np.cos(np.pi * u)
    return cleaned, phase, weights


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
    ground_contact: np.ndarray | None = None,
    foot_positions: np.ndarray | None = None,
    foot_jacobians: np.ndarray | None = None,
    foot_sides: np.ndarray | None = None,
    contact_positions: np.ndarray | None = None,
    contact_jacobians: np.ndarray | None = None,
    contact_targets: np.ndarray | None = None,
    contact_weights: np.ndarray | None = None,
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
    ground = None if ground_contact is None else np.asarray(ground_contact, dtype=bool)
    foot_pos = None if foot_positions is None else np.asarray(foot_positions, dtype=float)
    foot_jac = None if foot_jacobians is None else np.asarray(foot_jacobians, dtype=float)
    sides = None if foot_sides is None else np.asarray(foot_sides, dtype=int)
    if any(value is not None for value in (ground, foot_pos, foot_jac, sides)):
        if ground is None or foot_pos is None or foot_jac is None or sides is None:
            raise ValueError("ground_contact, foot_positions, foot_jacobians, and foot_sides are required together")
        if ground.shape != (len(reference), 2) or foot_pos.shape[:1] != (len(reference),):
            raise ValueError("ground/foot arrays must match the reference frame count")
        if foot_jac.shape != (*foot_pos.shape, len(TEMPORAL_QPOS_INDICES)) or sides.shape != (foot_pos.shape[1],):
            raise ValueError("foot Jacobian or side shape is invalid")
    contact_pos = None if contact_positions is None else np.asarray(contact_positions, dtype=float)
    contact_jac = None if contact_jacobians is None else np.asarray(contact_jacobians, dtype=float)
    targets = None if contact_targets is None else np.asarray(contact_targets, dtype=float)
    weights = None if contact_weights is None else np.asarray(contact_weights, dtype=float)
    if any(value is not None for value in (contact_pos, contact_jac, targets, weights)):
        if contact_pos is None or contact_jac is None or targets is None or weights is None:
            raise ValueError("contact positions, Jacobians, targets, and weights are required together")
        if contact_pos.shape != targets.shape or weights.shape != contact_pos.shape[:2]:
            raise ValueError("contact target/weight shape is invalid")
        if contact_jac.shape != (*contact_pos.shape, len(TEMPORAL_QPOS_INDICES)):
            raise ValueError("contact Jacobian shape is invalid")

    foot_anchors = None
    if ground is not None and foot_pos is not None and sides is not None:
        foot_anchors = np.full_like(foot_pos, np.nan)
        for point, side in enumerate(sides):
            starts = np.flatnonzero(ground[:, side] & ~np.r_[False, ground[:-1, side]])
            ends = np.flatnonzero(ground[:, side] & ~np.r_[ground[1:, side], False])
            for start, end in zip(starts, ends):
                foot_anchors[start:end + 1, point] = foot_pos[start, point]

    working = reference.copy()
    frame_window = np.full(len(reference), -1, dtype=np.int32)
    frame_iterations = np.zeros(len(reference), dtype=np.int32)
    velocity_cost = np.zeros(len(reference), dtype=float)
    acceleration_cost = np.zeros(len(reference), dtype=float)
    contact_cost = np.zeros(len(reference), dtype=float)
    foot_anchor_cost = np.zeros(len(reference), dtype=float)
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
        delta = state - window_reference
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
        if foot_anchors is not None and foot_pos is not None and foot_jac is not None and sides is not None:
            for local_frame, frame in enumerate(range(start, end)):
                for point, side in enumerate(sides):
                    if not ground[frame, side]:
                        continue
                    predicted = foot_pos[frame, point] + foot_jac[frame, point] @ delta[local_frame]
                    anchor = foot_anchors[frame, point]
                    slack = cp.Variable(nonneg=True, name=f"foot_slack_{window_index}_{local_frame}_{point}")
                    constraints.extend([
                        slack <= cfg.foot_slack_limit,
                        cp.abs(predicted[:2] - anchor[:2]) <= cfg.foot_xy_tolerance + slack,
                        predicted[2] >= cfg.foot_z_lower,
                        predicted[2] <= cfg.foot_z_upper,
                    ])
                    terms.append(cfg.foot_anchor_weight * cp.sum_squares(predicted[:2] - anchor[:2]))
        if contact_pos is not None and contact_jac is not None and targets is not None and weights is not None:
            predicted_contacts: dict[tuple[int, int], cp.Expression] = {}
            for local_frame, frame in enumerate(range(start, end)):
                for channel in range(contact_pos.shape[1]):
                    weight = float(weights[frame, channel])
                    if weight <= 0 or not np.all(np.isfinite(targets[frame, channel])):
                        continue
                    predicted = contact_pos[frame, channel] + contact_jac[frame, channel] @ delta[local_frame]
                    predicted_contacts[(local_frame, channel)] = predicted
                    terms.append(cfg.contact_position_weight * weight * cp.sum_squares(predicted - targets[frame, channel]))
                    previous_key = (local_frame - 1, channel)
                    if previous_key in predicted_contacts and np.all(np.isfinite(targets[frame - 1, channel])):
                        target_delta = targets[frame, channel] - targets[frame - 1, channel]
                        terms.append(
                            cfg.contact_velocity_weight * min(weight, float(weights[frame - 1, channel]))
                            * cp.sum_squares(predicted - predicted_contacts[previous_key] - target_delta)
                        )
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
    final_delta = working[:, TEMPORAL_QPOS_INDICES] - reference[:, TEMPORAL_QPOS_INDICES]
    if foot_anchors is not None and foot_pos is not None and foot_jac is not None and sides is not None:
        predicted = foot_pos + np.einsum("tpcd,td->tpc", foot_jac, final_delta)
        for frame in range(len(reference)):
            errors = [
                np.linalg.norm(predicted[frame, point, :2] - foot_anchors[frame, point, :2])
                for point, side in enumerate(sides) if ground[frame, side]
            ]
            foot_anchor_cost[frame] = float(np.mean(np.square(errors))) if errors else 0.0
    if contact_pos is not None and contact_jac is not None and targets is not None and weights is not None:
        predicted = contact_pos + np.einsum("tpcd,td->tpc", contact_jac, final_delta)
        for frame in range(len(reference)):
            active = (weights[frame] > 0) & np.all(np.isfinite(targets[frame]), axis=1)
            if np.any(active):
                contact_cost[frame] = float(np.mean(np.square(predicted[frame, active] - targets[frame, active])))
    return PhaseWindowResult(
        working, frame_window, frame_iterations, velocity_cost, acceleration_cost,
        contact_cost, foot_anchor_cost,
    )
