from __future__ import annotations

import numpy as np

from holosoma_retargeting.config_types.retargeter import PhaseWindowConfig
from holosoma_retargeting.src.phase_window_retargeter import (
    CONTACT_ACTIVE,
    CONTACT_APPROACH,
    CONTACT_RELEASE,
    build_contact_phases,
    receding_windows,
    refine_temporal_sequence,
    select_active_collision_indices,
)


def _reference(frames: int) -> np.ndarray:
    qpos = np.zeros((frames, 43), dtype=float)
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.linspace(0.0, 1.0, frames)
    qpos[:, 39] = 1.0
    return qpos


def test_receding_windows_commit_every_frame_once() -> None:
    windows = receding_windows(73, 30, 15)
    committed = [frame for start, _, commit_end in windows for frame in range(start, commit_end)]
    assert committed == list(range(73))
    assert windows[-1][1:] == (73, 73)


def test_temporal_refinement_enforces_velocity_and_joint_bounds() -> None:
    reference = _reference(20)
    result = refine_temporal_sequence(
        reference, 10.0, -np.ones(29), np.ones(29), np.full(29, 0.8),
        config=PhaseWindowConfig(window_frames=8, stride_frames=4, trust_radius=1.0),
    )
    velocity = np.diff(result.qpos[:, 7:36], axis=0) * 10.0
    assert np.max(np.abs(velocity)) <= 0.8 + 1e-6
    assert np.all(result.qpos[:, 7:36] <= 1.0 + 1e-8)
    assert np.all(result.window_id >= 0)
    np.testing.assert_allclose(np.linalg.norm(result.qpos[:, 3:7], axis=1), 1.0)


def test_temporal_refinement_is_fps_invariant() -> None:
    outputs = []
    for fps in (30.0, 60.0, 90.0):
        time = np.arange(int(fps) + 1) / fps
        reference = np.zeros((len(time), 43)); reference[:, 3] = 1.0; reference[:, 39] = 1.0
        reference[:, 7] = 0.2 * time
        result = refine_temporal_sequence(
            reference, fps, -np.ones(29), np.ones(29), np.full(29, 1.0),
            config=PhaseWindowConfig(window_frames=int(fps), stride_frames=max(1, int(fps // 2))),
        )
        outputs.append(np.interp(np.linspace(0, 1, 31), time, result.qpos[:, 7]))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-2)
    np.testing.assert_allclose(outputs[0], outputs[2], atol=1.5e-2)


def test_contact_phase_closes_short_gaps_and_filters_short_runs() -> None:
    contact = np.zeros((20, 1), dtype=bool)
    contact[7:9] = True
    contact[10:14] = True
    contact[18] = True
    cleaned, phase, weights = build_contact_phases(contact)
    assert cleaned[7:14].all()
    assert not cleaned[18, 0]
    assert phase[6, 0] == CONTACT_APPROACH
    assert phase[10, 0] == CONTACT_ACTIVE
    assert phase[14, 0] == CONTACT_RELEASE
    assert weights[10, 0] == 1.0


def test_foot_anchor_constraints_bound_linearized_sliding() -> None:
    frames = 8
    reference = _reference(frames)
    reference[:, 0] = np.linspace(0.0, 0.2, frames)
    positions = np.zeros((frames, 2, 3)); positions[:, :, 0] = reference[:, 0, None]
    jacobians = np.zeros((frames, 2, 3, 32)); jacobians[:, :, 0, 0] = 1.0
    result = refine_temporal_sequence(
        reference, 30.0, -np.ones(29), np.ones(29), np.full(29, 20.0),
        ground_contact=np.ones((frames, 2), dtype=bool), foot_positions=positions,
        foot_jacobians=jacobians, foot_sides=np.asarray([0, 1]),
        config=PhaseWindowConfig(
            window_frames=8, stride_frames=8, trust_radius=0.3,
            reference_weight=0.1, foot_anchor_weight=1e4,
        ),
    )
    predicted_x = positions[..., 0] + (result.qpos[:, 0] - reference[:, 0])[:, None]
    assert np.max(np.abs(predicted_x - predicted_x[0])) <= 0.0081
    assert np.max(result.foot_anchor_cost) < 1e-4


def test_collision_active_set_keeps_penetrating_pairs_and_top_k() -> None:
    phis = np.asarray([-0.01, 0.04, 0.02, 0.03, 0.01])
    groups = np.asarray(["object", "object", "object", "ground", "ground"])
    selected = select_active_collision_indices(phis, groups, top_k=1)
    np.testing.assert_array_equal(selected, [0, 4])


def test_bounded_collision_recovery_restores_small_conflict() -> None:
    reference = _reference(4)
    reference[:, 7] = 0.0
    jacobians = [np.zeros((1, 32)) for _ in range(4)]
    phis = [np.asarray([-0.004]) for _ in range(4)]
    groups = [np.asarray(["object"]) for _ in range(4)]
    result = refine_temporal_sequence(
        reference, 30.0, -np.ones(29), np.ones(29), np.ones(29),
        collision_jacobians=jacobians, collision_phis=phis, collision_groups=groups,
        config=PhaseWindowConfig(window_frames=4, stride_frames=4),
    )
    assert result.collision_recovery_used.all()
    assert np.max(result.collision_recovery_max_slack) >= 0.0029
    assert np.max(result.collision_recovery_max_slack) <= 0.005 + 1e-8
