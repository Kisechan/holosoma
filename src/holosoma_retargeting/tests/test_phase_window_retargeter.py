from __future__ import annotations

import numpy as np

from holosoma_retargeting.config_types.retargeter import PhaseWindowConfig
from holosoma_retargeting.src.phase_window_retargeter import receding_windows, refine_temporal_sequence


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
