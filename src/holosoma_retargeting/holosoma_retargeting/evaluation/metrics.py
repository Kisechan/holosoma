"""Pure metric helpers shared by retargeting evaluation and benchmark audits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PENETRATION_TOLERANCES_M = (0.0, 0.002, 0.005, 0.010)
OMNICONTACT_ORDER = ("left_ankle", "right_ankle", "left_wrist", "right_wrist")
STRICT_FOOT_SKATING_THRESHOLD_MPS = 0.3
LEGACY_FOOT_SKATING_THRESHOLD_M_PER_FRAME = 0.01


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_foot_skating_threshold(
    metric_mode: str,
    *,
    strict_threshold_mps: float = STRICT_FOOT_SKATING_THRESHOLD_MPS,
    legacy_threshold_m_per_frame: float = LEGACY_FOOT_SKATING_THRESHOLD_M_PER_FRAME,
) -> tuple[float, str]:
    """Return the active foot-skating threshold and its physical unit."""
    if metric_mode == "strict":
        threshold = float(strict_threshold_mps)
        unit = "m_per_second"
    elif metric_mode == "legacy":
        threshold = float(legacy_threshold_m_per_frame)
        unit = "m_per_frame"
    else:
        raise ValueError(f"unsupported metric mode: {metric_mode}")
    if threshold <= 0:
        raise ValueError("foot-skating threshold must be positive")
    return threshold, unit


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize_penetration_depths(
    frame_depths_m: Sequence[float], tolerances_m: Sequence[float] = PENETRATION_TOLERANCES_M
) -> dict[str, float]:
    depths = np.asarray(frame_depths_m, dtype=float)
    total = max(int(depths.size), 1)
    result = {
        f"tol_{round(tol * 1000)}mm": float(np.count_nonzero(depths > tol) / total)
        for tol in tolerances_m
    }
    positive = depths[depths > 0]
    result.update(
        {
            "p95": float(np.percentile(positive, 95)) if positive.size else 0.0,
            "p99": float(np.percentile(positive, 99)) if positive.size else 0.0,
            "max": float(np.max(positive)) if positive.size else 0.0,
        }
    )
    return result


def foot_skating_metrics(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    stance: np.ndarray,
    *,
    fps: float,
    threshold_mps: float,
    legacy: bool = False,
) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    stance = np.asarray(stance, dtype=bool)
    if stance.ndim != 2 or stance.shape[1] != 2:
        raise ValueError("stance must have shape (T, 2)")
    scale = 1.0 if legacy else float(fps)
    left_v = np.r_[0.0, np.linalg.norm(np.diff(left_xy, axis=0), axis=1)] * scale
    right_v = np.r_[0.0, np.linalg.norm(np.diff(right_xy, axis=0), axis=1)] * scale
    velocities = np.column_stack((left_v, right_v))
    active = stance[: len(velocities)]
    skating = active & (velocities > threshold_mps)
    denom = int(np.count_nonzero(np.any(active, axis=1)))
    per_frame_max = np.max(np.where(skating, velocities, 0.0), axis=1)
    return {
        "duration": float(np.count_nonzero(per_frame_max > 0) / denom) if denom else 0.0,
        "max_velocity": float(np.max(per_frame_max)) if per_frame_max.size else 0.0,
        "stance_frames": denom,
        "threshold_mps": float(threshold_mps),
        "fps": float(fps),
    }


def binary_contact_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | bool]:
    target = np.asarray(target, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if target.shape != predicted.shape:
        raise ValueError(f"contact shapes differ: {target.shape} != {predicted.shape}")
    tp = int(np.count_nonzero(target & predicted))
    fp = int(np.count_nonzero(~target & predicted))
    fn = int(np.count_nonzero(target & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    switches = int(np.count_nonzero(np.diff(predicted.astype(np.int8), axis=0))) if predicted.shape[0] > 1 else 0
    return {
        "available": True,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_switches": switches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_omnicontact_labels(
    root: Path,
    sequence: str,
    target_frames: int,
    target_fps: float,
) -> tuple[np.ndarray | None, str | None]:
    capture_tokens = sequence.split("omnicontact_", 1)
    if len(capture_tokens) != 2:
        return None, "omnicontact capture id missing from sequence name"
    capture_id = capture_tokens[1].rsplit("_original", 1)[0]
    matches = list(root.rglob(f"{capture_id}_with_contact.npz"))
    if len(matches) != 1:
        return None, f"expected one GT file for {capture_id}, found {len(matches)}"
    with np.load(matches[0], allow_pickle=False) as archive:
        if "contact_info" not in archive:
            return None, "contact_info missing"
        labels = np.asarray(archive["contact_info"]).squeeze()
        source_fps = float(np.asarray(archive.get("fps", target_fps)).reshape(-1)[0])
    if labels.ndim != 2 or labels.shape[1] != len(OMNICONTACT_ORDER):
        return None, f"unsupported contact_info shape {labels.shape}"
    indices = np.minimum(
        np.round(np.arange(target_frames) * source_fps / target_fps).astype(int), labels.shape[0] - 1
    )
    return labels[indices].astype(bool), str(matches[0])
