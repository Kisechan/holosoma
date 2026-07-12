"""
Evaluation script for retargeting trajectories.
Evaluates:
1) Penetration depth & time duration
2) Contact precision (keypoints <=2cm from object/terrain surface)
3) Foot sliding
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Sequence

import igl  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    SMPLH_DEMO_JOINTS,
    MotionDataConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.evaluation.metrics import (  # noqa: E402
    PENETRATION_TOLERANCES_M,
    binary_contact_metrics,
    canonical_config_hash,
    foot_skating_metrics,
    git_commit,
    load_omnicontact_labels,
    summarize_penetration_depths,
)
from holosoma_retargeting.src.mujoco_utils import _world_mesh_from_geom  # type: ignore[import-not-found]  # noqa: E402
from holosoma_retargeting.src.utils import (  # type: ignore[import-not-found]  # noqa: E402
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_xml,
    extract_foot_sticking_sequence_velocity,
    load_intermimic_data,
    preprocess_motion_data,
    transform_points_world_to_local,
    transform_y_up_to_z_up,
)


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    *,
    object_name: str | None = None,
    object_dir: str | None = None,
) -> SimpleNamespace:
    """Create a mutable namespace that mimics the old constants modules."""
    namespace = SimpleNamespace()

    # Copy UPPER_CASE attributes from robot config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(namespace, attr, getattr(robot_config, attr))

    # Copy legacy constants from motion data config
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(namespace, attr, value)

    # Override or supplement object information if requested
    if object_name is not None:
        namespace.OBJECT_NAME = object_name

    # Provide default object asset paths for non-ground objects
    if namespace.OBJECT_NAME != "ground":
        namespace.OBJECT_URDF_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.urdf"
        namespace.OBJECT_MESH_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.obj"
        namespace.OBJECT_URDF_TEMPLATE = f"models/templates/{namespace.OBJECT_NAME}.urdf.jinja"
        namespace.SCENE_XML_FILE = (
            f"models/{robot_config.robot_type}/"
            f"{robot_config.robot_type}_{namespace.ROBOT_DOF}dof_w_{namespace.OBJECT_NAME}.xml"
        )
    else:
        namespace.SCENE_XML_FILE = namespace.ROBOT_URDF_FILE.replace(".urdf", ".xml")

    if object_dir is not None:
        namespace.OBJECT_DIR = object_dir
        namespace.OBJECT_URDF_FILE = f"{object_dir}/{namespace.OBJECT_NAME}.urdf"
        namespace.OBJECT_MESH_FILE = f"{object_dir}/{namespace.OBJECT_NAME}.obj"

    return namespace


class RetargetingEvaluator:
    """Evaluates retargeting trajectories against quality metrics."""

    def __init__(
        self,
        robot_model_path: str,
        object_model_path: str | None,
        object_name: str,
        demo_joints: List[str],
        joints_mapping: Dict[str, str],
        visualize: bool = True,
        constants: SimpleNamespace | None = None,
        metric_mode: Literal["strict", "legacy"] = "strict",
        contact_label_root: Path | None = None,
    ):
        """Initialize evaluator with robot and object models."""
        if constants is None:
            raise ValueError("constants must be provided")

        self.object_name = object_name
        self.demo_joints = demo_joints
        self.joints_mapping = joints_mapping
        self.metric_mode = metric_mode
        self.contact_label_root = contact_label_root

        if self.object_name == "multi_boxes":
            self.collision_detection_threshold = 0.1
            self.penetration_tolerance = 0.01
            self.contact_threshold = 0.1
        else:
            self.collision_detection_threshold = 0.1
            self.penetration_tolerance = 0.01
            self.contact_threshold = 0.02

        # Foot sliding threshold (velocity in m/s)
        self.sliding_threshold = 0.01

        # Load Mujoco model
        if self.object_name == "ground":
            robot_xml_path = robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = constants.SCENE_XML_FILE  # type: ignore[attr-defined]
        else:
            robot_xml_path = robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        self._geom_names = [
            mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            for gid in range(self.robot_model.ngeom)
        ]

        if self.robot_data.qpos.shape[0] > 7 + constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        # For climbing task, we need to load the terrain
        # ===== libigl object mesh in WORLD frame (static) =====
        self._have_terrain_mesh = False
        self._ground_z = 0.0  # ground z is always 0.0

        self._obj_V_local = np.empty((0, 3), dtype=np.float64)
        self._obj_F_local = np.empty((0, 3), dtype=np.int32)
        if hasattr(constants, "OBJECT_MESH_FILE") and constants.OBJECT_MESH_FILE and Path(constants.OBJECT_MESH_FILE).exists():
            mesh = trimesh.load(constants.OBJECT_MESH_FILE, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))  # type: ignore[attr-defined]
            V = np.asarray(mesh.vertices, dtype=np.float64)  # type: ignore[attr-defined]
            F = np.asarray(mesh.faces, dtype=np.int32)  # type: ignore[attr-defined]
            if V.size == 0 or F.size == 0:
                raise ValueError("Empty object mesh")

            self._obj_V_local = V
            self._obj_F_local = F

            if not self.has_dynamic_object:
                self._obj_VW = V  # WORLD-frame vertices
                self._obj_FW = F
                self._have_terrain_mesh = True
        else:
            self._have_terrain_mesh = False

        if self._have_terrain_mesh:
            self._bake_object_mesh_from_xml()

        self.constants = constants

    def _bake_object_mesh_from_xml(self):
        """Bake world-frame triangle soup for geoms whose name contains self.object_name (mesh geoms only)."""
        m, d = self.robot_model, self.robot_data
        mujoco.mj_forward(m, d)

        obj_Vs, obj_Fs, v_acc = [], [], 0
        for gid in range(m.ngeom):
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue  # mesh-only
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if self.object_name not in name:
                continue
            Vw, F = _world_mesh_from_geom(m, d, gid, name)  # your helper
            if Vw is None or F is None or Vw.size == 0 or F.size == 0:
                continue
            obj_Vs.append(Vw.astype(np.float64))
            obj_Fs.append(F.astype(np.int32) + v_acc)
            v_acc += Vw.shape[0]

        self._obj_VW = np.vstack(obj_Vs) if obj_Vs else np.zeros((0, 3), np.float64)
        self._obj_FW = np.vstack(obj_Fs) if obj_Fs else np.zeros((0, 3), np.int32)

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco.

        Assumes q is in MuJoCo order:
        - [0:3] robot base position (xyz)
        - [3:7] robot base quaternion (wxyz)
        - [7:7+R] robot joints
        - [-7:-4] object position (xyz) if has_dynamic_object
        - [-4:] object quaternion (wxyz) if has_dynamic_object
        """
        self.robot_data.qpos[:] = q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []
        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold
        mujoco.mj_collision(m, d)

        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def evaluate_penetration(self, q_retarget: np.ndarray):
        """
        MuJoCo version of evaluate_penetration_old using your prefilter and distance calls.

        Returns:
            (fraction_with_penetration, penetration_max_depths)
            - fraction_with_penetration: float in [0,1]
            - penetration_max_depths: list[float], maximum penetration depth per penetrating frame
        """
        m, d = self.robot_model, self.robot_data

        penetration_max_depths = []
        penetration_frames = []

        # helper for name checks (populated by _prefilter_pairs_with_mj_collision)
        def _is_obj(g):
            return self.object_name in self._geom_names[g]

        def _is_ground(g):
            return "ground" in self._geom_names[g]

        def masks_ok(g1, g2):
            # skip geoms with both masks off
            if m.geom_contype[g1] == 0 and m.geom_conaffinity[g1] == 0:
                return False
            if m.geom_contype[g2] == 0 and m.geom_conaffinity[g2] == 0:
                return False
            # exclude object-ground specifically (either order)
            if (_is_obj(g1) and _is_ground(g2)) or (_is_obj(g2) and _is_ground(g1)):
                return False
            # keep only pairs that involve ground or object
            return _is_obj(g1) or _is_obj(g2) or _is_ground(g1) or _is_ground(g2)

        fromto = np.zeros(6, dtype=float)

        for i, q in enumerate(q_retarget):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)  # compute kinematics, aabbs, etc.

            # 1) collect near pairs with temporary margins (also populates _geom_names)
            candidates = self._prefilter_pairs_with_mj_collision(self.collision_detection_threshold)

            # 2) precise distance on candidates; count only strict penetrations
            depths_this_frame = []
            for g1, g2 in candidates:
                if not masks_ok(g1, g2):
                    continue
                fromto[:] = 0.0
                dist = mujoco.mj_geomDistance(m, d, g1, g2, self.collision_detection_threshold, fromto)
                # penetration = negative signed distance
                if dist < -self.penetration_tolerance:
                    depths_this_frame.append(-float(dist))

            if depths_this_frame:
                penetration_frames.append(i)
                penetration_max_depths.append(float(np.max(depths_this_frame)))

        frac = len(penetration_frames) / max(len(q_retarget), 1)
        return frac, penetration_max_depths

    def evaluate_penetration_strict(self, q_retarget: np.ndarray) -> dict[str, float]:
        """Evaluate every frame once and summarize 0/2/5/10 mm thresholds."""
        m, d = self.robot_model, self.robot_data
        frame_depths = np.zeros(len(q_retarget), dtype=float)

        def is_relevant(g1: int, g2: int) -> bool:
            n1, n2 = self._geom_names[g1], self._geom_names[g2]
            is_obj = self.object_name in n1 or self.object_name in n2
            is_ground = "ground" in n1 or "ground" in n2
            obj_ground = (self.object_name in n1 and "ground" in n2) or (
                self.object_name in n2 and "ground" in n1
            )
            return (is_obj or is_ground) and not obj_ground

        fromto = np.zeros(6, dtype=float)
        for frame, q in enumerate(q_retarget):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            candidates = self._prefilter_pairs_with_mj_collision(self.collision_detection_threshold)
            depths = []
            for g1, g2 in candidates:
                if not is_relevant(g1, g2):
                    continue
                if m.geom_contype[g1] == 0 and m.geom_conaffinity[g1] == 0:
                    continue
                if m.geom_contype[g2] == 0 and m.geom_conaffinity[g2] == 0:
                    continue
                distance = mujoco.mj_geomDistance(
                    m, d, g1, g2, self.collision_detection_threshold, fromto
                )
                if distance < 0:
                    depths.append(-float(distance))
            frame_depths[frame] = max(depths, default=0.0)
        return summarize_penetration_depths(frame_depths, PENETRATION_TOLERANCES_M)

    def detect_demo_contact(
        self,
        human_joints,
        joint_names: Sequence[str] | None = None,
    ):
        contact: dict[str, np.ndarray] = {}
        have_obj = self._obj_VW.shape[0] > 0
        if not have_obj:
            return contact  # no object mesh baked

        if joint_names is None:
            joint_names = (
                "LeftHandMiddle3",
                "RightHandMiddle3",
                "LeftFoot",
                "RightFoot",
                "LeftToeBase",
                "RightToeBase",
            )

        for jn in joint_names:
            if jn not in self.demo_joints:
                continue
            p = human_joints[self.demo_joints.index(jn)].reshape(1, 3).astype(np.float64)
            S, _, _, _ = igl.signed_distance(p, self._obj_VW, self._obj_FW)
            if S[0] <= self.contact_threshold:  # e.g., 0.02 for 2 cm
                contact[jn] = p.flatten()

        return contact

    def evaluate_contact_precision(
        self,
        human_joints_motion,
        object_poses,
        q_trajectory,
        joint_names: Sequence[str] | None = None,
    ):
        """
        Evaluate contact precision for keypoints within 2cm of surfaces.

        Args:
            q_trajectory: Robot joint configurations (N, DOF)
            object_poses: Object poses (N, 7)
            contact_sequences: Contact information per frame

        Returns:
            dict: Contact precision metrics
        """
        if joint_names is None:
            joint_names = ("L_Wrist", "R_Wrist")

        demo_local_points_list: list[np.ndarray] = []
        robot_local_points_list: list[np.ndarray] = []

        robot_joint_names = [self.joints_mapping[joint_name] for joint_name in joint_names]

        for q, human_joints, object_pose in zip(q_trajectory, human_joints_motion, object_poses):
            demo_points = np.array([human_joints[self.demo_joints.index(joint_name)] for joint_name in joint_names])
            demo_local_points_list.append(
                transform_points_world_to_local(object_pose[:4], object_pose[4:], demo_points)
            )
            robot_joint_pos = self._get_robot_link_positions(q, robot_joint_names)
            # Object pose in MuJoCo order: [-7:-4] pos, [-4:] quat
            robot_local_points_list.append(transform_points_world_to_local(q[-4:], q[-7:-4], robot_joint_pos))
        demo_local_points = np.array(demo_local_points_list)
        robot_local_points = np.array(robot_local_points_list)

        demo_contact = np.linalg.norm(demo_local_points, axis=-1) <= 0.28
        robot_contact = np.linalg.norm(robot_local_points, axis=-1) <= 0.28

        miss_contact = demo_contact & (demo_contact != robot_contact)
        worst_miss_contact = np.logical_or.reduce(miss_contact, axis=1)

        return 1 - np.sum(worst_miss_contact) / len(q_trajectory)

    def evaluate_contact_surface(
        self,
        human_joints_motion: np.ndarray,
        object_poses: np.ndarray,
        q_trajectory: np.ndarray,
        *,
        task_name: str,
    ) -> dict[str, Any]:
        """Compare robot mesh-surface contact against source mesh contact or optional GT labels."""
        joint_names = ("L_Wrist", "R_Wrist")
        frames = min(len(human_joints_motion), len(object_poses), len(q_trajectory))
        if self._obj_V_local.size == 0:
            return {"available": False, "reason": "object mesh unavailable", "method": "mesh_surface"}
        source = np.zeros((frames, 2), dtype=bool)
        predicted = np.zeros((frames, 2), dtype=bool)
        object_geom_ids = [
            gid
            for gid, name in enumerate(self._geom_names)
            if self.object_name in name
        ]
        robot_names = [self.joints_mapping[name] for name in joint_names]
        for frame in range(frames):
            points = np.array(
                [human_joints_motion[frame, self.demo_joints.index(name)] for name in joint_names]
            )
            local = transform_points_world_to_local(object_poses[frame, :4], object_poses[frame, 4:], points)
            signed, _, _, _ = igl.signed_distance(local, self._obj_V_local, self._obj_F_local)
            source[frame] = np.abs(signed) <= self.contact_threshold

            self.robot_data.qpos[:] = q_trajectory[frame]
            mujoco.mj_forward(self.robot_model, self.robot_data)
            for idx, body_name in enumerate(robot_names):
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                robot_geoms = np.flatnonzero(self.robot_model.geom_bodyid == body_id)
                distances = []
                for g1 in robot_geoms:
                    for g2 in object_geom_ids:
                        distance = mujoco.mj_geomDistance(
                            self.robot_model,
                            self.robot_data,
                            int(g1),
                            int(g2),
                            self.collision_detection_threshold,
                            np.zeros(6),
                        )
                        distances.append(float(distance))
                predicted[frame, idx] = bool(distances and min(distances) <= self.contact_threshold)

        target = source
        method = "mesh_surface"
        gt_path = None
        if self.contact_label_root is not None:
            gt, gt_path = load_omnicontact_labels(
                self.contact_label_root, task_name, frames, self.current_fps
            )
            if gt is not None:
                target = gt[:, 2:4]
                method = "omnicontact_gt"
        result = binary_contact_metrics(target, predicted)
        result.update({"method": method, "threshold_m": self.contact_threshold, "gt_path": gt_path})
        return result

    def detect_foot_sliding(self, q_trajectory, contact_sequences):
        """
        Detect foot sliding during contact phases.

        Args:
            q_trajectory: Robot joint configurations (N, DOF)
            contact_sequences: Contact information per frame

        Returns:
            dict: Foot sliding metrics
        """

        left_toe_positions = []
        right_toe_positions = []
        for q in q_trajectory:
            toe_positions = self._get_robot_link_positions(
                q, ["left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"]
            )
            left_toe_positions.append(toe_positions[0])
            right_toe_positions.append(toe_positions[1])

        left_toe_positions = np.array(left_toe_positions)
        right_toe_positions = np.array(right_toe_positions)

        left_toe_xy_velocities = np.linalg.norm(np.diff(left_toe_positions[:, :2], axis=0), axis=1)
        right_toe_xy_velocities = np.linalg.norm(np.diff(right_toe_positions[:, :2], axis=0), axis=1)
        left_toe_xy_velocities = np.concatenate([[0], left_toe_xy_velocities])
        right_toe_xy_velocities = np.concatenate([[0], right_toe_xy_velocities])

        left_foot_sticking_sequence = np.array([contact_sequence["L_Toe"] for contact_sequence in contact_sequences])
        right_foot_sticking_sequence = np.array([contact_sequence["R_Toe"] for contact_sequence in contact_sequences])

        left_foot_sliding_sequence = left_foot_sticking_sequence & (left_toe_xy_velocities > self.sliding_threshold)
        right_foot_sliding_sequence = right_foot_sticking_sequence & (right_toe_xy_velocities > self.sliding_threshold)

        num_foot_sticking_frames = np.sum(left_foot_sticking_sequence | right_foot_sticking_sequence)
        max_toe_sliding_velocities = np.max(
            np.array(
                [
                    left_toe_xy_velocities * left_foot_sliding_sequence,
                    right_toe_xy_velocities * right_foot_sliding_sequence,
                ]
            ),
            axis=0,
        )
        max_toe_sliding_velocities = max_toe_sliding_velocities[max_toe_sliding_velocities > 0]

        return (
            len(max_toe_sliding_velocities) / num_foot_sticking_frames,
            max_toe_sliding_velocities,
        )

    def detect_foot_skating_strict(self, q_trajectory, contact_sequences, fps: float) -> dict[str, float]:
        left, right = [], []
        for q in q_trajectory:
            points = self._get_robot_link_positions(
                q, ["left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"]
            )
            left.append(points[0, :2])
            right.append(points[1, :2])
        stance = np.array(
            [[row["L_Toe"], row["R_Toe"]] for row in contact_sequences[: len(q_trajectory)]], dtype=bool
        )
        return foot_skating_metrics(
            np.asarray(left), np.asarray(right), stance, fps=fps, threshold_mps=self.sliding_threshold
        )

    def evaluate_trajectory(self, task_name, data_dir, input_data_dir):
        """
        Evaluate a complete retargeting trajectory.

        Args:
            data_file: Path to pickle file containing retargeting data
            dt: Time step duration

        Returns:
            dict: Complete evaluation results
        """
        try:
            rt_res_data = np.load(f"{data_dir}", allow_pickle=True)
            q_retarget = rt_res_data["qpos"]
        except (OSError, KeyError, ValueError):
            return None
        self.current_fps = float(np.asarray(rt_res_data.get("fps", 30.0)).reshape(-1)[0])

        human_joints, object_poses = load_intermimic_data(f"{input_data_dir}/{task_name}.pt")
        contact_sequences = extract_foot_sticking_sequence_velocity(
            human_joints,
            self.demo_joints,
            ["L_Toe", "R_Toe"],
            fps=self.current_fps if self.metric_mode == "strict" else None,
        )
        if self.metric_mode == "legacy":
            penetration_duration, penetration_depths = self.evaluate_penetration(q_retarget)
            sliding_duration, sliding_velocities = self.detect_foot_sliding(q_retarget, contact_sequences)
            return {
                "penetration": {
                    "tol_10mm": penetration_duration,
                    "max": float(np.max(penetration_depths)) if penetration_depths else 0.0,
                },
                "foot_skating": {
                    "duration": sliding_duration,
                    "max_velocity": float(np.max(sliding_velocities)) if len(sliding_velocities) else 0.0,
                    "unit": "m_per_frame",
                },
                "contact": {
                    "available": True,
                    "legacy_preservation": self.evaluate_contact_precision(human_joints, object_poses, q_retarget),
                    "method": "origin_radius_0.28m",
                },
                "opt_cost": float(np.asarray(rt_res_data["cost"]).mean()),
            }
        return {
            "penetration": self.evaluate_penetration_strict(q_retarget),
            "foot_skating": self.detect_foot_skating_strict(q_retarget, contact_sequences, self.current_fps),
            "contact": self.evaluate_contact_surface(
                human_joints, object_poses, q_retarget, task_name=task_name
            ),
            "opt_cost": float(np.asarray(rt_res_data["cost"]).mean()),
        }

    def evaluate_terrain_contact_precision(
        self,
        human_joints_motion: np.ndarray,  # [T, J, 3] world
        q_trajectory: np.ndarray,  # [T, nq]
        joint_names=(
            "LeftHandMiddle3",
            "RightHandMiddle3",
            "LeftFoot",
            "RightFoot",
            "LeftToeBase",
            "RightToeBase",
        ),
    ) -> float:
        """
        For each frame:
        1) Detect demo contacts vs OBJECT.
        2) For each contacted joint, require mapped robot body to be within threshold to OBJECT.
        Returns preserved fraction over frames with any demo contact.
        """
        have_obj = self._obj_VW.shape[0] > 0
        if not have_obj:
            return 1.0  # nothing to check against

        preserved = []

        obj_gids = [
            g
            for g in range(self.robot_model.ngeom)
            if self.object_name in (mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        ]

        for _q, demo_joints in zip(q_trajectory, human_joints_motion):
            # demo contacts (object only)
            dc = self.detect_demo_contact(demo_joints, joint_names)
            if not dc:
                continue

            ok = True
            for jn in dc:
                rb = self.joints_mapping.get(jn, "")
                if not rb:
                    continue
                bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, rb)
                if bid == -1:
                    continue
                dist_min = np.inf
                fromto = np.zeros(6)
                for g1 in range(self.robot_model.ngeom):
                    if self.robot_model.geom_bodyid[g1] != bid:
                        continue
                    for g2 in obj_gids:
                        dist = mujoco.mj_geomDistance(
                            self.robot_model, self.robot_data, g1, g2, self.collision_detection_threshold, fromto
                        )
                        dist_min = min(dist_min, dist)

                if dist_min > self.contact_threshold:
                    ok = False
                    break
            preserved.append(ok)
        return 1.0 if not preserved else float(np.mean(preserved))

    def evaluate_robot_terrain_trajectory(self, task_name, data_dir, input_data_dir):
        """
        Evaluate a complete retargeting trajectory.

        Args:
            data_file: Path to pickle file containing retargeting data
            dt: Time step duration

        Returns:
            dict: Complete evaluation results
        """
        try:
            rt_res_data = np.load(f"{data_dir}", allow_pickle=True)
            q_retarget = rt_res_data["qpos"]
        except (OSError, KeyError, ValueError):
            return None
        self.current_fps = float(np.asarray(rt_res_data.get("fps", 30.0)).reshape(-1)[0])

        input_data_path = f"{input_data_dir}/{task_name}"
        npy_file = next(iter(Path(input_data_path).glob("*.npy")))
        smpl_scale = self.constants.ROBOT_HEIGHT / 1.78
        human_joints = np.load(npy_file)[::4] * smpl_scale

        contact_sequences = extract_foot_sticking_sequence_velocity(
            human_joints,
            self.demo_joints,
            ["LeftToeBase", "RightToeBase"],
            fps=self.current_fps if self.metric_mode == "strict" else None,
        )
        if self.metric_mode == "legacy":
            sliding_duration, sliding_velocities = self.detect_foot_sliding(q_retarget, contact_sequences)
            penetration_duration, penetration_depths = self.evaluate_penetration(q_retarget)
            skating = {
                "duration": sliding_duration,
                "max_velocity": float(np.max(sliding_velocities)) if len(sliding_velocities) else 0.0,
                "unit": "m_per_frame",
            }
            penetration = {
                "tol_10mm": penetration_duration,
                "max": float(np.max(penetration_depths)) if penetration_depths else 0.0,
            }
        else:
            skating = self.detect_foot_skating_strict(q_retarget, contact_sequences, self.current_fps)
            penetration = self.evaluate_penetration_strict(q_retarget)

        contact_results = self.evaluate_terrain_contact_precision(human_joints, q_retarget)

        opt_cost = rt_res_data["cost"]

        return {
            "penetration": penetration,
            "foot_skating": skating,
            "contact": {"available": True, "terrain_preservation": contact_results, "method": "terrain_mesh"},
            "opt_cost": float(np.asarray(opt_cost).mean()),
        }

    def evaluate_robot_only_trajectory(self, task_name, data_dir, input_data_dir):
        """
        Evaluate a complete retargeting trajectory.

        Args:
            task_name: Name of the task/sequence
            data_dir: Path to retargeting result file (.npz)
            input_data_dir: Path to input data directory

        Returns:
            dict: Complete evaluation results
        """
        try:
            rt_res_data = np.load(f"{data_dir}", allow_pickle=True)
            q_retarget = rt_res_data["qpos"]
        except (OSError, KeyError, ValueError):
            return None
        self.current_fps = float(np.asarray(rt_res_data.get("fps", 30.0)).reshape(-1)[0])

        # Determine data format by checking file existence
        data_name = task_name.split("_original")[0]
        npy_path = Path(input_data_dir) / f"{data_name}.npy"
        pt_path = Path(input_data_dir) / f"{data_name}.pt"

        # Determine data format and toe names based on file extension
        if pt_path.exists():
            # OMOMO (smplh) data format
            toe_names = ["L_Toe", "R_Toe"]
            human_joints, _ = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(data_name, self.constants.ROBOT_HEIGHT)

            # For smplh data, we need to use smplh demo_joints for contact extraction
            # Check if toe names are in current demo_joints
            if all(toe in self.demo_joints for toe in toe_names):
                # Use current demo_joints
                demo_joints_for_contact = self.demo_joints
                human_joints = preprocess_motion_data(human_joints, self, toe_names, smpl_scale)
            else:
                # Use smplh demo_joints for contact extraction
                demo_joints_for_contact = SMPLH_DEMO_JOINTS
                # Just scale without normalization (smplh data doesn't need height normalization)
                human_joints = human_joints * smpl_scale
        elif npy_path.exists():
            # LAFAN data format
            toe_names = ["LeftToeBase", "RightToeBase"]
            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = self.demo_joints.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = getattr(self.constants, "DEFAULT_SCALE_FACTOR", None) or 1.0

            human_joints = preprocess_motion_data(human_joints, self, toe_names, smpl_scale)
            demo_joints_for_contact = self.demo_joints
        else:
            raise FileNotFoundError(f"Neither {npy_path} nor {pt_path} found for task {data_name}")

        contact_sequences = extract_foot_sticking_sequence_velocity(
            human_joints,
            demo_joints_for_contact,
            toe_names,
            fps=self.current_fps if self.metric_mode == "strict" else None,
        )
        if self.metric_mode == "legacy":
            penetration_duration, penetration_depths = self.evaluate_penetration(q_retarget)
            sliding_duration, sliding_velocities = self.detect_foot_sliding(q_retarget, contact_sequences)
            penetration = {
                "tol_10mm": penetration_duration,
                "max": float(np.max(penetration_depths)) if penetration_depths else 0.0,
            }
            skating = {
                "duration": sliding_duration,
                "max_velocity": float(np.max(sliding_velocities)) if len(sliding_velocities) else 0.0,
                "unit": "m_per_frame",
            }
        else:
            penetration = self.evaluate_penetration_strict(q_retarget)
            skating = self.detect_foot_skating_strict(q_retarget, contact_sequences, self.current_fps)

        opt_cost = rt_res_data["cost"]

        return {
            "penetration": penetration,
            "foot_skating": skating,
            "contact": {"available": False, "reason": "robot-only sequence"},
            "opt_cost": float(np.asarray(opt_cost).mean()),
        }


def _evaluate_single_task(
    task_name: str,
    data_path: str,
    input_data_dir: str,
    robot_config_kwargs: Dict[str, Any],
    motion_data_config_kwargs: Dict[str, Any],
    object_name: str | None,
    data_type: str,
    metric_mode: Literal["strict", "legacy"],
    contact_label_root: str | None,
):
    robot_config = RobotConfig(**robot_config_kwargs)
    motion_data_config = MotionDataConfig(**motion_data_config_kwargs)

    constants = create_task_constants(
        robot_config,
        motion_data_config,
        object_name=object_name,
    )

    if data_type == "robot_terrain":
        # For robot_terrain task
        constants.OBJECT_DIR = f"{input_data_dir}/{task_name}"
        constants.OBJECT_URDF_FILE = f"{constants.OBJECT_DIR}/{constants.OBJECT_NAME}.urdf"
        constants.OBJECT_MESH_FILE = f"{constants.OBJECT_DIR}/{constants.OBJECT_NAME}.obj"

        box_asset_xml = f"{constants.OBJECT_DIR}/box_assets.xml"
        scene_xml_name = constants.ROBOT_URDF_FILE.split("/")[-1].replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_path = f"{constants.OBJECT_DIR}/{scene_xml_name}"

        object_scale = np.array([1, 1, 1])
        smpl_scale = constants.ROBOT_HEIGHT / 1.78

        # Update object scale in .xml file
        object_asset_xml_path = create_scaled_multi_boxes_xml(
            box_asset_xml,
            object_scale * smpl_scale,
        )
        new_scene_xml_path = create_new_scene_xml_file(
            scene_xml_path,
            object_scale * smpl_scale,
            object_asset_xml_path,
        )
        constants.SCENE_XML_FILE = new_scene_xml_path

    object_model_path: str | None = getattr(constants, "OBJECT_URDF_FILE", None)

    evaluator = RetargetingEvaluator(
        robot_model_path=constants.ROBOT_URDF_FILE,
        object_model_path=object_model_path,
        object_name=constants.OBJECT_NAME,
        demo_joints=constants.DEMO_JOINTS,
        joints_mapping=constants.JOINTS_MAPPING,
        visualize=False,
        constants=constants,
        metric_mode=metric_mode,
        contact_label_root=Path(contact_label_root) if contact_label_root else None,
    )
    if data_type == "robot_object":
        return task_name, evaluator.evaluate_trajectory(task_name, data_path, input_data_dir)
    if data_type == "robot_only":
        return task_name, evaluator.evaluate_robot_only_trajectory(task_name, data_path, input_data_dir)
    if data_type == "robot_terrain":
        return task_name, evaluator.evaluate_robot_terrain_trajectory(task_name, data_path, input_data_dir)
    raise ValueError(f"Invalid data type: {data_type}")


def get_task_names(data_dir, data_type, input_data_dir: str | None = None):
    data_path = Path(data_dir)
    input_path = Path(input_data_dir) if input_data_dir else None
    if data_type == "robot_object":
        outputs = {p.name.removesuffix("_original.npz"): p for p in data_path.glob("*_original.npz")}
        expected = {p.stem for p in input_path.glob("*.pt")} if input_path and input_path.exists() else set()
        task_names = sorted(set(outputs) | expected)
        files = [outputs.get(name, data_path / f"{name}_original.npz") for name in task_names]
    elif data_type == "robot_only":
        outputs = {
            p.name.removesuffix("_original.npz").removesuffix(".npz"): p for p in data_path.glob("*.npz")
        }
        expected = set()
        if input_path and input_path.exists():
            expected = {p.stem for pattern in ("*.npy", "*.pt") for p in input_path.glob(pattern)}
        task_names = sorted(set(outputs) | expected)
        files = [outputs.get(name, data_path / f"{name}_original.npz") for name in task_names]
    elif data_type == "robot_terrain":
        files = sorted(data_path.glob("*_original.npz"))
        task_names = [p.name.replace("_original.npz", "").split("_joint_positions")[0] for p in files]
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    return task_names, [str(p) for p in files]


@dataclass
class Args:
    """Evaluation configuration."""

    res_dir: Path
    data_dir: Path
    data_type: Literal["robot_object", "robot_only", "robot_terrain"] = "robot_object"
    robot: str = "g1"  # Use str to allow dynamic robot types
    data_format: str | None = None  # Use str to allow dynamic data formats
    object_name: str | None = None
    max_workers: int = 1
    metric_mode: Literal["strict", "legacy"] = "strict"
    contact_label_root: Path | None = None
    output: Path | None = None
    dataset: str | None = None

    # Nested configs for overrides
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )


def main(cfg: Args) -> None:
    started_at = datetime.now(timezone.utc)
    default_data_formats = {
        "robot_object": "smplh",
        "robot_only": "smplh",
        "robot_terrain": "mocap",
    }

    data_format = cfg.data_format or default_data_formats[cfg.data_type]

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)

    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(
            data_format=data_format,  # data_format is now str, no cast needed
            robot_type=cfg.robot,
        )

    # Determine default object name when none provided
    if cfg.object_name is not None:
        object_name = cfg.object_name
    elif cfg.data_type == "robot_object":
        object_name = "largebox"
    elif cfg.data_type == "robot_terrain":
        object_name = "multi_boxes"
    else:
        # Default to "ground" for robot-only scenarios (matches robot defaults)
        object_name = "ground"

    task_names, files = get_task_names(str(cfg.res_dir), cfg.data_type, str(cfg.data_dir))
    print(f"Found {len(task_names)} tasks")

    robot_config_kwargs = asdict(cfg.robot_config)
    motion_data_config_kwargs = asdict(cfg.motion_data_config)

    results: Dict[str, Dict[str, Any]] = {}
    max_workers = max(1, cfg.max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_task,
                task_name,
                file_path,
                str(cfg.data_dir),
                robot_config_kwargs,
                motion_data_config_kwargs,
                object_name,
                cfg.data_type,
                cfg.metric_mode,
                str(cfg.contact_label_root) if cfg.contact_label_root else None,
            ): task_name
            for task_name, file_path in zip(task_names, files)
        }
        for fut in as_completed(futures):
            try:
                task_name, res = fut.result()
                if res is None:
                    results[task_name] = {"retarget_success": False, "error": "trajectory could not be loaded"}
                else:
                    results[task_name] = {"retarget_success": True, **res}
            except Exception as exc:
                results[task_name] = {"retarget_success": False, "error": f"{type(exc).__name__}: {exc}"}

    file_by_task = dict(zip(task_names, files))
    records = []
    for task_name in task_names:
        raw = results.get(task_name, {"retarget_success": False, "error": "worker produced no result"})
        output_path = file_by_task[task_name]
        frames = 0
        fps = None
        try:
            with np.load(output_path, allow_pickle=True) as archive:
                frames = int(len(archive["qpos"]))
                fps = float(np.asarray(archive.get("fps", 30.0)).reshape(-1)[0])
        except (OSError, KeyError, ValueError):
            pass
        source_candidates = [cfg.data_dir / f"{task_name}.pt", cfg.data_dir / f"{task_name}.npy"]
        source_path = next((path for path in source_candidates if path.exists()), cfg.data_dir / task_name)
        expected_frames = None
        try:
            if source_path.suffix == ".npy":
                expected_frames = int(len(np.load(source_path, mmap_mode="r")))
            elif source_path.suffix == ".pt":
                expected_frames = int(len(load_intermimic_data(str(source_path))[0]))
        except (OSError, ValueError, RuntimeError):
            pass
        completed_ratio = (
            min(frames / expected_frames, 1.0) if expected_frames and raw.get("retarget_success") else 0.0
        )
        record = {
            "sequence": task_name,
            "source_path": str(source_path),
            "output_path": output_path,
            "frames": frames,
            "expected_frames": expected_frames,
            "fps": fps,
            "completed_frame_ratio": completed_ratio,
            **raw,
        }
        records.append(record)

    successful = [record for record in records if record["retarget_success"]]
    dataset = cfg.dataset or cfg.res_dir.parent.name
    run_config = {
        "dataset": dataset,
        "data_type": cfg.data_type,
        "metric_mode": cfg.metric_mode,
        "penetration_tolerances_m": list(PENETRATION_TOLERANCES_M),
        "sliding_threshold_mps": 0.01,
        "res_dir": str(cfg.res_dir),
        "data_dir": str(cfg.data_dir),
        "contact_label_root": str(cfg.contact_label_root) if cfg.contact_label_root else None,
        "robot": cfg.robot,
        "object_name": object_name,
    }
    config_hash = canonical_config_hash(run_config)
    root_commit = git_commit(src_root.parents[2])
    holosoma_commit = git_commit(src_root.parents[1])
    for record in records:
        record["provenance"] = {
            "config_hash": config_hash,
            "root_git_commit": root_commit,
            "holosoma_git_commit": holosoma_commit,
            "metric_mode": cfg.metric_mode,
        }
    payload = {
        "schema_version": "1.0.0",
        "run": {
            **run_config,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "config_hash": config_hash,
            "root_git_commit": root_commit,
            "holosoma_git_commit": holosoma_commit,
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "results": records,
        "summary": {
            "total": len(records),
            "successful": len(successful),
            "failed": len(records) - len(successful),
            "retarget_success_rate": len(successful) / len(records) if records else 0.0,
        },
    }
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output = cfg.output or cfg.res_dir.parent / "evaluation" / f"eval_{dataset}_{cfg.data_type}_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved structured evaluation to {output}")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
