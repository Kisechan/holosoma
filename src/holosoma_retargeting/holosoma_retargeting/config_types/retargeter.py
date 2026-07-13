"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class PhaseWindowConfig:
    """Configuration for receding-horizon temporal retargeting refinement."""

    window_frames: int = 30
    stride_frames: int = 15
    max_iterations: int = 8
    trust_radius: float = 0.1
    min_trust_radius: float = 0.0125
    reference_weight: float = 10.0
    velocity_weight: float = 1.0
    acceleration_weight: float = 0.2
    contact_position_weight: float = 1000.0
    contact_velocity_weight: float = 100.0
    foot_anchor_weight: float = 1000.0
    collision_active_top_k: int = 32
    collision_recovery_max_slack: float = 0.005
    collision_recovery_weight: float = 1e6
    foot_xy_tolerance: float = 0.003
    foot_slack_limit: float = 0.005
    foot_z_lower: float = -0.001
    foot_z_upper: float = 0.003

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.stride_frames < 1 or self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must be in [1, window_frames]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.min_trust_radius <= 0 or self.trust_radius < self.min_trust_radius:
            raise ValueError("trust radii must be positive and ordered")
        if self.collision_active_top_k < 1:
            raise ValueError("collision_active_top_k must be positive")
        if self.collision_recovery_max_slack < 0:
            raise ValueError("collision_recovery_max_slack must be nonnegative")


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    collision_constraint_mode: str = "hard"
    """Collision constraint mode: hard or soft."""

    collision_slack_weight: float = 1e6
    """Penalty weight for elastic collision slack when collision_constraint_mode is soft."""

    collision_max_slack: float | None = None
    """Optional upper bound for elastic collision slack."""

    feasibility_recovery_mode: str = "off"
    """Recovery ablation: off (baseline), fixed, or adaptive trust region."""

    restoration_collision_slack_weight: float = 1e6
    """Collision slack weight used only by the restoration QP."""

    restoration_foot_slack_weight: float = 1e5
    """Foot-sticking slack weight used only by the restoration QP."""

    restoration_collision_max_slack: float | None = None
    """Optional collision slack cap used only during restoration."""

    restoration_foot_max_slack: float | None = None
    """Optional foot-sticking slack cap used only during restoration."""

    restoration_max_steps: int = 1
    """Maximum elastic recovery steps before declaring the hard SQP retry infeasible."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    adaptive_min_step_size: float = 0.025
    adaptive_max_step_size: float = 0.4
    adaptive_shrink_factor: float = 0.5
    adaptive_grow_factor: float = 1.25

    initial_unbounded_retry: bool = True
    """Legacy first-frame fallback that removes the trust-region SOC after failure."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
