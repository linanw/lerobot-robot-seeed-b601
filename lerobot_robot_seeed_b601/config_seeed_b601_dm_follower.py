from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-145.0, 145.0),
            "shoulder_lift": (-170.0, 0.0),
            "elbow_flex": (-200.0, 0.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )

    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_yaw": 1.0,
            "wrist_roll": -1.0,
            "gripper": -6.0,
        }
    )

    # The v_des parameter for the position-velocity control mode of the joints.
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150, 150, 150, 150, 150, 150, 150]
    )

    # Track small LeRobot setpoint increments continuously instead of moving
    # every frame at the full 150 deg/s limit and stopping between frames.
    adaptive_pos_vel_velocity: bool = True
    # Raise only a lagging joint's vlim from its latest physical position error,
    # which keeps loaded shoulder/elbow joints synchronized with the wrist.
    adaptive_pos_vel_feedback: bool = True
    pos_vel_tracking_ratio: float = 1.05
    # Cartesian keyboard jogging has its own acceleration ramp. It does not
    # affect Ctrl/Shift arm following or Tab gripper following.
    limit_cartesian_jog_acceleration: bool = True
    cartesian_jog_accel_m_s2: float = 0.12
    # Keep this below the slowest expected joint rate. A larger floor makes
    # low-scale Cartesian motion asynchronous because small-delta joints arrive
    # much earlier than the rest of the arm.
    pos_vel_min_velocity: float = 0.02

    # Seeed's B601-DM POS_VEL tuning. Applied at connect time without saving to
    # motor flash. Set configure_pos_vel_gains=false to retain persisted values.
    configure_pos_vel_gains: bool = True
    pos_vel_kp_asr: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0125,
            "shoulder_lift": 0.0125,
            "elbow_flex": 0.0125,
            "wrist_flex": 0.0008,
            "wrist_yaw": 0.0008,
            "wrist_roll": 0.0008,
        }
    )
    pos_vel_ki_asr: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.004,
            "shoulder_lift": 0.004,
            "elbow_flex": 0.004,
            "wrist_flex": 0.002,
            "wrist_yaw": 0.002,
            "wrist_roll": 0.002,
        }
    )
    pos_vel_kp_apr: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 150.0,
            "shoulder_lift": 150.0,
            "elbow_flex": 150.0,
            "wrist_flex": 50.0,
            "wrist_yaw": 50.0,
            "wrist_roll": 50.0,
        }
    )
    pos_vel_ki_apr: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.5,
            "shoulder_lift": 0.5,
            "elbow_flex": 0.5,
            "wrist_flex": 1.0,
            "wrist_yaw": 1.0,
            "wrist_roll": 1.0,
        }
    )
    # Runtime arm hold vlim used by keyboard-gated Home/End. A stationary
    # target otherwise uses pos_vel_min_velocity (0.02 deg/s), which masks
    # large KP_APR changes by limiting the motor's corrective response.
    arm_hold_velocity_step: float = 0.5
    arm_hold_velocity_min: float = 0.02
    arm_hold_velocity_max: float = 5.0
    # Home/End adjusts joints 2, 3, and 4 upward from these startup values.
    # The startup values are also the runtime lower limits.
    arm_hold_kp_asr_initial: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0125,
            "shoulder_lift": 0.0125,
            "elbow_flex": 0.0125,
            "wrist_flex": 0.0008,
            "wrist_yaw": 0.0008,
            "wrist_roll": 0.0008,
        }
    )
    # Lower speed-loop gains used while Ctrl/Shift/XYZ is actively moving the
    # arm. Tab controls only the gripper and does not select these gains.
    arm_motion_kp_asr: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0125,
            "shoulder_lift": 0.0125,
            "elbow_flex": 0.0125,
            "wrist_flex": 0.0008,
            "wrist_yaw": 0.0008,
            "wrist_roll": 0.0008,
        }
    )
    arm_hold_kp_asr_step: float = 0.0004
    arm_hold_kp_asr_extra_max: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_lift": 0.004,
            "elbow_flex": 0.004,
            "wrist_flex": 0.004,
        }
    )

    # Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1
