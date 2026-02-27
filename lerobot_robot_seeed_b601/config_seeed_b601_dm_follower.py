from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.robot import RobotConfig


@dataclass
class SeeedB601DMFollowerConfigBase:
    # CAN interfaces - one per arm (though B601 is single arm usually)
    # Linux: "can0", "can1", etc.
    # Mac: "/dev/tty.usbmodem..." (slcan)
    port: str

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "auto"

    # CAN FD settings (Use CAN FD by default for Damiao motors)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # Safety limit for relative target positions
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Motor configuration for B601-DM (6 DOF + Gripper)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on TRLC-DK1 hardware context and user confirmation:
    # Joint 1, 4-6, Gripper: DM4310
    # Joint 2-3: DM4340
    # IDs: 0x01 - 0x07
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm4310"),  # Base (DM4310)
            "joint_2": (0x02, 0x12, "dm4340"),  # Shoulder (DM4340)
            "joint_3": (0x03, 0x13, "dm4340"),  # Elbow (DM4340)
            "joint_4": (0x04, 0x14, "dm4310"),  # Wrist 1 (DM4310)
            "joint_5": (0x05, 0x15, "dm4310"),  # Wrist 2 (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # Wrist 3 (DM4310)
            "gripper": (0x07, 0x17, "dm4310"),  # Gripper (DM4310)
        }
    )

    # MIT control parameters for position control (used in send_action)
    # List of 7 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
    # Initial guessing based on DM motor defaults and OpenArm. Needs tuning.
    position_kp: list[float] = field(
        default_factory=lambda: [100.0, 150.0, 150.0, 80.0, 80.0, 80.0, 100.0]
    )
    position_kd: list[float] = field(
        default_factory=lambda: [2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    )

    # Values for joint limits (Degrees)
    # Note: These are soft limits. Physical verification is recommended.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-170.0, 170.0),
            "joint_2": (-90.0, 90.0),
            "joint_3": (-150.0, 150.0),
            "joint_4": (-100.0, 100.0),
            "joint_5": (-90.0, 90.0),
            "joint_6": (-170.0, 170.0),
            "gripper": (-270.0, 0.0),
        }
    )


@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig, SeeedB601DMFollowerConfigBase):
    pass
