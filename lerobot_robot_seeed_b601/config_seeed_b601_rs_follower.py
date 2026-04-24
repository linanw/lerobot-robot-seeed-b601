from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_rs_follower")
@dataclass
class SeeedB601RSFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    # The Kp parameter for the MIT control mode of the gripper.
    gripper_mit_kp: float = 3.5

    # The Kd parameter for the MIT control mode of the gripper.
    gripper_mit_kd: float = 0.3

    # Max absolute output torque allowed in MIT mode for the gripper
    gripper_mit_torque_limit: float = 2.0

    # The v_des parameter for the position-velocity control mode of the joints.
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [300, 300, 300, 300, 300, 300, 0]
    )
