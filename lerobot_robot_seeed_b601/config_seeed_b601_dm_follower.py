from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):

    # The v_des parameter for the position-velocity control mode of the joints.
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150, 150, 150, 150, 150, 150, 150]
    )

    # Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1

