from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_rs_follower")
@dataclass
class SeeedB601RSFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    motor_models: dict[str, str] = field(
        default_factory=lambda: {
            "joint_1": "04",
            "joint_2": "04",
            "joint_3": "04",
            "joint_4": "04",
            "joint_5": "04",
            "joint_6": "04",
            "gripper": "04",
        }
    )
