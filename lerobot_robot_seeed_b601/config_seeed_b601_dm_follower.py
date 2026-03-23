from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    motor_models: dict[str, str] = field(
        default_factory=lambda: {
            "joint_1": "dm4310",
            "joint_2": "dm4340",
            "joint_3": "dm4340",
            "joint_4": "dm4310",
            "joint_5": "dm4310",
            "joint_6": "dm4310",
            "gripper": "dm4310",
        }
    )
