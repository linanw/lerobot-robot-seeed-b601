from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.robot import RobotConfig


@dataclass
class SeeedB601RSFollowerConfigBase:
    """
    Configuration for Seeed B601-RS (RobStride Motors).
    Currently a placeholder.
    """
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Add RobStride specific config here later


@RobotConfig.register_subclass("seeed_b601_rs_follower")
@dataclass
class SeeedB601RSFollowerConfig(RobotConfig, SeeedB601RSFollowerConfigBase):
    pass
