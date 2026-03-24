import logging
import math
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position


@dataclass
class SeeedB601FollowerConfigBase:
    """Base configuration for the Seeed B601 Follower arm."""
    port: str
    # CAN adapter type:
    #   "socketcan"  - SocketCAN based adapters (PCAN, slcan, embedded can controller, etc.)
    #   "damiao"     - Damiao dedicated serial bridge
    #   "robstride"  - RobStride dedicated adapter (placeholder, not yet supported)
    can_adapter: str = "socketcan"
    # Baud rate for Damiao serial bridge (only used when can_adapter="damiao")
    dm_serial_baud: int = 921600
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Motor configuration for B601 (6 DOF + Gripper)
    # Maps motor names to (send_can_id, recv_can_id)
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11),
            "joint_2": (0x02, 0x12),
            "joint_3": (0x03, 0x13),
            "joint_4": (0x04, 0x14),
            "joint_5": (0x05, 0x15),
            "joint_6": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )
    
    # Motor hardware model specifications (to be defined by subclass)
    motor_models: dict[str, str] = field(default_factory=dict)

    # Control parameters are defined by concrete subclasses so different motor families
    # can keep their own defaults.
    # MIT gains used only by the gripper motor.
    gripper_mit_kp: float = 0.0
    gripper_mit_kd: float = 0.0
    # Default target velocity for joints running in POS_VEL mode, in degrees/s.
    pos_vel_velocity: float | list[float] = field(default_factory=list)

    # Values for joint limits (Degrees)
    # Note: These are soft limits. Physical verification is recommended.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-145.0, 145.0),
            "joint_2": (-170.0, 1.0),
            "joint_3": (-200.0, 1.0),
            "joint_4": (-80.0, 90.0),
            "joint_5": (-90.0, 90.0),
            "joint_6": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


logger = logging.getLogger(__name__)


FOLLOWER_GRIPPER_MOTOR = "gripper"


class SeeedB601FollowerBase(Robot):
    """
    Base class for Seeed B601 Follower Arms (DM and RS variants).
    Uses CAN bus communication via motorbridge.
    """

    def __init__(self, config: SeeedB601FollowerConfigBase):
        super().__init__(config)
        self.config = config
        self.bus = None
        self.motors = {}
        self.motor_names = list(config.motor_can_ids.keys())

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        logger.setLevel(logging.DEBUG)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.motor_names:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus is not None and all(cam.is_connected for cam in self.cameras.values())

    def _add_motors_to_bus(self):
        """Must be implemented by subclasses to add specific motor types to self.bus."""
        raise NotImplementedError

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the follower arm and optionally calibrate."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting arm on {self.config.port} (adapter={self.config.can_adapter})...")
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "robstride":
            raise NotImplementedError(
                "RobStride dedicated USB-to-CAN adapter is not yet supported in motorbridge Python SDK."
            )
        else:
            # Default: socketcan (PCAN, slcan, etc.)
            self.bus = MotorBridgeController(channel=self.config.port)
        
        self._add_motors_to_bus()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        if self.is_calibrated:
            for motor in self.motors.values():
                motor.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return bool(self.calibration)

    def calibrate(self) -> None:
        """Calibration procedure for B601."""
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration for {self}")
        
        for motor in self.motors.values():
            motor.disable()

        print(
            "\nCalibration: Set Zero Position\n"
            "Please MANUALLY move the robot to its ZERO POSITION.\n"
            "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n"
        )
        input("Press ENTER when the robot is in ZERO POSITION...")

        for motor in self.motors.values():
            motor.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        self.bus.enable_all()

        # Damiao motors need a short delay after enable before register operations
        time.sleep(0.3)

        for motor_name, motor in self.motors.items():
            target_mode = (
                MotorBridgeMode.MIT
                if motor_name == FOLLOWER_GRIPPER_MOTOR
                else MotorBridgeMode.POS_VEL
            )
            try:
                motor.ensure_mode(target_mode)
            except Exception:
                logger.warning(
                    f"ensure_mode({target_mode.name}) failed for {motor_name}, continuing anyway"
                )

    def get_observation(self) -> RobotObservation:
        """Get current observation from robot."""
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Request and poll feedback from motorbridge
        for motor in self.motors.values():
            motor.request_feedback()

        self.bus.poll_feedback_once()

        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                # motorbridge works natively in radians. Convert to degrees.
                obs_dict[f"{motor_name}.pos"] = math.degrees(state.pos)
                obs_dict[f"{motor_name}.vel"] = math.degrees(state.vel)
                obs_dict[f"{motor_name}.torque"] = state.torq
            else:
                obs_dict[f"{motor_name}.pos"] = 0.0
                obs_dict[f"{motor_name}.vel"] = 0.0
                obs_dict[f"{motor_name}.torque"] = 0.0

        # Capture images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")
        print(f"{self} get_observation took: {dt_ms:.1f}ms")
        print(f"Observation: {obs_dict}")

        return obs_dict

    def send_action(
        self,
        action: RobotAction,
        custom_gripper_mit_kp: float | None = None,
        custom_gripper_mit_kd: float | None = None,
    ) -> RobotAction:
        """Send action command to robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_vel = {key.removesuffix(".vel"): val for key, val in action.items() if key.endswith(".vel")}

        # Apply joint limit clipping
        for motor_name, position in goal_pos.items():
            # print(f"motor_name: {motor_name}, position: {position}")
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}")
                goal_pos[motor_name] = clipped_position

        # Safety: Cap relative target
        if self.config.max_relative_target is not None:
            # We need current position in degrees to compare against relative limit safely
            present_pos = {}
            for motor_name, motor in self.motors.items():
                state = motor.get_state()
                if state is not None:
                    present_pos[motor_name] = math.degrees(state.pos)
                else:
                    present_pos[motor_name] = 0.0
            
            goal_present_pos = {key: (g_pos, present_pos.get(key, g_pos)) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Prepare and send commands
        for motor_name, position_degrees in goal_pos.items():
            try:
                idx = self.motor_names.index(motor_name)
            except ValueError:
                idx = 0 # Fallback

            # Convert target position from degrees to radians for motorbridge
            pos_rad = math.radians(position_degrees)
            # if motor_name in goal_vel:
            #     vel_deg_s = goal_vel[motor_name]
            # else:
            #     vel_deg_s = (
            #         self.config.pos_vel_velocity[idx]
            #         if isinstance(self.config.pos_vel_velocity, list)
            #         else self.config.pos_vel_velocity
            #     )
            

            motor = self.motors.get(motor_name)
            if motor is not None:
                if motor_name == FOLLOWER_GRIPPER_MOTOR:
                    # Keep the gripper in MIT mode for finer compliance control.
                    kp = (
                        custom_gripper_mit_kp
                        if custom_gripper_mit_kp is not None
                        else self.config.gripper_mit_kp
                    )
                    kd = (
                        custom_gripper_mit_kd
                        if custom_gripper_mit_kd is not None
                        else self.config.gripper_mit_kd
                    )
                    motor.send_mit(pos_rad, 0.0, kp, kd, 0.0)
                    print(f"Sent MIT command to {motor_name}: pos={position_degrees:.2f}°, kp={kp}, kd={kd}")
                else:
                    vel_deg_s = (
                        self.config.pos_vel_velocity[idx]
                        if isinstance(self.config.pos_vel_velocity, list)
                        else self.config.pos_vel_velocity
                    )
                    vel_rad = math.radians(vel_deg_s)
                    motor.send_pos_vel(pos_rad, vel_rad)
                    print(f"Sent POS_VEL command to {motor_name}: pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s")

        # motorbridge sends packets mostly synchronously here over loop, 
        # so we don't need a bulk send command through ctypes.

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.disable_torque_on_disconnect:
            for motor in self.motors.values():
                motor.disable()

        self.bus.close()
        self.bus = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
