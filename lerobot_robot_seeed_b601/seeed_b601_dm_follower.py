import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from .config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig

logger = logging.getLogger(__name__)


class SeeedB601DMFollower(Robot):
    """
    Seeed B601-DM Robot Arm (6-DOF + Gripper) using Damiao motors.
    Uses CAN bus communication via DamiaoMotorsBus (MIT Mode).
    """

    config_class = SeeedB601DMFollowerConfig
    name = "seeed_b601_dm_follower"

    def __init__(self, config: SeeedB601DMFollowerConfig):
        super().__init__(config)
        self.config = config

        # Initialize motors based on config
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
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
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        # For B601, we might need a specific calibration routine.
        # For now, we follow standard logic: if no calibration file, run calibrate().
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        if self.is_calibrated:
            self.bus.set_zero_position()

        self.bus.enable_torque()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Calibration procedure for B601-DM.
        Since B601 structure is different from OpenArm (cannot just hang down),
        we assume the user Manually Positions the robot to the Zero Configuration before running this.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # Step 1: Set zero position
        print(
            "\nCalibration: Set Zero Position\n"
            "Please MANUALLY move the robot to its ZERO POSITION.\n"
            "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n"
        )
        input("Press ENTER when the robot is in ZERO POSITION...")

        # Set current position as zero
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting default range (-180 to 180) for safety")
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )  # range is not used for Damiao motors

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def get_observation(self) -> RobotObservation:
        """Get current observation from robot."""
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Sync read all states
        states = self.bus.sync_read_all_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
            obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Capture images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """Send action command to robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply joint limit clipping
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}")
                goal_pos[motor_name] = clipped_position

        # Safety: Cap relative target
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Prepare batch commands
        # Map motor name to index in config.position_kp list
        motor_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"
        ]
        
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            try:
                idx = motor_names.index(motor_name)
            except ValueError:
                idx = 0 # Fallback

            # Determine Kp/Kd
            if custom_kp and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = self.config.position_kp[idx] if isinstance(self.config.position_kp, list) else self.config.position_kp
            
            if custom_kd and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = self.config.position_kd[idx] if isinstance(self.config.position_kd, list) else self.config.position_kd

            # For Gripper, we might want higher torque limit? 
            # DamiaoMotorsBus handles basic encoding. 
            # If explicit torque control needed, implies modifying wrapper, but MIT mode allows torque feedforward.
            # Here we just use Position Control (MIT Mode with q, qdot=0, kp, kd, tau=0)
            
            commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)

        self.bus._mit_control_batch(commands)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
