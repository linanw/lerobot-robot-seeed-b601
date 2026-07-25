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

    # Communication port for CAN adapter (e.g., "can0" for SocketCAN, or "/dev/ttyACM0" for Damiao serial bridge)
    port: str

    # CAN adapter type:
    #   "socketcan"  - SocketCAN based adapters (PCAN, slcan, embedded can controller, etc.)
    #   "damiao"     - Damiao dedicated serial bridge
    #   "robstride"  - RobStride dedicated adapter (placeholder, not yet supported)
    can_adapter: str = "socketcan"

    # Baud rate for Damiao serial bridge (only used when can_adapter="damiao")
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # Max relative target for joint movements, in degrees
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Motor configuration must be provided by concrete subclasses.
    # Maps motor names to (send_can_id, recv_can_id)
    motor_can_ids: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Control parameters are defined by concrete subclasses so different motor families
    # can keep their own defaults.
    ## Default target velocity for joints running in POS_VEL mode, in degrees/s.
    pos_vel_velocity: float | list[float] = field(default_factory=list)

    # Dynamically lower POS_VEL vlim to follow the commanded joint trajectory
    # continuously. pos_vel_velocity remains the hard upper bound.
    adaptive_pos_vel_velocity: bool = False
    adaptive_pos_vel_feedback: bool = False
    pos_vel_tracking_ratio: float = 1.25
    pos_vel_min_velocity: float = 2.0

    # Damiao POS_VEL loop gains. These are applied at connect time without
    # storing them to motor flash, so power cycling restores persisted values.
    configure_pos_vel_gains: bool = False
    pos_vel_kp_asr: dict[str, float] = field(default_factory=dict)
    pos_vel_ki_asr: dict[str, float] = field(default_factory=dict)
    pos_vel_kp_apr: dict[str, float] = field(default_factory=dict)
    pos_vel_ki_apr: dict[str, float] = field(default_factory=dict)

    ## Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1

    # Soft joint limits in degrees. Concrete subclasses should define defaults.
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Per-joint action direction/scale applied before joint-limit clipping.
    # Use -1 for sign flip, 1 for no flip, and other values when scaling is required.
    joint_directions: dict[str, float] = field(default_factory=dict)


logger = logging.getLogger(__name__)


FOLLOWER_GRIPPER_MOTOR = "gripper"
WRIST_FLEX_MOTOR = "wrist_flex"
EXTRA_STIFFNESS_MOTORS = ("shoulder_lift", "elbow_flex", WRIST_FLEX_MOTOR)
DAMIAO_KP_ASR_REGISTER = 25
LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01


class SeeedB601FollowerBase(Robot):
    """
    Base class for Seeed B601 Follower Arms (DM and RS variants).
    Uses CAN bus communication via motorbridge.
    """

    motor_type: str = ""

    def __init__(self, config: SeeedB601FollowerConfigBase):
        super().__init__(config)
        self.config = config
        self.bus = None
        self.motors = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self._in_safe_zero = False
        self._emergency_disable_requested = False
        self._last_goal_pos_deg: dict[str, float] | None = None
        self._last_action_time: float | None = None
        self._runtime_arm_hold_velocity_deg_s = (
            config.arm_hold_velocity_min
            if self.motor_type == "dm"
            else config.pos_vel_min_velocity
        )
        self._runtime_arm_hold_kp_asr = (
            dict(config.arm_hold_kp_asr_initial)
            if self.motor_type == "dm"
            else dict(config.pos_vel_kp_asr)
        )
        self._runtime_arm_kp_asr_extra = {
            motor_name: 0.0 for motor_name in EXTRA_STIFFNESS_MOTORS
        }
        self._arm_kp_asr_mode = "hold"

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.motor_names:
            features[f"{motor}.pos"] = float
            # features[f"{motor}.vel"] = float
            # features[f"{motor}.torque"] = float
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
    def limit_cartesian_jog_acceleration(self) -> bool:
        return bool(self.config.limit_cartesian_jog_acceleration)

    @property
    def cartesian_jog_accel_m_s2(self) -> float:
        return float(self.config.cartesian_jog_accel_m_s2)

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def _add_motors_to_bus(self):
        """Must be implemented by subclasses to add specific motor types to self.bus."""
        raise NotImplementedError

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the follower arm and optionally calibrate."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(
            f"Connecting arm on {self.config.port} (adapter={self.config.can_adapter})..."
        )
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

        self.bus.disable_all()

        print(
            "\nCalibration: Set Zero Position\n"
            "Please MANUALLY move the robot to its ZERO POSITION, and close its gripper.\n"
            "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(LONG_TIMEOUT_SEC)

        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        self.calibration = {}
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
        # Keep torque off while switching modes, then enable after all motors are configured.
        self.bus.disable_all()
        if self.motor_type == "dm":
            arm_motor_names = {
                name for name in self.motor_names if name != FOLLOWER_GRIPPER_MOTOR
            }
            for field_name, gains in (
                ("arm_hold_kp_asr_initial", self.config.arm_hold_kp_asr_initial),
                ("arm_motion_kp_asr", self.config.arm_motion_kp_asr),
            ):
                if set(gains) != arm_motor_names:
                    raise ValueError(
                        f"{field_name} must contain exactly all six arm motors"
                    )
                if any(value <= 0 for value in gains.values()):
                    raise ValueError(f"{field_name} values must be greater than 0")
            if set(self.config.arm_hold_kp_asr_extra_max) != set(
                EXTRA_STIFFNESS_MOTORS
            ):
                raise ValueError(
                    "arm_hold_kp_asr_extra_max must contain exactly joints 2, 3, and 4"
                )
            if any(
                value < 0 for value in self.config.arm_hold_kp_asr_extra_max.values()
            ):
                raise ValueError("arm_hold_kp_asr_extra_max values must not be negative")
            self._runtime_arm_hold_velocity_deg_s = self.config.arm_hold_velocity_min
            self._runtime_arm_hold_kp_asr = dict(
                self.config.arm_hold_kp_asr_initial
            )
            self._runtime_arm_kp_asr_extra = {
                motor_name: 0.0 for motor_name in EXTRA_STIFFNESS_MOTORS
            }
            self._arm_kp_asr_mode = "hold"
        num_retry = 9
        for motor_name, motor in self.motors.items():
            target_mode = (
                MotorBridgeMode.MIT
                if self.motor_type == "rs"
                else (
                    MotorBridgeMode.FORCE_POS
                    if motor_name == FOLLOWER_GRIPPER_MOTOR
                    else MotorBridgeMode.POS_VEL
                )
            )
            if (
                self.motor_type == "dm"
                and motor_name != FOLLOWER_GRIPPER_MOTOR
                and self.config.configure_pos_vel_gains
            ):
                gains = (
                    (25, "KP_ASR", self.config.pos_vel_kp_asr),
                    (26, "KI_ASR", self.config.pos_vel_ki_asr),
                    (27, "KP_APR", self.config.pos_vel_kp_apr),
                    (28, "KI_APR", self.config.pos_vel_ki_apr),
                )
                for register_id, register_name, values in gains:
                    if motor_name not in values:
                        raise ValueError(
                            f"Missing Damiao POS_VEL {register_name} value for {motor_name}"
                        )
                    value = (
                        self._configured_kp_asr(motor_name)
                        if register_id == DAMIAO_KP_ASR_REGISTER
                        else values[motor_name]
                    )
                    motor.write_register_f32(register_id, value)
                time.sleep(MEDIUM_TIMEOUT_SEC * 2)
                logger.info(
                    "%s POS_VEL gains: KP_ASR=%g KI_ASR=%g KP_APR=%g KI_APR=%g",
                    motor_name,
                    self._configured_kp_asr(motor_name),
                    self.config.pos_vel_ki_asr[motor_name],
                    self.config.pos_vel_kp_apr[motor_name],
                    self.config.pos_vel_ki_apr[motor_name],
                )
            for _ in range(num_retry + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)
            logger.info(f"{motor_name} ensure mode {target_mode}")
        self.bus.enable_all()

    def _configured_kp_asr(self, motor_name: str) -> float:
        if self.motor_type == "dm":
            return self.config.arm_hold_kp_asr_initial[motor_name]
        return self.config.pos_vel_kp_asr[motor_name]

    def disable_torque(self) -> None:
        """Disable follower motor torque so the arm can be moved by hand during read-only debugging."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def adjust_arm_hold_stiffness(
        self,
        steps: int,
    ) -> tuple[float, dict[str, float]]:
        """Adjust all arm joints' hold vlim and speed-loop gain in RAM."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.motor_type != "dm":
            raise RuntimeError(
                "Runtime arm hold-stiffness adjustment requires a B601-DM follower."
            )
        if not isinstance(steps, int):
            raise TypeError(f"steps must be int, got {type(steps).__name__}")

        current_velocity = getattr(
            self,
            "_runtime_arm_hold_velocity_deg_s",
            self.config.arm_hold_velocity_min,
        )
        updated_velocity = max(
            self.config.arm_hold_velocity_min,
            min(
                self.config.arm_hold_velocity_max,
                current_velocity + steps * self.config.arm_hold_velocity_step,
            ),
        )
        current_extra = dict(
            getattr(
                self,
                "_runtime_arm_kp_asr_extra",
                {motor_name: 0.0 for motor_name in EXTRA_STIFFNESS_MOTORS},
            )
        )
        updated_extra = current_extra
        gain_delta = steps * self.config.arm_hold_kp_asr_step
        for motor_name in EXTRA_STIFFNESS_MOTORS:
            updated_extra[motor_name] = max(
                0.0,
                min(
                    self.config.arm_hold_kp_asr_extra_max[motor_name],
                    updated_extra[motor_name] + gain_delta,
                ),
            )

        verified_gains = dict(
            getattr(self, "_runtime_arm_hold_kp_asr", self.config.pos_vel_kp_asr)
        )
        arm_motor_names = [
            name for name in self.motor_names if name != FOLLOWER_GRIPPER_MOTOR
        ]
        for motor_name in arm_motor_names:
            requested_kp_asr = self._configured_kp_asr(motor_name)
            requested_kp_asr += updated_extra.get(motor_name, 0.0)
            if getattr(self, "_arm_kp_asr_mode", "hold") == "motion":
                verified_gains[motor_name] = requested_kp_asr
                continue
            motor = self.motors[motor_name]
            motor.write_register_f32(DAMIAO_KP_ASR_REGISTER, requested_kp_asr)
            try:
                verified_gains[motor_name] = motor.get_register_f32(
                    DAMIAO_KP_ASR_REGISTER,
                    500,
                )
            except Exception:
                verified_gains[motor_name] = requested_kp_asr
                logger.warning(
                    "%s KP_ASR readback failed; using requested runtime value %g.",
                    motor_name,
                    requested_kp_asr,
                )

        self._runtime_arm_hold_velocity_deg_s = updated_velocity
        self._runtime_arm_kp_asr_extra = updated_extra
        self._runtime_arm_hold_kp_asr = verified_gains

        logger.info(
            "Arm hold response: vlim=%g deg/s, KP_ASR=%s (RAM only)",
            updated_velocity,
            ", ".join(
                f"{name}={verified_gains[name]:g}" for name in arm_motor_names
            ),
        )
        return updated_velocity, verified_gains

    def set_arm_motion_active(self, active: bool) -> None:
        """Select motion or hold KP_ASR gains, writing only on mode changes."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.motor_type != "dm":
            return

        requested_mode = "motion" if active else "hold"
        if getattr(self, "_arm_kp_asr_mode", "hold") == requested_mode:
            return

        gains = (
            self.config.arm_motion_kp_asr
            if active
            else self._runtime_arm_hold_kp_asr
        )
        arm_motor_names = [
            name for name in self.motor_names if name != FOLLOWER_GRIPPER_MOTOR
        ]
        for motor_name in arm_motor_names:
            self.motors[motor_name].write_register_f32(
                DAMIAO_KP_ASR_REGISTER,
                gains[motor_name],
            )

        self._arm_kp_asr_mode = requested_mode
        logger.info(
            "Arm KP_ASR mode: %s (%s)",
            requested_mode,
            ", ".join(f"{name}={gains[name]:g}" for name in arm_motor_names),
        )

    def _read_motor_temperatures(self) -> dict[str, float]:
        """Read per-motor MOS temperatures once and return available values."""
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("Temperature check poll feedback failed.")

        temps: dict[str, float] = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                temps[motor_name] = state.t_mos

        return temps

    def mit_output_torque_limit(
        self,
        motor: Any,
        pos_target_rad: float,
    ) -> float | None:
        """Compute MIT torque command from target position and motor state."""
        return 0.0

    def safe_zero(
        self, step_interval_s: float = 0.02, exit_on_complete: bool = True
    ) -> None:
        """Move arm joints back to zero in a safer two-stage interpolation.

        Stage 1: CAN ID 1/4/5/6 -> 0
        Stage 2: CAN ID 2/3 -> 0
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._in_safe_zero:
            logger.warning("safe_zero skipped: already running.")
            return

        if step_interval_s < 0.0:
            raise ValueError("step_interval_s must be >= 0")

        self._in_safe_zero = True
        try:
            id_to_joint: dict[int, str] = {
                send_id: motor_name
                for motor_name, (send_id, _) in self.config.motor_can_ids.items()
                if motor_name != FOLLOWER_GRIPPER_MOTOR
            }

            stage_1 = [id_to_joint[i] for i in (1, 4, 5, 6) if i in id_to_joint]
            stage_2 = [id_to_joint[i] for i in (2, 3) if i in id_to_joint]
            controlled_joints = stage_1 + [
                name for name in stage_2 if name not in stage_1
            ]

            if not controlled_joints:
                logger.warning(
                    "safe_zero skipped: no arm joints mapped to CAN IDs 1-6."
                )
                return

            def _read_action_pos(joint_name: str) -> float:
                motor = self.motors.get(joint_name)
                if motor is None:
                    raise RuntimeError(
                        f"safe_zero failed: motor '{joint_name}' not found"
                    )

                max_retry = 10
                for attempt in range(1, max_retry + 1):
                    try:
                        motor.request_feedback()
                        self.bus.poll_feedback_once()
                    except Exception:
                        logger.debug(
                            "safe_zero feedback poll failed for %s (attempt %d/%d)",
                            joint_name,
                            attempt,
                            max_retry,
                        )

                    state = motor.get_state()
                    if state is not None:
                        current_deg = math.degrees(state.pos)
                        direction = (
                            self.config.joint_directions.get(joint_name, 1.0) or 1.0
                        )
                        return current_deg / direction

                    if attempt < max_retry:
                        time.sleep(MEDIUM_TIMEOUT_SEC)

                raise RuntimeError(
                    f"safe_zero failed: unable to read state for '{joint_name}' after {max_retry} attempts"
                )

            def _frame_count(starts: dict[str, float]) -> int:
                max_delta_deg = max((abs(v) for v in starts.values()), default=0.0)
                return max(1, math.ceil(max_delta_deg * 2.0))

            def _interp_to_zero(
                active_starts: dict[str, float], hold_joints: dict[str, float]
            ) -> bool:
                if not active_starts:
                    return False
                frames = _frame_count(active_starts)
                emergency_disable_threshold_c = 135.0
                for frame in range(1, frames + 1):
                    temperatures = self._read_motor_temperatures()
                    for motor_name, temp_c in temperatures.items():
                        if temp_c > emergency_disable_threshold_c:
                            logger.error(
                                "Auto-disable on overtemperature during safe_zero: %s t_mos=%.2fC > %.2fC.",
                                motor_name,
                                temp_c,
                                emergency_disable_threshold_c,
                            )
                            self._emergency_disable_requested = True
                            self.disable_torque()
                            logger.error(
                                "safe_zero aborted: emergency overtemperature."
                            )
                            return True

                    ratio = frame / frames
                    action: RobotAction = {}
                    for joint, start in hold_joints.items():
                        action[f"{joint}.pos"] = start
                    for joint, start in active_starts.items():
                        action[f"{joint}.pos"] = start * (1.0 - ratio)
                    self.send_action(action)
                    if step_interval_s > 0.0:
                        time.sleep(step_interval_s)

                return False

            stage_1_start = {joint: _read_action_pos(joint) for joint in stage_1}
            stage_2_start = {joint: _read_action_pos(joint) for joint in stage_2}

            logger.info("safe_zero stage1 start: joints=%s", stage_1)
            if _interp_to_zero(stage_1_start, stage_2_start):
                return
            logger.info("safe_zero stage2 start: joints=%s", stage_2)
            if _interp_to_zero(stage_2_start, {joint: 0.0 for joint in stage_1}):
                return
            logger.info("safe_zero done.")
            time.sleep(2.0)
            if exit_on_complete:
                # Raise KeyboardInterrupt so upper-level control loops handle this
                # the same way as Ctrl+C.
                raise KeyboardInterrupt("safe_zero completed")
        finally:
            self._in_safe_zero = False

    def get_observation(self) -> RobotObservation:
        """Get current observation from robot."""
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Request and poll feedback from motorbridge
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("CAN bus poll feedback failed.")

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
        # logger.debug(f"Observation: {obs_dict}")

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action command to robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._in_safe_zero:
            alarm_threshold_c = 80.0
            overheat_threshold_c = 100.0
            temperatures = self._read_motor_temperatures()
            for motor_name, temp_c in temperatures.items():
                if temp_c > alarm_threshold_c:
                    print(
                        f"[HIGH TEMP] {motor_name} t_mos={temp_c:.2f}C > {alarm_threshold_c:.2f}C"
                    )
                if temp_c > overheat_threshold_c:
                    logger.error(
                        "Overheat detected in send_action: %s t_mos=%.2fC > %.2fC.",
                        motor_name,
                        temp_c,
                        overheat_threshold_c,
                    )
                    raise KeyboardInterrupt("Overheat detected")

        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # Apply per-joint direction/scale mapping before clipping.
        for motor_name, position in goal_pos.items():
            direction = self.config.joint_directions.get(motor_name, 0.0)
            position = position * direction
            # print(f"motor_name: {motor_name}, position: {position}")
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(
                        f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}"
                    )
                position = clipped_position

            goal_pos[motor_name] = position

        # To tolerate 6-DOF leader arms that don't have a wrist_yaw joint, we can allow the follower to ignore missing wrist_yaw commands by treating them as 0.
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

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

            goal_present_pos = {
                key: (g_pos, present_pos.get(key, g_pos))
                for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        command_time = time.perf_counter()
        last_action_time = getattr(self, "_last_action_time", None)
        control_dt_s = (
            None
            if last_action_time is None
            else min(max(command_time - last_action_time, 1e-3), 0.1)
        )
        previous_goal_pos = getattr(self, "_last_goal_pos_deg", None)

        # Prepare and send commands
        for motor_name, position_degrees in goal_pos.items():
            try:
                idx = self.motor_names.index(motor_name)
            except ValueError:
                idx = 0  # Fallback

            # Convert target position from degrees to radians for motorbridge
            pos_rad = math.radians(position_degrees)
            vel_deg_s = (
                self.config.pos_vel_velocity[idx]
                if isinstance(self.config.pos_vel_velocity, list)
                else self.config.pos_vel_velocity
            )
            motor = self.motors.get(motor_name)
            if (
                self.motor_type == "dm"
                and motor_name != FOLLOWER_GRIPPER_MOTOR
                and self.config.adaptive_pos_vel_velocity
                and control_dt_s is not None
                and previous_goal_pos is not None
                and motor_name in previous_goal_pos
            ):
                command_delta_deg = abs(
                    position_degrees - previous_goal_pos[motor_name]
                )
                requested_velocity = command_delta_deg / control_dt_s
                if (
                    self.config.adaptive_pos_vel_feedback
                    and command_delta_deg > 1e-9
                    and motor is not None
                ):
                    # Feedback compensation is useful while following a moving
                    # trajectory. Once the target is held, feeding encoder
                    # error back into vlim makes tiny load/encoder deviations
                    # repeatedly accelerate the joint and can cause visible
                    # pitch/yaw hunting around a stationary posture.
                    state = motor.get_state()
                    state_position = getattr(state, "pos", None)
                    if isinstance(state_position, int | float):
                        tracking_velocity = (
                            abs(position_degrees - math.degrees(state_position))
                            / control_dt_s
                        )
                        requested_velocity = max(requested_velocity, tracking_velocity)
                min_velocity = min(vel_deg_s, self.config.pos_vel_min_velocity)
                if command_delta_deg <= 1e-9:
                    min_velocity = min(
                        vel_deg_s,
                        getattr(
                            self,
                            "_runtime_arm_hold_velocity_deg_s",
                            self.config.arm_hold_velocity_min,
                        ),
                    )
                vel_deg_s = min(
                    vel_deg_s,
                    max(
                        min_velocity,
                        requested_velocity * self.config.pos_vel_tracking_ratio,
                    ),
                )
            vel_rad = math.radians(vel_deg_s)

            if motor is not None:
                if motor_name == FOLLOWER_GRIPPER_MOTOR:
                    if self.motor_type == "rs":
                        tau_ff = self.mit_output_torque_limit(motor, pos_rad)
                        if tau_ff is None:
                            tau_ff = 0.0
                        motor.send_mit(0, 0, 0, 1.5, tau_ff)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: pos={position_degrees:.2f}°, "
                            f"tau_ff={tau_ff:.2f}"
                        )
                    else:
                        motor.send_force_pos(
                            pos_rad, vel_rad, self.config.force_pos_torque_ration
                        )
                        logger.debug(
                            f"Sent FORCE_POS command to {motor_name}: pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s, ratio={0.1}"
                        )
                else:
                    if self.motor_type == "rs":
                        kp = getattr(self.config, "mit_kp", {}).get(motor_name, 0.0)
                        kd = getattr(self.config, "mit_kd", {}).get(motor_name, 0.0)
                        motor.send_mit(pos_rad, 0, kp, kd, 0)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: "
                            f"pos={position_degrees:.2f}°, kp={kp}, kd={kd}"
                        )
                    else:
                        motor.send_pos_vel(pos_rad, vel_rad)
                        logger.debug(
                            f"Sent POS_VEL command to {motor_name}: target={pos_rad:.2f}, "
                            f"pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s"
                        )

        self._last_goal_pos_deg = dict(goal_pos)
        self._last_action_time = command_time

        # motorbridge sends packets mostly synchronously here over loop,
        # so we don't need a bulk send command through ctypes.

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._in_safe_zero and not self._emergency_disable_requested:
            try:
                self.safe_zero(exit_on_complete=False)
            except Exception:
                logger.exception("safe_zero during disconnect failed.")

        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()

        self.bus.close()
        self.bus = None

        for cam in self.cameras.values():
            cam.disconnect()

        self._emergency_disable_requested = False
        logger.info(f"{self} disconnected.")
