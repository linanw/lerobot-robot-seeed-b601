import math
from unittest.mock import MagicMock, call

import pytest

from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import (
    SeeedB601DMFollowerConfig,
)
from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower


SCRIPT_INITIAL_GAINS = {
    "shoulder_pan": 0.0157,
    "shoulder_lift": 0.0197,
    "elbow_flex": 0.0197,
    "wrist_flex": 0.012,
    "wrist_yaw": 0.004,
    "wrist_roll": 0.004,
}


def test_dm_arm_uses_configured_pos_vel_velocity_on_first_command() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(
        port="/dev/null",
        pos_vel_velocity=[75.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0],
    )
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False

    robot.send_action({"shoulder_pan.pos": 10.0})

    robot.motors["shoulder_pan"].send_pos_vel.assert_called_once_with(
        pytest.approx(math.radians(-10.0)),
        pytest.approx(math.radians(75.0)),
    )


def test_dm_arm_adapts_pos_vel_limit_to_joint_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_times = iter((10.0, 10.02))
    monkeypatch.setattr(
        "lerobot_robot_seeed_b601.seeed_b601_follower.time.perf_counter",
        lambda: next(command_times),
    )
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False
    robot._last_goal_pos_deg = None
    robot._last_action_time = None

    robot.send_action({"shoulder_pan.pos": 10.0})
    robot.send_action({"shoulder_pan.pos": 10.1})

    calls = robot.motors["shoulder_pan"].send_pos_vel.call_args_list
    assert calls[0] == call(
        pytest.approx(math.radians(-10.0)),
        pytest.approx(math.radians(150.0)),
    )
    # 0.1 degrees in 20 ms is 5 deg/s. The 1.05 tracking margin produces
    # a synchronized 5.25 deg/s vlim instead of the fixed 150 deg/s ceiling.
    assert calls[1] == call(
        pytest.approx(math.radians(-10.1)),
        pytest.approx(math.radians(5.25)),
    )


def test_dm_arm_preserves_synchronized_vlim_at_small_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_times = iter((20.0, 20.02))
    monkeypatch.setattr(
        "lerobot_robot_seeed_b601.seeed_b601_follower.time.perf_counter",
        lambda: next(command_times),
    )
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False
    robot._last_goal_pos_deg = None
    robot._last_action_time = None

    robot.send_action({"shoulder_lift.pos": 10.0, "elbow_flex.pos": -20.0})
    robot.send_action({"shoulder_lift.pos": 10.008, "elbow_flex.pos": -19.998})

    shoulder_vlim = robot.motors["shoulder_lift"].send_pos_vel.call_args_list[1].args[1]
    elbow_vlim = robot.motors["elbow_flex"].send_pos_vel.call_args_list[1].args[1]
    # Physical rates are 0.4 and 0.1 deg/s. Keeping the same 1.05 ratio gives
    # both joints the same arrival time, using about 95% of the 20 ms frame.
    assert math.degrees(shoulder_vlim) == pytest.approx(0.42)
    assert math.degrees(elbow_vlim) == pytest.approx(0.105)
    assert 0.008 / math.degrees(shoulder_vlim) == pytest.approx(
        0.002 / math.degrees(elbow_vlim)
    )


def test_dm_arm_raises_vlim_only_for_joint_with_feedback_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_times = iter((30.0, 30.02))
    monkeypatch.setattr(
        "lerobot_robot_seeed_b601.seeed_b601_follower.time.perf_counter",
        lambda: next(command_times),
    )
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False
    robot._last_goal_pos_deg = None
    robot._last_action_time = None

    robot.send_action({"shoulder_lift.pos": 10.0, "wrist_flex.pos": 20.0})
    # Physical coordinates after direction mapping: shoulder target -10.008,
    # wrist target 20.002. Shoulder is still 0.02 degrees behind while wrist
    # has reached its previous target.
    robot.motors["shoulder_lift"].get_state.return_value.pos = math.radians(-9.988)
    robot.motors["wrist_flex"].get_state.return_value.pos = math.radians(20.0)
    robot.send_action({"shoulder_lift.pos": 10.008, "wrist_flex.pos": 20.002})

    shoulder_vlim = math.degrees(
        robot.motors["shoulder_lift"].send_pos_vel.call_args_list[1].args[1]
    )
    wrist_vlim = math.degrees(
        robot.motors["wrist_flex"].send_pos_vel.call_args_list[1].args[1]
    )
    assert shoulder_vlim == pytest.approx(1.05)
    assert wrist_vlim == pytest.approx(0.105)


def test_dm_arm_does_not_chase_feedback_error_while_target_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_times = iter((40.0, 40.02))
    monkeypatch.setattr(
        "lerobot_robot_seeed_b601.seeed_b601_follower.time.perf_counter",
        lambda: next(command_times),
    )
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False
    robot._last_goal_pos_deg = None
    robot._last_action_time = None

    robot.send_action({"wrist_flex.pos": 20.0})
    # Simulate a small stationary-position error from load, encoder noise, or
    # overshoot. The outer feedback loop must not turn it into repeated vlim
    # acceleration when the commanded target itself has not moved.
    robot.motors["wrist_flex"].get_state.return_value.pos = math.radians(19.9)
    robot.send_action({"wrist_flex.pos": 20.0})

    held_vlim = math.degrees(
        robot.motors["wrist_flex"].send_pos_vel.call_args_list[1].args[1]
    )
    assert held_vlim == pytest.approx(robot.config.pos_vel_min_velocity)
    robot.motors["wrist_flex"].get_state.assert_not_called()


def test_dm_configure_applies_seeed_pos_vel_gains_without_storing() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}

    robot.configure()

    shoulder = robot.motors["shoulder_lift"]
    assert shoulder.write_register_f32.call_args_list == [
        call(25, 0.0125),
        call(26, 0.004),
        call(27, 150.0),
        call(28, 0.5),
    ]
    wrist = robot.motors["wrist_flex"]
    assert wrist.write_register_f32.call_args_list == [
        call(25, 0.0008),
        call(26, 0.002),
        call(27, 50.0),
        call(28, 1.0),
    ]
    robot.motors["gripper"].write_register_f32.assert_not_called()
    for motor in robot.motors.values():
        motor.store_parameters.assert_not_called()


def test_dm_configure_applies_all_initial_arm_kp_asr_values() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(
        port="/dev/null",
        arm_hold_kp_asr_initial=SCRIPT_INITIAL_GAINS,
    )
    robot.bus = MagicMock()
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}

    robot.configure()

    for motor_name, initial_gain in SCRIPT_INITIAL_GAINS.items():
        assert robot.motors[motor_name].write_register_f32.call_args_list[0] == call(
            25, initial_gain
        )
        assert robot._runtime_arm_hold_kp_asr[motor_name] == pytest.approx(
            initial_gain
        )


def test_arm_hold_stiffness_adjustment_updates_only_joints_2_3_4() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._runtime_arm_hold_velocity_deg_s = robot.config.arm_hold_velocity_min
    robot._runtime_arm_hold_kp_asr = dict(robot.config.arm_hold_kp_asr_initial)
    robot._runtime_arm_kp_asr_extra = {
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
    }
    arm_names = [name for name in robot.motor_names if name != "gripper"]
    for name in arm_names:
        base = robot.config.arm_hold_kp_asr_initial[name]
        adjusted = base + 0.0004 if name in robot._runtime_arm_kp_asr_extra else base
        robot.motors[name].get_register_f32.side_effect = [
            adjusted,
            base,
        ]

    velocity, gains = robot.adjust_arm_hold_stiffness(1)
    assert velocity == pytest.approx(0.52)
    assert gains["shoulder_pan"] == pytest.approx(0.0125)
    assert gains["shoulder_lift"] == pytest.approx(0.0129)
    assert gains["elbow_flex"] == pytest.approx(0.0129)
    assert gains["wrist_flex"] == pytest.approx(0.0012)
    assert gains["wrist_yaw"] == pytest.approx(0.0008)

    velocity, gains = robot.adjust_arm_hold_stiffness(-2)
    assert velocity == pytest.approx(0.02)
    assert gains == pytest.approx(robot.config.arm_hold_kp_asr_initial)

    for name in arm_names:
        base = robot.config.arm_hold_kp_asr_initial[name]
        adjusted = base + 0.0004 if name in robot._runtime_arm_kp_asr_extra else base
        assert robot.motors[name].write_register_f32.call_args_list == [
            call(25, pytest.approx(adjusted)),
            call(25, pytest.approx(base)),
        ]
        robot.motors[name].store_parameters.assert_not_called()
    robot.motors["gripper"].write_register_f32.assert_not_called()


def test_arm_hold_stiffness_adjustment_clamps_to_safe_range() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(
        port="/dev/null",
        arm_hold_kp_asr_initial=SCRIPT_INITIAL_GAINS,
    )
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._runtime_arm_hold_velocity_deg_s = robot.config.arm_hold_velocity_min
    robot._runtime_arm_hold_kp_asr = dict(robot.config.arm_hold_kp_asr_initial)
    robot._runtime_arm_kp_asr_extra = {
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
    }
    arm_names = [name for name in robot.motor_names if name != "gripper"]
    for name in arm_names:
        base = robot.config.arm_hold_kp_asr_initial[name]
        maximum = base + (0.004 if name in robot._runtime_arm_kp_asr_extra else 0.0)
        one_step_down = base + (
            0.0036 if name in robot._runtime_arm_kp_asr_extra else 0.0
        )
        robot.motors[name].get_register_f32.side_effect = [
            maximum,
            one_step_down,
            base,
        ]

    velocity, gains = robot.adjust_arm_hold_stiffness(100)
    assert velocity == 5.0
    assert gains["shoulder_pan"] == pytest.approx(0.0157)
    assert gains["shoulder_lift"] == pytest.approx(0.0237)
    assert gains["elbow_flex"] == pytest.approx(0.0237)
    assert gains["wrist_flex"] == pytest.approx(0.016)
    assert gains["wrist_yaw"] == pytest.approx(0.004)

    velocity, gains = robot.adjust_arm_hold_stiffness(-1)
    assert velocity == 4.5
    assert gains["shoulder_pan"] == pytest.approx(0.0157)
    assert gains["shoulder_lift"] == pytest.approx(0.0233)
    assert gains["elbow_flex"] == pytest.approx(0.0233)
    assert gains["wrist_flex"] == pytest.approx(0.0156)
    assert gains["wrist_yaw"] == pytest.approx(0.004)

    velocity, gains = robot.adjust_arm_hold_stiffness(-100)
    assert velocity == 0.02
    assert gains == pytest.approx(SCRIPT_INITIAL_GAINS)
    assert robot.motors["wrist_flex"].write_register_f32.call_args_list == [
        call(25, pytest.approx(0.016)),
        call(25, pytest.approx(0.0156)),
        call(25, pytest.approx(0.012)),
    ]
    assert robot.motors["shoulder_lift"].write_register_f32.call_args_list == [
        call(25, pytest.approx(0.0237)),
        call(25, pytest.approx(0.0233)),
        call(25, pytest.approx(0.0197)),
    ]


def test_arm_kp_asr_switches_once_per_motion_hold_transition() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(
        port="/dev/null",
        arm_hold_kp_asr_initial=SCRIPT_INITIAL_GAINS,
    )
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._runtime_arm_hold_kp_asr = dict(SCRIPT_INITIAL_GAINS)
    robot._arm_kp_asr_mode = "hold"
    arm_names = [name for name in robot.motor_names if name != "gripper"]

    robot.set_arm_motion_active(True)
    robot.set_arm_motion_active(True)
    for name in arm_names:
        robot.motors[name].write_register_f32.assert_called_once_with(
            25,
            robot.config.arm_motion_kp_asr[name],
        )

    robot.set_arm_motion_active(False)
    for name in arm_names:
        assert robot.motors[name].write_register_f32.call_args_list[-1] == call(
            25,
            SCRIPT_INITIAL_GAINS[name],
        )
        assert robot.motors[name].write_register_f32.call_count == 2
    robot.motors["gripper"].write_register_f32.assert_not_called()


def test_home_updates_pending_hold_gains_without_overwriting_motion_gains() -> None:
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(
        port="/dev/null",
        arm_hold_kp_asr_initial=SCRIPT_INITIAL_GAINS,
    )
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._runtime_arm_hold_velocity_deg_s = robot.config.arm_hold_velocity_min
    robot._runtime_arm_hold_kp_asr = dict(SCRIPT_INITIAL_GAINS)
    robot._runtime_arm_kp_asr_extra = {
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
    }
    robot._arm_kp_asr_mode = "motion"

    _, hold_gains = robot.adjust_arm_hold_stiffness(1)

    assert hold_gains["shoulder_lift"] == pytest.approx(0.0201)
    assert hold_gains["wrist_flex"] == pytest.approx(0.0124)
    for motor in robot.motors.values():
        motor.write_register_f32.assert_not_called()


def test_all_stationary_arm_targets_use_adjustable_hold_vlim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_times = iter((50.0, 50.02))
    monkeypatch.setattr(
        "lerobot_robot_seeed_b601.seeed_b601_follower.time.perf_counter",
        lambda: next(command_times),
    )
    robot = SeeedB601DMFollower.__new__(SeeedB601DMFollower)
    robot.config = SeeedB601DMFollowerConfig(port="/dev/null")
    robot.bus = MagicMock()
    robot.cameras = {}
    robot.motor_names = list(robot.config.motor_can_ids)
    robot.motors = {name: MagicMock() for name in robot.motor_names}
    robot._in_safe_zero = True
    robot._emergency_disable_requested = False
    robot._last_goal_pos_deg = None
    robot._last_action_time = None
    robot._runtime_arm_hold_velocity_deg_s = 2.5

    action = {"shoulder_pan.pos": 5.0, "wrist_yaw.pos": 10.0}
    robot.send_action(action)
    robot.send_action(action)

    for name in ("shoulder_pan", "wrist_yaw"):
        held_vlim = math.degrees(
            robot.motors[name].send_pos_vel.call_args_list[1].args[1]
        )
        assert held_vlim == pytest.approx(2.5)
