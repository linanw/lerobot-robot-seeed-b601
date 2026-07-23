import math
from unittest.mock import MagicMock, call

import pytest

from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import (
    SeeedB601DMFollowerConfig,
)
from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower


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
