# Seeed reBot Arm B601 Follower Integration with LeRobot

This repository provides the integration for the **reBot Arm B601** with the [LeRobot](https://github.com/huggingface/lerobot) framework. It enables the B601 arm to be used as the follower.

## Supported Hardware

*   **Robot**: Seeed reBot Arm B601 Series (6-DOF + Gripper)
*   **Motors**: Damiao (DM4340 + DM4310), RobStride
*   **Communication**: CAN Bus (via USB-CAN adapter, e.g., derivatives of Candle/slcan or SocketCAN compatible devices)

## Installation

1.  **Install LeRobot**:
    Follow the instructions in the [LeRobot repository](https://github.com/huggingface/lerobot) to install the base library.

2.  **Install this package**:
    Clone this repository and install in editable mode:
    ```bash
    git clone https://github.com/Seeed-Studio/lerobot-robot-seeed-b601.git
    cd lerobot-robot-seeed-b601
    pip install -e .
    ```

    Or install from PyPI:
    ```bash
    pip install lerobot-robot-seeed-b601
    ```

    Upon the installation, the Seeed reBot Arm B601 will be registered in LeRobot. Two variants are registered:
    *   `lerobot_robot_seeed_b601_dm`: B601 using Damiao motors (6-DOF + Gripper)
    *   `lerobot_robot_seeed_b601_rs`: B601 using RobStride motors (6-DOF + Gripper)

## Configuration

The default configuration for B601-DM is located in `lerobot_robot_seeed_b601/config_seeed_b601_dm.py`.

*   **Motor IDs**: 0x01 - 0x06 (Joints), 0x07 (Gripper)
*   **Motor Types**:
    *   Joint 1, 4, 5, 6, Gripper: `dm4310`
    *   Joint 2, 3: `dm4340`

Ensure your robot's motor IDs match this configuration.

TODO: Refer to the wiki page where shows the guide to configure the motors.

## Usage

### Quick Start

A simple verification script is provided to test the connection and basic motor control without loading the full LeRobot gym environment.

**1. Connect to the robot:**
```bash
# Replace /dev/tty.usbmodem* with your actual CAN port (e.g., can0 on Linux)
python examples/verification_dm.py --port /dev/tty.usbmodem12345 --action connect
```

**2. Read motor states:**
```bash
python examples/verification_dm.py --port /dev/tty.usbmodem12345 --action read
```
This will continuously print the current joint positions. Manually move the robot arm to verify that the readings change.

**3. Move a joint (Caution):**
```bash
# Example: Move Joint 6 to 5 degrees
python examples/verification_dm.py --port /dev/tty.usbmodem12345 --action move --joint joint_6 --angle 5
```

### Use within the LeRobot

TODO: Write the guide to use this robot within the LeRobot.
