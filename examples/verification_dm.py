import argparse
import logging
import time

from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verification_dm")


def test_connection(port):
    logger.info(f"Testing connection on port {port}")
    config = SeeedB601DMFollowerConfig(port=port)
    robot = SeeedB601DMFollower(config)
    
    try:
        robot.connect(calibrate=False)
        logger.info("Connection successful!")
        
        # Read states
        logger.info("Reading states...")
        obs = robot.get_observation()
        for key, value in obs.items():
            if "img" not in key: # Skip images for cleaner log
                print(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_read_loop(port, duration=10):
    logger.info(f"Starting read loop for {duration} seconds...")
    config = SeeedB601DMFollowerConfig(port=port)
    robot = SeeedB601DMFollower(config)
    
    try:
        robot.connect(calibrate=False)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            obs = robot.get_observation()
            # Print joint positions
            positions = {k: v for k, v in obs.items() if "pos" in k}
            print(f"\rPositions: {positions}", end="")
            time.sleep(0.1)
        print("\nRead loop finished.")
        
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_move_joint(port, joint_name, target_angle):
    logger.info(f"Moving {joint_name} to {target_angle} degrees...")
    config = SeeedB601DMFollowerConfig(port=port)
    robot = SeeedB601DMFollower(config)
    
    try:
        robot.connect(calibrate=False)
        
        # Read current position to initialize action
        obs = robot.get_observation()
        current_action = {k.replace(".pos", ""): v for k, v in obs.items() if "pos" in k}
        
        # Update target joint
        if joint_name not in current_action:
            logger.error(f"Joint {joint_name} not found in observation")
            return
            
        current_action[joint_name] = float(target_angle)
        
        # Send action
        # Note: In real control loop this should be interpolated. 
        # Here we just step command (be careful with large steps!)
        logger.warning("Sending step command! Ensure target is safe.")
        robot.send_action(current_action)
        
        # Monitor for 2 seconds
        for _ in range(20):
            obs = robot.get_observation()
            current_pos = obs.get(f"{joint_name}.pos", "N/A")
            print(f"\rCurrent {joint_name}: {current_pos}", end="")
            time.sleep(0.1)
        print("\nMove command finished.")

    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Seeed B601-DM functionality")
    parser.add_argument("--port", type=str, required=True, help="CAN interface port (e.g., can0 or /dev/tty.usbmodem...)")
    parser.add_argument("--action", type=str, choices=["connect", "read", "move"], required=True, help="Action to perform")
    parser.add_argument("--joint", type=str, help="Joint name for move action")
    parser.add_argument("--angle", type=float, help="Target angle for move action")
    
    args = parser.parse_args()
    
    if args.action == "connect":
        test_connection(args.port)
    elif args.action == "read":
        test_read_loop(args.port)
    elif args.action == "move":
        if not args.joint or args.angle is None:
            print("Error: --joint and --angle are required for 'move' action")
        else:
            test_move_joint(args.port, args.joint, args.angle)
