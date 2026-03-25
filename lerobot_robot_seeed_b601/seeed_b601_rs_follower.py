from .config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from .seeed_b601_follower import SeeedB601FollowerBase


class SeeedB601RSFollower(SeeedB601FollowerBase):
    """
    Seeed B601-RS Robot Arm (RobStride Motors).
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601RSFollowerConfig
    name = "seeed_b601_rs_follower"

    motor_model_mapping = {
        "shoulder_pan":  "rs-04",
        "shoulder_lift": "rs-04",
        "elbow_flex":    "rs-04",
        "wrist_flex":    "rs-02",
        "wrist_roll":    "rs-02",
        "wrist_yaw":     "rs-02",
        "gripper":       "rs-02",
    }

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.motor_model_mapping[motor_name]
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, motor_type_str.upper())
