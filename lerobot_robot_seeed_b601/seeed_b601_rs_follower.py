from .config_seeed_b601_rs_follower import SeeedB601RSFollowerConfig
from .seeed_b601_follower import SeeedB601FollowerBase


class SeeedB601RSFollower(SeeedB601FollowerBase):
    """
    Seeed B601-RS Robot Arm (RobStride Motors).
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601RSFollowerConfig
    name = "seeed_b601_rs_follower"

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.config.motor_models[motor_name]
            # Assumes RobStride uses something like "04"
            self.motors[motor_name] = self.bus.add_robstride_motor(send_id, recv_id, motor_type_str)
