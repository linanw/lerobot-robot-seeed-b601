from .config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
from .seeed_b601_follower import SeeedB601FollowerBase


class SeeedB601DMFollower(SeeedB601FollowerBase):
    """
    Seeed B601-DM Robot Arm (6-DOF + Gripper) using Damiao motors.
    Uses CAN bus communication via motorbridge SDK.
    """

    config_class = SeeedB601DMFollowerConfig
    name = "seeed_b601_dm_follower"

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            motor_type_str = self.config.motor_models[motor_name]
            model_str = motor_type_str.upper().replace("DM", "")
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, model_str)
