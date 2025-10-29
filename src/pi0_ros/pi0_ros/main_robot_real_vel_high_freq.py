#!/usr/bin/env python3
import threading
import time
from typing import List, Optional, Union

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_srvs.srv import Trigger

from pi0_ros.utils.utils_ros import stamp_to_seconds, time_to_seconds

class RobotRealVelHighFreq(Node):

    def __init__(self):
        super().__init__("robot_real_vel_high_freq")
        # -------- hyper-parameters begin --------
        self.n_arm_joints = 7
        self.hf_commands_window_size = 5
        self.timer_period = 0.01  # seconds
        # -------- hyper-parameters end --------

        self.sent_hf_commands_window = []
        self.sent_hf_commands_stamp_window = []
        self.joint_vel_command_high_freq = None
        self.sent_num_command_high_freq = 0

        self.srv = self.create_service(
            Trigger,
            "robot/reset_high_freq_node",
            self._trigger_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.lock = threading.Lock()
        self.arm_joint_vel_command_pub = self.create_publisher(Float64MultiArray, "franka/joint_velocity_command", 1)
        self.joint_vel_command_sub = self.create_subscription(
            JointState,
            "robot/joint_vel_command_low_freq",
            self._robot_command_callback,
            1,
            callback_group=ReentrantCallbackGroup(),
        )

        self.timer = self.create_timer(self.timer_period, self._timer_callback, callback_group=ReentrantCallbackGroup())
        self.get_logger().info("Ready ...")

    def _trigger_callback(self, request, response):
        self.get_logger().info("Received reset request.")
        # reset the window
        self._reset_window()

        response.success = True
        response.message = "Reset high_freq node successfully!"
        return response

    def _reset_window(self):
        self.sent_hf_commands_window = []
        self.sent_hf_commands_stamp_window = []
        self.joint_vel_command_high_freq = None
        self.sent_num_command_high_freq = 0

    def _publish_arm_joint_vel_command(self, joint_vel):
        assert len(joint_vel) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_vel]
        self.arm_joint_vel_command_pub.publish(msg)

    def _robot_command_callback(self, msg):
        self.get_logger().info(f"Received low-freqency joint command: {list(msg.velocity)}")

        current_time = time_to_seconds(self.get_clock().now())
        target_command = np.asarray(msg.velocity)
        target_reach_time = stamp_to_seconds(msg.header.stamp)

        if len(self.sent_hf_commands_window) >= 2:
            self.lock.acquire()
            xs = self.sent_hf_commands_stamp_window + [target_reach_time]
            ys = self.sent_hf_commands_window + [target_command]
            self.lock.release()
            spline_func = CubicSpline(xs, ys, bc_type="natural")
            x_new = np.arange(current_time, target_reach_time, self.timer_period)
            y_new = spline_func(x_new)
        else:
            y_new = np.asarray([target_command])

        self.lock.acquire()
        self.joint_vel_command_high_freq = y_new
        self.sent_num_command_high_freq = 0
        self.lock.release()

    def _timer_callback(self):
        self.lock.acquire()

        if (self.joint_vel_command_high_freq is not None) and self.sent_num_command_high_freq < len(
            self.joint_vel_command_high_freq
        ):
            command = self.joint_vel_command_high_freq[self.sent_num_command_high_freq]
            self._publish_arm_joint_vel_command(command[: self.n_arm_joints])
            self.sent_num_command_high_freq += 1

            # maintain the window
            self.sent_hf_commands_window.append(command)
            self.sent_hf_commands_stamp_window.append(time_to_seconds(self.get_clock().now()))
            if len(self.sent_hf_commands_window) > self.hf_commands_window_size:
                self.sent_hf_commands_window.pop(0)
                self.sent_hf_commands_stamp_window.pop(0)

            self.get_logger().debug(f"Sending high-frequency command: {command}.")

            # print("HF command: ", command)
            # print("time.time(): ", time.time())

        self.lock.release()


def main(args=None):
    rclpy.init(args=args)

    robot = RobotRealVelHighFreq()

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
