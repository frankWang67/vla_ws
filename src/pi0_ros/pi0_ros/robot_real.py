#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.srv import SetJointStiffnessDamping
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from my_franka_msgs.msg import GripperGrasp
from std_msgs.msg import Float64MultiArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand

from pi0_ros.utils.utils_ros import seconds_to_stamp

def time_to_seconds(time) -> float:
    return time.nanoseconds / 1e9


class RobotReal:
    """
    Panda + original gripper
    """

    def __init__(self, node: Node, ctrl_type, gripper_type="original", use_high_freq_interp=False):
        # ------------ hyper-parameters ------------

        # change to 10Hz for collecting demo by lkc
        self.timestep = 0.1

        self.n_arm_joints = 7
        self.arm_joint_names = [f"panda_joint{i+1}" for i in range(self.n_arm_joints)]
        self.n_joints = self.n_arm_joints + 1
        self.gripper_communi_mode = "ACTION"
        # ------------ hyper-parameters ------------

        # variable
        self.curr_joint_pos = np.array(
            [0, -np.pi / 4.0, 0, -3.0 / 4.0 * np.pi, 0, np.pi / 2.0, 1.0 / 4.0 * np.pi, 0.08]
        )
        self.target_joint_pos = self.curr_joint_pos.copy()
        self.curr_joint_vel = np.zeros(self.n_arm_joints)
        self.target_joint_vel = self.curr_joint_vel.copy()
        self.one_step_time_record = time.time()

        self.node = node
        self.use_high_freq_interp = use_high_freq_interp

        # state subscriber
        self.received_arm_joint_states = False
        self.received_gripper_joint_states = False
        self.arm_joint_states_sub = self.node.create_subscription(
            JointState,
            "/franka/joint_states",
            self._arm_joint_states_callback,
            1,
            callback_group=ReentrantCallbackGroup(),
        )
        self.gripper_type = gripper_type
        if self.gripper_type == "original":
            self.gripper_joint_states_sub = self.node.create_subscription(
                JointState,
                "/fer_gripper/joint_states",
                self._gripper_joint_states_callback,
                1,
                callback_group=ReentrantCallbackGroup(),
            )
        elif self.gripper_type == "robotiq":
            self._gripper_joint_states_sub = self.node.create_subscription(
                JointState,
                "/robotiq_gripper/joint_states",
                self._gripper_joint_states_callback,
                1,
                callback_group=ReentrantCallbackGroup(),
            )

        # command publisher
        if gripper_type == "original":
            if self.gripper_communi_mode == "TOPIC":
                self.gripper_grasp_command_pub = self.node.create_publisher(GripperGrasp, "/fer_gripper/grasp_command", 1)
            elif self.gripper_communi_mode == "ACTION":
                self._gripper_homing_action_client = ActionClient(node, Homing, "/fer_gripper/homing")
                self._gripper_grasp_action_client = ActionClient(node, Grasp, "/fer_gripper/grasp")
                self._gripper_move_action_client = ActionClient(node, Move, "/fer_gripper/move")
            else:
                raise NotImplementedError()
        elif gripper_type == "robotiq":
            self._robotiq_gripper_action_client = ActionClient(node, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")
        else:
            raise NotImplementedError("Gripper type must be original or robotiq.")

        self.ctrl_type = ctrl_type
        if self.ctrl_type == "pos":
            if self.use_high_freq_interp:
                print("Please launch the main_robot_real_high_freq.py node.")
                self.robot_joint_pos_command_pub = self.node.create_publisher(
                    JointState, "robot/joint_pos_command_low_freq", 1
                )
                self.reset_high_freq_node_client = self.node.create_client(Trigger, "robot/reset_high_freq_node")
            else:
                self.arm_joint_pos_command_pub = self.node.create_publisher(
                    Float64MultiArray, "franka/joint_impedance_command", 1
                )
        elif self.ctrl_type == "vel":
            if self.use_high_freq_interp:
                print("Please launch the main_robot_real_vel_high_freq.py node.")
                self.robot_joint_vel_command_pub = self.node.create_publisher(
                    JointState, "robot/joint_vel_command_low_freq", 1
                )
                self.reset_high_freq_node_client = self.node.create_client(Trigger, "robot/reset_high_freq_node")
            else:
                self.arm_joint_vel_command_pub = self.node.create_publisher(
                    Float64MultiArray, "franka/joint_velocity_command", 1
                )

        if self.ctrl_type == "pos":
            self.set_joint_stiffness_client = self.node.create_client(
                SetJointStiffnessDamping, "franka/set_joint_stiffness"
            )

    def wait_for_initialization(self):
        self.node.get_logger().info("Waiting for initialization.")
        time.sleep(1.0)

        self.node.get_logger().info("Waiting for gripper's action server to be available...")
        if self.gripper_type == "original":
            if self.gripper_communi_mode == "ACTION":
                self._gripper_homing_action_client.wait_for_server()
                self._gripper_grasp_action_client.wait_for_server()
                self._gripper_move_action_client.wait_for_server()
        elif self.gripper_type == "robotiq":
            self._robotiq_gripper_action_client.wait_for_server()
        self.node.get_logger().info("Gripper's action server ready.")

        # self.update_joint_pos()
        # while (
        #     self.received_arm_joint_states is False or self.received_gripper_joint_states is False
        # ):  # wait for receiving the arm joint states
        #     self.update_joint_pos()
        #     time.sleep(0.01)

        # joint stiffness reset server
        if self.ctrl_type == "pos":
            self.set_joint_stiffness_client.wait_for_service()
            self.reset_joint_stiffness(stiffness=[300, 300, 300, 300, 125, 75, 25], damping=[15, 15, 15, 15, 5, 5, 2.5])
            self.node.get_logger().info("Reset joint stiffness service ready.")

        if self.use_high_freq_interp:
            self.reset_high_freq_node_client.wait_for_service()
            self._send_reset_high_freq_node_request()
            self.node.get_logger().info("High_freq node ready.")

        self.target_joint_pos = self.curr_joint_pos.copy()
        self.target_joint_vel = self.curr_joint_vel.copy()
        self.node.get_logger().info("--- Robot initialization done ---")

    def _send_reset_high_freq_node_request(self):
        request = Trigger.Request()
        # clear the high_freq_command window
        response = self.reset_high_freq_node_client.call_async(request)

        # send three commands of the current joint pos to re-fill the high_freq_command window
        if self.ctrl_type == "pos":
            for i in range(3):
                self.ctrl_joint_pos(self.get_joint_pos())
                self.step()
        elif self.ctrl_type == "vel":
            for i in range(3):
                self.ctrl_joint_vel(np.zeros(self.n_arm_joints))
                self.step()

    def reset_joint_stiffness(self, stiffness: np.ndarray, damping: np.ndarray):
        assert len(stiffness) == 7
        assert len(damping) == 7

        request = SetJointStiffnessDamping.Request()
        request.joint_stiffness = np.asarray(stiffness, dtype=np.float64).tolist()
        request.joint_damping = np.asarray(damping, dtype=np.float64).tolist()
        response = self.set_joint_stiffness_client.call_async(request)

        self.node.get_logger().info("Sent reset joint stiffness request.")

    def _arm_joint_states_callback(self, msg):
        indices = [msg.name.index(name) for name in self.arm_joint_names]
        self.curr_joint_pos[: self.n_arm_joints] = np.asarray(msg.position)[indices]
        self.curr_joint_vel[: self.n_arm_joints] = np.asarray(msg.velocity)[indices]
        
        self.received_arm_joint_states = True

    def _gripper_joint_states_callback(self, msg):
        if self.gripper_type == "original":
            self.curr_joint_pos[-1] = np.sum(np.asarray(msg.position))
        elif self.gripper_type == "robotiq":
            # self.curr_joint_pos[-1] = (0.8 - msg.position[0]) / 0.8
            self.curr_joint_pos[-1] = msg.position[0] / 0.8 * 0.85
            # self.curr_joint_pos[-1] = msg.position[0]
        self.received_gripper_joint_states = True

    def _publish_arm_joint_pos_command(self, joint_pos):
        assert len(joint_pos) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        self.arm_joint_pos_command_pub.publish(msg)

    def _publish_arm_joint_vel_command(self, joint_vel):
        assert len(joint_vel) == self.n_arm_joints
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_vel]
        self.arm_joint_vel_command_pub.publish(msg)

    def _publish_robot_joint_pos_command(self, joint_pos):
        """
        Publish command to high-frequency interpolator node.
        """
        assert len(joint_pos) == self.n_arm_joints
        msg = JointState()

        # expect the robot to reach the goal waypoint in one timestep
        expected_time = (
            time_to_seconds(self.node.get_clock().now()) + self.timestep
        )  # TODO: is self.timestep appropriate?
        msg.header.stamp = seconds_to_stamp(expected_time)
        msg.position = [q for q in joint_pos]
        self.robot_joint_pos_command_pub.publish(msg)

    def _publish_robot_joint_vel_command(self, joint_vel):
        """
        Publish command to high-frequency interpolator node.
        """
        assert len(joint_vel) == self.n_arm_joints
        msg = JointState()

        # expect the robot to reach the goal waypoint in one timestep
        expected_time = (
            time_to_seconds(self.node.get_clock().now()) + self.timestep
        )  # TODO: is self.timestep appropriate?
        msg.header.stamp = seconds_to_stamp(expected_time)
        msg.velocity = [q for q in joint_vel]
        self.robot_joint_vel_command_pub.publish(msg)

    def step(self, refresh=False):
        time.sleep(0.01)  # at least 10 ms
        while (time.time() - self.one_step_time_record) < self.timestep:
            time.sleep(0.001)
        self.one_step_time_record = time.time()

    def ctrl_joint_pos(self, target_joint_pos):
        """
        Control arm.
        """
        arm_joint_pos = target_joint_pos[: self.n_arm_joints]
        # arm
        if self.use_high_freq_interp:
            self._publish_robot_joint_pos_command(arm_joint_pos)
        else:
            self._publish_arm_joint_pos_command(arm_joint_pos)
        self.target_joint_pos[:] = target_joint_pos

    def ctrl_joint_vel(self, target_joint_vel):
        """
        Control arm.
        """
        arm_joint_vel = target_joint_vel[: self.n_arm_joints]
        # arm
        if self.use_high_freq_interp:
            self._publish_robot_joint_vel_command(arm_joint_vel)
        else:
            self._publish_arm_joint_vel_command(arm_joint_vel)
        self.target_joint_vel[:] = target_joint_vel

    def gripper_homing(self):
        """
        Gripper to home (close and then open). Blocking the current program.
        """
        if self.gripper_type == "original":
            if self.gripper_communi_mode == "TOPIC":
                raise NotImplementedError()
            elif self.gripper_communi_mode == "ACTION":
                goal_msg = Homing.Goal()
                _ = self._gripper_homing_action_client.send_goal(goal_msg)
        else:
            raise NotImplementedError()

    def gripper_move(self, width, speed=0.05):
        """
        Non-blocking.
        Args:
            width: distance between the two fingers.
        """
        if self.gripper_type == "original":
            if self.gripper_communi_mode == "TOPIC":
                raise NotImplementedError()
            elif self.gripper_communi_mode == "ACTION":
                goal_msg = Move.Goal()
                goal_msg.speed = float(speed)
                goal_msg.width = float(width)
                _ = self._gripper_move_action_client.send_goal_async(goal_msg)
        elif self.gripper_type == "robotiq":
            # pos = (1.0 - width) * 0.8
            # pos = width * 0.8
            pos = width
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = float(pos)
            goal_msg.command.max_effort = 40.0
            _ = self._robotiq_gripper_action_client.send_goal_async(goal_msg)
        else:
            raise NotImplementedError()

    def gripper_grasp(self, width, speed=0.1, force=20):
        """
        Non-blocking.
        Args:
            width: distance between the two fingers.
        """
        if self.gripper_type == "original":
            if self.gripper_communi_mode == "TOPIC":
                msg = GripperGrasp()
                msg.width = float(width)
                msg.speed = float(speed)
                msg.force = float(force)
                self.gripper_grasp_command_pub.publish(msg)
            elif self.gripper_communi_mode == "ACTION":
                goal_msg = Grasp.Goal()
                goal_msg.speed = float(speed)
                goal_msg.width = float(width)
                goal_msg.force = float(force)
                goal_msg.epsilon.inner = 0.08
                goal_msg.epsilon.outer = 0.08
                _ = self._gripper_grasp_action_client.send_goal_async(goal_msg)
        elif self.gripper_type == "robotiq":
            # pos = (1.0 - width) * 0.8
            # pos = width * 0.8
            pos = width
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = float(pos)
            goal_msg.command.max_effort = force
            _ = self._robotiq_gripper_action_client.send_goal_async(goal_msg)
        else:
            raise NotImplementedError()

    def update_joint_pos(self):
        """
        Update self.curr_joint_pos
        """
        pass

    def get_joint_pos(self, update=True):
        """
        Args:
            update: if False, it will return the current self.curr_joint_pos,
                    but it does not mean the self.curr_joint_pos is the latest received joint pos.
        """
        if update:
            self.update_joint_pos()
        return self.curr_joint_pos.copy()

    def get_target_joint_pos(self):
        return self.target_joint_pos.copy()

    def get_joint_vel(self, update=True):
        """
        Args:
            update: if False, it will return the current self.curr_joint_pos,
                    but it does not mean the self.curr_joint_pos is the latest received joint pos.
        """
        if update:
            self.update_joint_pos()
        return self.curr_joint_vel.copy()
    
    def get_target_joint_vel(self):
        return self.target_joint_vel.copy()

def main():
    from threading import Thread

    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=None)
    node = Node("node_name")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot_real = RobotReal(node, use_high_freq_interp=True)
    robot_real.wait_for_initialization()

    # stiffness = np.array([30, 30, 30, 30, 12, 7, 2]) / 10000
    # damping = np.array([2, 2, 2, 2, 1, 1, 0.5]) / 10
    # robot_real.reset_joint_stiffness(stiffness=stiffness, damping=damping)

    # curr_joint_pos = robot_real.get_joint_pos()
    # target_joint_pos = curr_joint_pos.copy()

    # target_joint_pos[0] += 0.1
    # robot_real.ctrl_joint_pos(target_joint_pos)
    # print("publish_command time: ", time.time())

    # i = 0
    # while i < 200:
    #     time.sleep(0.001)
    #     print("current time: ", time.time())
    #     joint_pos = robot_real.get_joint_pos()
    #     print("current joint_pos: ", joint_pos)
    #     i += 1

    while True:
        t1 = time.time()
        joint_pos = robot_real.get_joint_pos()

        print("current joint_pos: ", joint_pos)

        robot_real.ctrl_joint_pos(joint_pos)
        robot_real.step()

        # print("current joint_pos: ", joint_pos)
        # print("get_joint_pos() time cost: ", time.time() - t1)
        # time.sleep(0.2)

    # Cleanup on shutdown
    robot_real.node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
