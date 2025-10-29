import os
import sys
import cv2
import time
import tyro
import collections
import dataclasses
import numpy as np
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from pi0_ros.robot_adaptor import RobotAdaptor
from pi0_ros.robot_control import RobotControl
from pi0_ros.robot_pinocchio import RobotPinocchio

from pi0_ros.utils.utils_calc import _quat2axisangle, _axisangle2quat

EXT_CAM_TOPIC = "/camera/exterior_cam/color/image_raw"
# WST_CAM_TOPIC = "/camera/wrist_cam/color/image_raw"
WST_CAM_TOPIC = "/wrist_cam/zed_node/left/image_rect_color"
# WST_CAM_TOPIC = "/wrist_cam/zed_node/rgb/image_rect_color"
EEF_POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"

POLICIES = ["libero", "droid"]
CONTROL_TYPES = {
    "libero": "pos",
    # "droid": "vel",
    "droid": "pos",
}
GRIPPER_TYPES = {
    "libero": "original",
    "droid": "robotiq",
}

MAX_VEL = 0.5

@dataclasses.dataclass
class Args:
    prompt: str
    policy: str = "droid"
    max_hz: int = 10
    # host: str = "0.0.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    seed: int = 7  # Random Seed (for reproducibility)

class VlaExpMain:
    def __init__(self, args: Args):
        assert args.policy in POLICIES

        self.prompt = args.prompt
        self.policy = args.policy
        self.max_hz = args.max_hz
        self.time_interval = 1.0 / self.max_hz
        # self.time_last = time.time()
        self.resize_size = args.resize_size
        self.replan_steps = args.replan_steps

        np.random.seed(args.seed)
        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        # --------- hyper-parameters ---------
        desc_dir = get_package_share_directory("franka_description")
        urdf_file_name = os.path.join(desc_dir, "robots", "panda_arm.urdf")
        actuated_joints_name = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint"]
        mapping_matrix_from_doa_to_dof = np.block(
            [[np.eye(7), np.zeros((7, 1))], [np.zeros((2, 7)), 0.5 * np.ones((2, 1))]]
        )

        self.use_hardware = True
        self.use_high_freq_interp = True
        # -------------------------------------

        self.node = Node("main_node")
        # self.executor = MultiThreadedExecutor()
        # self.executor.add_node(self.node)
        # self.spin_thread = Thread(target=self.executor.spin, daemon=False)
        # self.spin_thread.start()

        self.need_initialize = True

        self.robot_model = RobotPinocchio(
            robot_file_path=urdf_file_name,
            robot_file_type="urdf",
        )
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            actuated_joints_name=actuated_joints_name,
            mapping_matrix_from_doa_to_dof=mapping_matrix_from_doa_to_dof,
        )
        self.robot_control = RobotControl(
            robot_model=self.robot_model,
            robot_adaptor=self.robot_adaptor,
            ctrl_type=CONTROL_TYPES[self.policy],
            gripper_type=GRIPPER_TYPES[self.policy],
            use_hardware=self.use_hardware,
            use_high_freq_interp=self.use_high_freq_interp,
            node=self.node,
        )

        self.gripper_open = self.robot_control.env.curr_joint_pos[-1] < 0.4

        # self.ext_cam_sub = Subscriber(self.node, RosImage, EXT_CAM_TOPIC)
        # self.wst_cam_sub = Subscriber(self.node, RosImage, WST_CAM_TOPIC)
        # self.eef_pose_sub = Subscriber(self.node, PoseStamped, EEF_POSE_TOPIC)
        # self.joint_state_sub = Subscriber(self.node, JointState, JOINT_STATE_TOPIC)

        self.ext_img = None
        self.wst_img = None
        if self.policy == "libero":
            self.eef_pose_msg = None
        # self.obs = None

        self.bridge = CvBridge()

        self.stay_still = False
        self.stay_still_cnt = 0
        self.stay_still_cnt_max = 10

        # self.sync = ApproximateTimeSynchronizer(
        #     [self.ext_cam_sub, self.wst_cam_sub, self.eef_pose_sub, self.joint_state_sub],
        #     queue_size=10,
        #     slop=0.01
        # )

        # # Register the callback function for synchronized messages
        # self.sync.registerCallback(self.listener_callback)

        self.node.create_subscription(RosImage, EXT_CAM_TOPIC, self.ext_cam_callback, qos_profile_sensor_data)
        self.node.create_subscription(RosImage, WST_CAM_TOPIC, self.wst_cam_callback, qos_profile_sensor_data)
        if self.policy == "libero":
            self.node.create_subscription(PoseStamped, EEF_POSE_TOPIC, self.eef_pose_callback, qos_profile_system_default)

        self.action_plan = collections.deque()
        self.node.create_timer(self.time_interval, self.main_loop)

    def ext_cam_callback(self, ext_cam_msg: RosImage):
        ext_img = self.bridge.imgmsg_to_cv2(ext_cam_msg, "bgr8")
        self.ext_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(ext_img, self.resize_size, self.resize_size)
        )
        # cv2.imshow("ext_raw", ext_img)
        cv2.imshow("ext", self.ext_img)
        cv2.waitKey(1)
    
    def wst_cam_callback(self, wst_cam_msg: RosImage):
        wst_img = self.bridge.imgmsg_to_cv2(wst_cam_msg, "bgr8")
        self.wst_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wst_img, self.resize_size, self.resize_size)
        )
        # cv2.imshow("wst_raw", wst_img)
        cv2.imshow("wst", self.wst_img)
        cv2.waitKey(1)
    
    def eef_pose_callback(self, eef_pose_msg: PoseStamped):
        self.eef_pose_msg = eef_pose_msg
        
    # def listener_callback(self, ext_cam_msg: RosImage, wst_cam_msg: RosImage, eef_pose_msg: PoseStamped, joint_state_msg: JointState):
    #     print("########################################################")
    #     print("########################################################")
    #     print("########################################################")
    #     print("########################################################")
    #     print("########################################################")
    #     self.ext_img = self.bridge.imgmsg_to_cv2(ext_cam_msg, "rgb8")
    #     self.wst_img = self.bridge.imgmsg_to_cv2(wst_cam_msg, "rgb8")
    #     self.ext_img = image_tools.convert_to_uint8(
    #         image_tools.resize_with_pad(self.ext_img, self.resize_size, self.resize_size)
    #     )
    #     self.wst_img = image_tools.convert_to_uint8(
    #         image_tools.resize_with_pad(self.wst_img, self.resize_size, self.resize_size)
    #     )
    #     cv2.imshow("ext", self.ext_img)
    #     cv2.imshow("wst", self.wst_img)
    #     cv2.waitKey(1)

    #     if self.policy == "libero":
    #         self.eef_state = np.concatenate(
    #             (
    #                 np.array([eef_pose_msg.pose.position.x, eef_pose_msg.pose.position.y, eef_pose_msg.pose.position.z]),
    #                 _quat2axisangle(np.array([eef_pose_msg.pose.orientation.x, eef_pose_msg.pose.orientation.y, eef_pose_msg.pose.orientation.z, eef_pose_msg.pose.orientation.w])),
    #                 np.array([joint_state_msg.position[7], joint_state_msg.position[8]]),
    #             )
    #         )

    #         self.obs = {
    #             "observation/image": self.ext_img,
    #             "observation/wrist_image": self.wst_img,
    #             "observation/state": self.eef_state,
    #             "prompt": self.prompt,
    #         }
    #     elif self.policy == "droid":
    #         self.joint_position = np.array(joint_state_msg.position[:7])
    #         self.gripper_position = np.array([joint_state_msg.position[7]])
    #         self.obs = {
    #             "observation/exterior_image_1_left": self.ext_img,
    #             "observation/wrist_image_left": self.wst_img,
    #             "observation/joint_position": self.joint_position,
    #             "observation/gripper_position": self.gripper_position,
    #             "prompt": self.prompt,
    #         }

    def step(self, action):
        if self.policy == "libero":
            target_tcp_pose = np.concatenate(
                (action[:3], _axisangle2quat(action[3:6]))
            )
            gripper_width = action[6]

            self.robot_control.ctrl_tcp_pos(target_tcp_pose)
            self.robot_control.gripper_move(gripper_width)
        elif self.policy == "droid":
            target_joint_vel = action[:7]
            vel_val = np.linalg.norm(target_joint_vel)
            if vel_val > MAX_VEL:
                target_joint_vel = target_joint_vel / vel_val * MAX_VEL
            # self.robot_control.ctrl_joint_vel(target_joint_vel)

            current_joint_pos = self.robot_control.get_joint_pos()
            target_joint_vel = action[:8]
            target_joint_pos = current_joint_pos[:8] + target_joint_vel * self.time_interval
            self.robot_control.ctrl_joint_pos(target_joint_pos)

            target_gripper_pos = action[-1] / 0.85 * 0.8
            if target_gripper_pos < 0.01 or target_gripper_pos > 0.79 or np.abs(target_gripper_pos - current_joint_pos[-1]) > 0.05:
                self.robot_control.gripper_move(target_gripper_pos)

    def main_loop(self):
        if self.need_initialize:
            self.robot_control.env.wait_for_initialization()
            self.need_initialize = False

        if self.policy == "libero":
            if self.ext_img is None or self.wst_img is None or self.eef_pose_msg is None:
                return
            
            self.eef_state = np.concatenate(
                (
                    np.array([self.eef_pose_msg.pose.position.x, self.eef_pose_msg.pose.position.y, self.eef_pose_msg.pose.position.z]),
                    _quat2axisangle(np.array([self.eef_pose_msg.pose.orientation.x, self.eef_pose_msg.pose.orientation.y, self.eef_pose_msg.pose.orientation.z, self.eef_pose_msg.pose.orientation.w])),
                    np.array([self.joint_state_msg.position[7], self.joint_state_msg.position[8]]),
                )
            )

            self.obs = {
                "observation/image": self.ext_img,
                "observation/wrist_image": self.wst_img,
                "observation/state": self.eef_state,
                "prompt": self.prompt,
            }
        elif self.policy == "droid":
            if self.ext_img is None or self.wst_img is None:
                return
            joint_pos = self.robot_control.env.get_joint_pos()
            self.joint_position = joint_pos[:self.robot_control.env.n_arm_joints]
            self.gripper_position = joint_pos[-1]
            self.obs = {
                "observation/exterior_image_1_left": self.ext_img,
                "observation/wrist_image_left": self.wst_img,
                "observation/joint_position": self.joint_position,
                "observation/gripper_position": self.gripper_position,
                "prompt": self.prompt,
            }

        if self.stay_still:
            self.stay_still_cnt += 1
            if self.stay_still_cnt > self.stay_still_cnt_max:
                self.stay_still = False
            if self.policy == "libero":
                action = self.obs["observation/state"][:7]
            elif self.policy == "droid":
                action = np.zeros(8)
        else:
            if not self.action_plan:
                action_chunk = self.client.infer(self.obs)["actions"]
                assert (
                    len(action_chunk) >= self.replan_steps
                ), f"We want to replan every {self.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                self.action_plan.extend(action_chunk[:self.replan_steps])
            action = self.action_plan.popleft()

        # print("------------------------------------")
        # if self.policy == "libero":
        #     print(f"{self.eef_state=}")
        # elif self.policy == "droid":
        #     print(f"{self.joint_position=}")
        # print(f"{action=}")
        self.step(action)
        
def main():
    argv = sys.argv[1:]  # 去掉程序名
    ros_args_index = argv.index("--ros-args") if "--ros-args" in argv else len(argv)
    user_args = tyro.cli(Args, args=argv[:ros_args_index])

    rclpy.init(args=argv[ros_args_index:])

    try:
        exp = VlaExpMain(args=user_args)
        rclpy.spin(exp.node)
    except KeyboardInterrupt:
        exp.node.destroy_node()
        rclpy.shutdown()
    except Exception:
        import traceback
        traceback.print_exc()

    cv2.destroyAllWindows()
