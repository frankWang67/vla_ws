from typing import Optional

import nlopt
import numpy as np
from rclpy.node import Node
from scipy.optimize import minimize

from pi0_ros.robot_adaptor import RobotAdaptor
from pi0_ros.robot_pinocchio import RobotPinocchio
from pi0_ros.robot_real import RobotReal
from pi0_ros.utils.utils_calc import (
    isometry3dToPosOri,
    jacoLeftBCHInverse,
    posOri2Isometry3d,
    sciR,
)


class RobotControl:
    """
    Robot controller connected to the simulated robot or real robot.
    """

    def __init__(
        self,
        robot_model: RobotPinocchio,
        robot_adaptor: RobotAdaptor,
        ctrl_type: str,
        gripper_type: str,
        use_hardware: bool = False,
        use_high_freq_interp: bool = False,
        node: Optional[Node] = None,
    ):
        # --------- hyper-parameters ---------
        # -------------------------------------

        self.robot_model = robot_model
        self.robot_adaptor = robot_adaptor
        self.ctrl_type = ctrl_type
        self.gripper_type = gripper_type
        self.use_hardware = use_hardware
        self.use_ros = self.use_hardware or False

        if self.use_ros:
            self.node = node

        if self.use_hardware:
            self.env = RobotReal(
                self.node,
                self.ctrl_type,
                self.gripper_type,
                use_high_freq_interp=use_high_freq_interp,
            )
            # self.env.wait_for_initialization()
        else:
            raise NotImplementedError()

        self.init_joint_pos = np.zeros((self.robot_adaptor.doa))
        self.init_joint_pos = np.array(
            [0, -np.pi / 4.0, 0, -3.0 / 4.0 * np.pi, 0, np.pi / 2.0, 1.0 / 4.0 * np.pi, 0.08]
        )

    def single_link_ik_nlopt(self, target_link_name, ref_link_pose, pose_weights, qpos_init, qpos_weight=1e-6):
        ref_link_pos, ref_link_ori = isometry3dToPosOri(ref_link_pose)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa = x
            qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)

            # -------------- objective --------------
            self.robot_model.compute_forward_kinematics(qpos_dof)
            link_pose = self.robot_model.get_frame_pose(target_link_name)
            link_pos, link_ori = isometry3dToPosOri(link_pose)
            link_pos_err = link_pos - ref_link_pos
            link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
            err = np.concatenate([link_pos_err, link_ori_err], axis=0)
            err = err.reshape(-1, 1)
            cost_pose = 1.0 / 2.0 * err.T @ pose_weights @ err

            qpos_cost = qpos_weight / 2.0 * ((qpos_doa - qpos_init) ** 2).sum()

            cost = cost_pose[0, 0] + qpos_cost

            # -------------- gradients --------------
            if grad.size > 0:
                self.robot_model.compute_jacobians(qpos_dof)
                jaco = self.robot_adaptor.backward_jacobian(self.robot_model.get_frame_space_jacobian(target_link_name))
                jaco[3:6, :] = np.matmul(jacoLeftBCHInverse(link_ori_err), jaco[3:6, :])
                grad[:] = (err.T @ pose_weights @ jaco).reshape(-1)

                grad[:] += qpos_weight * (qpos_doa - qpos_init).reshape(-1)

            return cost

        opt_dim = self.robot_adaptor.doa
        opt = nlopt.opt(nlopt.LD_SLSQP, opt_dim)
        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits.T).T
        epsilon = 1e-3
        opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())
        opt.set_ftol_abs(1e-8)
        opt.set_min_objective(objective)

        qpos_doa_res = opt.optimize(qpos_init)

        # # --- check the error ---
        # qpos_doa = qpos_doa_res.copy()
        # qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        # self.robot_model.compute_forward_kinematics(qpos_dof)
        # link_pose = self.robot_model.get_frame_pose(target_link_name)
        # link_pos, link_ori = isometry3dToPosOri(link_pose)
        # link_pos_err = link_pos - ref_link_pos
        # link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
        # err = np.concatenate([link_pos_err, link_ori_err], axis=0)
        # print("ik err: ", err)

        return qpos_doa_res

    def single_link_ik_scipy(self, target_link_name, ref_link_pose, weights, qpos_init):
        ref_link_pos, ref_link_ori = isometry3dToPosOri(ref_link_pose)

        def objective(x: np.ndarray):
            qpos_doa = x
            qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)

            # -------------- objective --------------
            self.robot_model.compute_forward_kinematics(qpos_dof)
            link_pose = self.robot_model.get_frame_pose(target_link_name)
            link_pos, link_ori = isometry3dToPosOri(link_pose)
            link_pos_err = link_pos - ref_link_pos
            link_ori_err = (link_ori * ref_link_ori.inv()).as_rotvec()
            err = np.concatenate([link_pos_err, link_ori_err], axis=0)
            err = err.reshape(-1, 1)

            cost_pose = 1.0 / 2.0 * err.T @ weights @ err
            cost = cost_pose[0, 0]

            # -------------- gradients --------------
            self.robot_model.compute_jacobians(qpos_dof)
            jaco = self.robot_adaptor.backward_jacobian(self.robot_model.get_frame_space_jacobian(target_link_name))
            jaco[3:6, :] = np.matmul(jacoLeftBCHInverse(link_ori_err), jaco[3:6, :])

            grad = (err.T @ weights @ jaco).reshape(-1)

            return cost, grad

        joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits.T).T
        joint_pos_bounds = [(joint_limits[i, 0], joint_limits[i, 1]) for i in range(joint_limits.shape[0])]

        res = minimize(
            fun=objective,
            jac=True,
            x0=qpos_init,
            bounds=joint_pos_bounds,
            method="SLSQP",
            options={"ftol": 1e-8, "disp": False},
        )

        qpos_doa_res = res.x.reshape(-1)
        return qpos_doa_res

    def move_to_joint_pos(self, target_joint_pos, max_joint_speed=[[0.2] * 7 + [1e5] * 1]):
        """
        Move to the target joint position in a controllable speed.
        The function is blocked until reaching the target.
        """
        # curr_joint_pos = self.env.get_joint_pos()
        curr_joint_pos = self.env.get_target_joint_pos()
        delta_joint_pos = target_joint_pos - curr_joint_pos
        delta_joint_pos_max = np.asarray(max_joint_speed) * self.env.timestep

        t_max = np.abs(delta_joint_pos_max / (delta_joint_pos + 1e-8))
        t_interp = np.min([np.min(t_max), 1.0])
        n_step = int(1.0 / t_interp)

        print(f"move_to_joint_pos(): n_step={n_step}")

        for i in range(1, n_step + 1):
            t = float(i) / float(n_step)
            joint_pos = (1 - t) * curr_joint_pos + t * target_joint_pos
            self.env.ctrl_joint_pos(joint_pos)
            if i < n_step:  # not call step() at the last step
                self.env.step()

    def cartesian_move(self, tcp_link_name: str, tcp_motion: np.ndarray):
        """
        Args:
            tcp_motion[:3]: translation defined in the world frame.
            tcp_motion[3:]: rotation defined in the tcp body frame.
        Return:
            target_joint_pos: joint pos command by IK.
        """
        assert len(tcp_motion) == 6
        target_link_name = tcp_link_name
        qpos_doa = self.env.get_target_joint_pos()
        qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        last_target_pose = self.robot_model.get_frame_pose(target_link_name, qpos_dof)
        curr_pos, curr_ori = isometry3dToPosOri(last_target_pose)

        target_pos = curr_pos + tcp_motion[:3]
        target_ori = curr_ori * sciR.from_rotvec(tcp_motion[3:])
        target_pose = posOri2Isometry3d(target_pos, target_ori)

        weights = np.diag([100, 100, 100, 10, 10, 10])
        qpos_doa_res = self.single_link_ik_nlopt(target_link_name, target_pose, weights, qpos_init=qpos_doa)
        self.env.ctrl_joint_pos(qpos_doa_res)
        return qpos_doa_res

    def constrained_free_move(self, tcp_link_name: str, tcp_pose: np.ndarray):
        """
        Just for test.
        """
        tcp_weights = [10, 10, 10, 1, 1, 1]

        target_link_name = tcp_link_name
        qpos_doa = self.env.get_joint_pos()
        qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        curr_pose = self.robot_model.get_frame_pose(target_link_name, qpos_dof)
        curr_pos, _ = isometry3dToPosOri(curr_pose)

        target_pos = curr_pos
        target_ori = sciR.from_matrix(tcp_pose[:3, :3])
        target_pose = posOri2Isometry3d(target_pos, target_ori)

        weights = np.diag(tcp_weights)
        qpos_doa_res = self.single_link_ik_nlopt(
            target_link_name, target_pose, weights, qpos_init=qpos_doa, qpos_weight=1e-2
        )
        self.env.ctrl_joint_pos(qpos_doa_res)
        return qpos_doa_res

    def keys_to_tcp_motion(self, pressed_keys, trans_speed=0.001, rot_speed=0.01):
        tcp_motion = np.zeros((6))
        if "d" in pressed_keys:
            tcp_motion[0] = -trans_speed
        if "a" in pressed_keys:
            tcp_motion[0] = +trans_speed
        if "w" in pressed_keys:
            tcp_motion[1] = -trans_speed
        if "s" in pressed_keys:
            tcp_motion[1] = +trans_speed
        if "q" in pressed_keys:
            tcp_motion[2] = -trans_speed
        if "e" in pressed_keys:
            tcp_motion[2] = +trans_speed
        return tcp_motion

    def step(self):
        self.env.step()

    def ctrl_joint_pos(self, target_joint_pos):
        self.env.ctrl_joint_pos(target_joint_pos)

    def ctrl_joint_vel(self, target_joint_vel):
        self.env.ctrl_joint_vel(target_joint_vel)

    def ctrl_tcp_pos(self, target_tcp_pos, rotation_type="quat", fix_joint=False):
        # qpos_doa = self.env.get_target_joint_pos()
        qpos_doa = self.get_joint_pos()
        target_pos = target_tcp_pos[:3]
        target_ori = sciR.from_quat(target_tcp_pos[3:7])
        target_pose = posOri2Isometry3d(target_pos, target_ori)

        # # TEMP: for 'pose' baseline comparison in Task collision avoidance.
        if fix_joint:
            print(qpos_doa)
            qpos_doa[1] += 0.005
            qpos_doa[1] = max(0.0, qpos_doa[1])

        weights = np.diag([100, 100, 100, 10, 10, 10])
        qpos_weight = 1e-1
        qpos_doa_res = self.single_link_ik_nlopt(
            "panda_hand_tcp", target_pose, weights, qpos_init=qpos_doa, qpos_weight=qpos_weight
        )
        self.env.ctrl_joint_pos(qpos_doa_res)

        # print("solved joint pos: ", qpos_doa_res)
        return qpos_doa_res

    def get_joint_pos(self):
        return self.env.get_joint_pos()
    
    def get_joint_vel(self):
        return self.env.get_joint_vel()

    def get_tcp_pose(self, rotation_type="quat"):
        qpos_doa = self.env.get_joint_pos()
        qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        last_target_pose = self.robot_model.get_frame_pose("panda_hand_tcp", qpos_dof)
        curr_pos, curr_ori = isometry3dToPosOri(last_target_pose)

        if rotation_type == "rotation":
            return curr_pos, curr_ori
        elif rotation_type == "quat":
            curr_ori = curr_ori.as_quat()
        elif rotation_type == "euler":
            curr_ori = curr_ori.as_euler()
        curr_pos = np.concatenate((curr_pos, curr_ori)).reshape(
            -1,
        )
        return curr_pos

    def get_tcp_pose_from_joint_pos(self, joint_pos, rotation_type="original", frame_name="panda_hand_tcp"):
        if joint_pos.shape[0] == 7:
            joint_pos = np.concatenate((joint_pos, [0.04, 0.04])).reshape(-1)

        target_tcp_pose = self.robot_model.get_frame_pose(frame_name, joint_pos)
        curr_pos, curr_ori = isometry3dToPosOri(target_tcp_pose)

        if rotation_type == "rotation":
            return curr_pos, curr_ori
        elif rotation_type == "quat":
            curr_ori = curr_ori.as_quat()
        elif rotation_type == "euler":
            curr_ori = curr_ori.as_euler()
        elif rotation_type == "original":
            return target_tcp_pose

        return np.concatenate((curr_pos, curr_ori)).reshape(
            -1,
        )

    def gripper_open_or_close(self, force=20):
        if self.get_joint_pos()[-1] > 0.04:
            self.env.gripper_grasp(width=0.0, force=force)
        else:
            self.env.gripper_grasp(width=1.0, force=force)

    def gripper_grasp(self, width, speed=0.1, force=20):
        self.env.gripper_grasp(width, speed, force)

    def gripper_move(self, width, speed=0.1):
        self.env.gripper_move(width, speed)

if __name__ == "__main__":
    pass
