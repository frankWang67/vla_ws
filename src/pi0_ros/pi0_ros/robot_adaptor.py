from typing import List

import numpy as np


class RobotAdaptor:
    def __init__(
        self,
        robot_model,
        actuated_joints_name: List[str],
        mapping_matrix_from_doa_to_dof: np.ndarray,
    ):
        self.robot_model = robot_model
        self.actuated_joints_name = actuated_joints_name
        self.mapping_matrix_from_doa_to_dof = mapping_matrix_from_doa_to_dof
        self.mapping_matrix_from_dof_to_doa = np.linalg.pinv(mapping_matrix_from_doa_to_dof)
        if mapping_matrix_from_doa_to_dof.shape[0] != self.robot_model.dof:
            raise ValueError("The shape of mapping_matrix_from_doa_to_dof does not match the robot DoF!")
        if mapping_matrix_from_doa_to_dof.shape[1] != len(actuated_joints_name):
            raise ValueError("The shape of mapping_matrix_from_doa_to_dof does not match the robot DoA!")

    @property
    def doa(self) -> int:
        return len(self.actuated_joints_name)

    def check_doa(self, q: np.ndarray):
        assert q.shape[-1] == self.doa

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        qpos_doa to qpos_dof. Support batch operation.
        Args:
            qpos: position of the actuated joints
        Return:
            qpos_f: position of all dof joints
        """
        self.check_doa(qpos)
        mapping_matrix_from_doa_to_dof = np.expand_dims(self.mapping_matrix_from_doa_to_dof, axis=0)
        qpos_dof = np.matmul(mapping_matrix_from_doa_to_dof, qpos.reshape(-1, qpos.shape[-1], 1))
        return qpos_dof.squeeze()

    def backward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        qpos_dof to qpos_doa. Support batch operation.
        """
        self.robot_model.check_joint_dim(qpos)
        mapping_matrix_from_dof_to_doa = np.expand_dims(self.mapping_matrix_from_dof_to_doa, axis=0)
        qpos_doa = np.matmul(mapping_matrix_from_dof_to_doa, qpos.reshape(-1, qpos.shape[-1], 1))
        return qpos_doa.squeeze()

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Args:
            jacobian: shape (n_batch, 6, n_dof) computed by self.robot_model
        Return:
            jacobian: shape (n_batch, 6, n_doa)
        """
        jacobian = jacobian.reshape(-1, jacobian.shape[-2], jacobian.shape[-1])
        m_shape = self.mapping_matrix_from_doa_to_dof.shape
        jacobian_doa = np.matmul(
            jacobian,
            self.mapping_matrix_from_doa_to_dof.reshape(1, m_shape[0], m_shape[1]),
        )
        return jacobian_doa.squeeze()


if __name__ == "__main__":
    from robot_pinocchio import RobotPinocchio

    robot_file_path = "ws_ros2/src/my_franka_description/urdf/panda_gripper.urdf"
    actuated_joints_name = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint"]
    mapping_matrix_from_doa_to_dof = np.block(
        [[np.eye(7), np.zeros((7, 1))], [np.zeros((2, 7)), 0.5 * np.ones((2, 1))]]
    )

    robot_model = RobotPinocchio(
        robot_file_path=robot_file_path,
        robot_file_type="urdf",
    )

    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        actuated_joints_name=actuated_joints_name,
        mapping_matrix_from_doa_to_dof=mapping_matrix_from_doa_to_dof,
    )

    print("DoF joint names: ", robot_model.joint_names)
    print("DoA joint names: ", robot_adaptor.actuated_joints_name)

    doa = robot_adaptor.doa
    dof = robot_model.dof

    qpos_dof = robot_adaptor.forward_qpos(np.zeros((doa)))
    print(qpos_dof.shape)

    jaco_doa = robot_adaptor.backward_jacobian(np.zeros((6, dof)))
    print(jaco_doa.shape)
