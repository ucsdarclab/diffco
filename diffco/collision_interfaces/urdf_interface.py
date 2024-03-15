# Description: This file contains the interfaces for different robots and environments
# They either read robot info and provides fkine function, or read obstacle info.
# Consider add parent classes for robot and environment interfaces, respectively.
"""
Differentiable robot model class
====================================
TODO
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

import torch

from yourdfpy import URDF
from trimesh import transformations as tf

import robot_data

from .rigid_body import RigidBody

robot_description_folder = robot_data.__path__[0]


def tensor_check(function):
    """
    A decorator for checking the device of input tensors
    """

    @dataclass
    class BatchInfo:
        shape: torch.Size = torch.Size([])
        init: bool = False

    def preprocess(arg, obj, batch_info):
        if type(arg) is torch.Tensor:
            # Check device
            assert (
                arg.device.type == obj._device.type
            ), f"Input argument of different device as module: {arg}"

            # Check dimensions & convert to 2-dim tensors
            assert arg.ndim in [1, 2], f"Input tensors must have ndim of 1 or 2."

            if batch_info.init:
                assert (
                    batch_info.shape == arg.shape[:-1]
                ), "Batch size mismatch between input tensors."
            else:
                batch_info.init = True
                batch_info.shape = arg.shape[:-1]

            if len(batch_info.shape) == 0:
                return arg.unsqueeze(0)

        return arg

    def postprocess(arg, batch_info):
        if type(arg) is torch.Tensor and batch_info.init and len(batch_info.shape) == 0:
            return arg[0, ...]

        return arg

    def wrapper(self, *args, **kwargs):
        batch_info = BatchInfo()

        # Parse input
        processed_args = [preprocess(arg, self, batch_info) for arg in args]
        processed_kwargs = {
            key: preprocess(kwargs[key], self, batch_info) for key in kwargs
        }

        # Perform function
        ret = function(self, *processed_args, **processed_kwargs)

        # Parse output
        if type(ret) is torch.Tensor:
            return postprocess(ret, batch_info)
        elif type(ret) is tuple:
            return tuple([postprocess(r, batch_info) for r in ret])
        else:
            return ret

    return wrapper


class URDFRobot:
    def __init__(self, urdf_path, name='', device="cpu"):
        self.robot = URDF.load(
            urdf_path, 
            build_scene_graph=True,
            build_collision_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=True)
        self.name = name
        self._device = torch.device(device)

        self._n_dofs = 0
        self._controlled_joints = []
        self._bodies = []

        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self.robot.link_map.values()):
            # Initialize body object
            rigid_body_params = self.get_body_parameters_from_urdf(i, link)
            body = RigidBody(
                rigid_body_params=rigid_body_params, device=self._device
            )

            # Joint properties
            body.joint_idx = None
            if rigid_body_params["joint_type"] != "fixed":
                body.joint_idx = self._n_dofs
                self._n_dofs += 1
                self._controlled_joints.append(i)

            # Add to data structures
            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i

        # Once all bodies are loaded, connect each body to its parent
        for body in self._bodies[1:]:
            parent_body_name = self.get_name_of_parent_body(body.name)
            parent_body_idx = self._name_to_idx_map[parent_body_name]
            body.set_parent(self._bodies[parent_body_idx])
            self._bodies[parent_body_idx].add_child(body)
    
    @tensor_check
    def compute_forward_kinematics_all_links(
        self, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        # Create joint state dictionary
        q_dict = {}
        for i, body_idx in enumerate(self._controlled_joints):
            q_dict[self._bodies[body_idx].name] = q[:, i].unsqueeze(1)

        # Call forward kinematics on root node
        pose_dict = self._bodies[0].forward_kinematics(q_dict)

        return {
            link: (pose_dict[link].translation(), pose_dict[link].rotation())
            for link in pose_dict.keys()
        }

    def find_joint_of_body(self, body_name):
        for jname, joint in self.robot.joint_map.items():
            if joint.child == body_name:
                return jname
        return None

    def get_name_of_parent_body(self, link_name):
        jid = self.find_joint_of_body(link_name)
        joint = self.robot.joint_map[jid]
        return joint.parent

    def get_body_parameters_from_urdf(self, i, link):
        body_params = {}
        body_params["joint_id"] = i
        body_params["link_name"] = link.name

        if i == 0:
            rot_angles = torch.zeros(3, device=self._device)
            trans = torch.zeros(3, device=self._device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self._device)
        else:
            link_name = link.name
            joint_name = self.find_joint_of_body(link_name)
            joint = self.robot.joint_map[joint_name]
            # joint_name = joint.name
            # find joint whose "child" of this body according to urdf

            rot_angles = torch.tensor(
                tf.euler_from_matrix(joint.origin[:3, :3], axes='sxyz'), 
                dtype=torch.float32, device=self._device
            )
            trans = torch.tensor(
                joint.origin[:3, 3], dtype=torch.float32, device=self._device
            )
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, device=self._device)
            joint_axis = torch.zeros((1, 3), device=self._device)
            if joint_type != "fixed":
                joint_limits = {
                    "effort": joint.limit.effort,
                    "lower": joint.limit.lower,
                    "upper": joint.limit.upper,
                    "velocity": joint.limit.velocity,
                }
                try:
                    joint_damping = torch.tensor(
                        [joint.dynamics.damping],
                        dtype=torch.float32,
                        device=self._device,
                    )
                except AttributeError:
                    joint_damping = torch.zeros(1, device=self._device)
                joint_axis = torch.tensor(
                    joint.axis, dtype=torch.float32, device=self._device
                ).reshape(1, 3)

        body_params["rot_angles"] = rot_angles
        body_params["trans"] = trans
        body_params["joint_name"] = joint_name
        body_params["joint_type"] = joint_type
        body_params["joint_limits"] = joint_limits
        body_params["joint_damping"] = joint_damping
        body_params["joint_axis"] = joint_axis

        if link.inertial is not None:
            mass = torch.tensor(
                [link.inertial.mass], dtype=torch.float32, device=self._device
            )
            com = (
                torch.tensor(
                    link.inertial.origin[:3, 3],
                    dtype=torch.float32,
                    device=self._device,
                )
                .reshape((1, 3))
                .to(self._device)
            )

            # inert_mat = torch.zeros((3, 3), device=self._device)
            # inert_mat[0, 0] = link.inertial.inertia.ixx
            # inert_mat[0, 1] = link.inertial.inertia.ixy
            # inert_mat[0, 2] = link.inertial.inertia.ixz
            # inert_mat[1, 0] = link.inertial.inertia.ixy
            # inert_mat[1, 1] = link.inertial.inertia.iyy
            # inert_mat[1, 2] = link.inertial.inertia.iyz
            # inert_mat[2, 0] = link.inertial.inertia.ixz
            # inert_mat[2, 1] = link.inertial.inertia.iyz
            # inert_mat[2, 2] = link.inertial.inertia.izz
            inert_mat = torch.tensor(
                link.inertial.inertia, 
                dtype=torch.float32,
                device=self._device
            ).reshape(3, 3)

            inert_mat = inert_mat.unsqueeze(0)
            body_params["mass"] = mass
            body_params["com"] = com
            body_params["inertia_mat"] = inert_mat
        else:
            body_params["mass"] = torch.ones((1,), device=self._device)
            body_params["com"] = torch.zeros((1, 3), device=self._device)
            body_params["inertia_mat"] = torch.eye(3, 3, device=self._device).unsqueeze(
                0
            )
            print(
                "Warning: No dynamics information for link: {}, setting all inertial properties to 1.".format(
                    link.name
                )
            )

        return body_params



class KUKAiiwa(URDFRobot):
    def __init__(self, device=None):
        rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.urdf_path, self.name, device=device)


class FrankaPanda(URDFRobot):
    def __init__(self, device=None):
        rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "differentiable_franka_panda"
        super().__init__(self.urdf_path, self.name, device=device)


class TwoLinkRobot(URDFRobot):
    def __init__(self, device=None):
        rel_urdf_path = "2link_robot.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "diff_2d_robot"
        super().__init__(self.urdf_path, self.name, device=device)


class TrifingerEdu(URDFRobot):
    def __init__(self, device=None):
        rel_urdf_path = "trifinger_edu_description/trifinger_edu.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "trifinger_edu"
        super().__init__(self.urdf_path, self.name, device=device)
    

class URDFEnv:
    def __init__(self, urdf_path):
        raise NotImplementedError
    