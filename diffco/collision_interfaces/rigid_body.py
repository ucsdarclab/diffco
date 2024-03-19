"""
Rigid body
====================================
TODO
"""

from typing import List, Optional
import json

import torch

from .spatial_vector_algebra import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)

class RigidBody:
    """
    Node representation of a link.
    Implements recursive forward kinematics.
    """

    _parent: Optional["RigidBody"]
    _children: List["RigidBody"]

    def __init__(self, rigid_body_params, device="cpu"):

        super().__init__()

        self._parent = None
        self._children = []

        self._device = torch.device(device)
        self.link_idx = rigid_body_params["link_idx"]
        self.name = rigid_body_params["link_name"]
        self.joint_name = rigid_body_params["joint_name"]

        self.join_trans = lambda: rigid_body_params["joint_trans"].reshape(1, 3)
        self.joint_rot_angles = lambda: rigid_body_params["joint_rot_angles"].reshape(1, 3)

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]

        self.joint_limits = rigid_body_params["joint_limits"]

        self.joint_pose = CoordinateTransform(device=self._device)
        self.joint_pose.set_translation(torch.reshape(self.join_trans(), (1, 3)))

        self.update_joint_state(
            torch.zeros([1, 1], device=self._device),
        )

        self.collision_origins = rigid_body_params.get("collision_origins", None)
        if self.collision_origins:
            self.collision_origins = [CoordinateTransform(
                rot=pose[:3, :3].reshape(1, 3, 3),
                trans=pose[:3, 3].reshape(1, 3),
                device=self._device,
            ) for pose in self.collision_origins]
        self.visual_origins = rigid_body_params.get("visual_origins", None)
        if self.visual_origins:
            self.visual_origins = [CoordinateTransform(
                rot=pose[:3, :3].reshape(1, 3, 3),
                trans=pose[:3, 3].reshape(1, 3),
                device=self._device,
            ) for pose in self.visual_origins]

        self.pose = CoordinateTransform(device=self._device)

    # Kinematic tree construction
    def set_parent(self, link: "RigidBody"):
        self._parent = link

    def add_child(self, link: "RigidBody"):
        self._children.append(link)

    # Recursive algorithms
    def forward_kinematics(self, q_dict, return_collision=False):
        """Recursive forward kinematics
        Computes transformations from self to all descendants.

        Returns: Dict[link_name, transform_from_self_to_link]
        """
        # Compute joint pose
        if self.name in q_dict:
            q = q_dict[self.name]
            batch_size = q.shape[0]

            rot_angles_vals = self.joint_rot_angles()
            roll = rot_angles_vals[0, 0]
            pitch = rot_angles_vals[0, 1]
            yaw = rot_angles_vals[0, 2]
            fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

            if torch.abs(self.joint_axis[0, 0]) == 1:
                rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q)
            elif torch.abs(self.joint_axis[0, 1]) == 1:
                rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q)
            else:
                rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q)

            joint_pose = CoordinateTransform(
                rot=fixed_rotation.repeat(batch_size, 1, 1) @ rot,
                trans=torch.reshape(self.join_trans(), (1, 3)).repeat(batch_size, 1),
                device=self._device,
            )

        else:
            joint_pose = self.joint_pose

        # Compute forward kinematics of children
        if return_collision:
            pose_dict = {self.name: [self.pose.multiply_transform(origin) for origin in self.collision_origins]}
        else:
            pose_dict = {self.name: [self.pose]}
        for child in self._children:
            pose_dict.update(child.forward_kinematics(q_dict, return_collision))

        # Apply joint pose
        # TODO: add center of mass to pose if getting body transforms (vs link transforms)
        return {
            body_name: [joint_pose.multiply_transform(p) for p in pose_dict[body_name]]
            for body_name in pose_dict
        }

    def body_transforms(self, q_dict):
        """Compute body transforms
        Computes transformations from self to all descendants.

        Returns: Dict[link_name, transform_from_self_to_link]
        """
        # Compute 

    # Get/set
    def update_joint_state(self, q):
        batch_size = q.shape[0]

        # joint_ang_vel = qd @ self.joint_axis
        # self.joint_vel = SpatialMotionVec(
        #     torch.zeros_like(joint_ang_vel), joint_ang_vel
        # )

        rot_angles_vals = self.joint_rot_angles()
        roll = rot_angles_vals[0, 0]
        pitch = rot_angles_vals[0, 1]
        yaw = rot_angles_vals[0, 2]

        fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # when we update the joint angle, we also need to update the transformation
        self.joint_pose.set_translation(
            torch.reshape(self.join_trans(), (1, 3)).repeat(batch_size, 1)
        )
        if torch.abs(self.joint_axis[0, 0]) == 1:
            rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q)
        elif torch.abs(self.joint_axis[0, 1]) == 1:
            rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q)
        else:
            rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q)

        self.joint_pose.set_rotation(fixed_rotation.repeat(batch_size, 1, 1) @ rot)
        return

    def get_joint_limits(self):
        return self.joint_limits

    def __repr__(self) -> str:
        # Pretty print all members
        return "\n"+"\n".join([f"{k}: {v}" for k, v in {
            "name": self.name,
            "children": [child.name for child in self._children],
            "parent": self._parent.name if self._parent else "None",
            "_device": self._device,
            "joint_idx": self.link_idx,
            "joint_axis": self.joint_axis,
            "joint_limits": self.joint_limits,
            "joint_pose": self.joint_pose,
        }.items()])+"\n"