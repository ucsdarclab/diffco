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
        self.joint_id = rigid_body_params["joint_id"]
        self.name = rigid_body_params["link_name"]

        self.trans = lambda: rigid_body_params["trans"].reshape(1, 3)
        self.rot_angles = lambda: rigid_body_params["rot_angles"].reshape(1, 3)

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]

        self.joint_limits = rigid_body_params["joint_limits"]

        self.joint_pose = CoordinateTransform(device=self._device)
        self.joint_pose.set_translation(torch.reshape(self.trans(), (1, 3)))

        self.update_joint_state(
            torch.zeros([1, 1], device=self._device),
        )

        self.pose = CoordinateTransform(device=self._device)

    # Kinematic tree construction
    def set_parent(self, link: "RigidBody"):
        self._parent = link

    def add_child(self, link: "RigidBody"):
        self._children.append(link)

    # Recursive algorithms
    def forward_kinematics(self, q_dict):
        """Recursive forward kinematics
        Computes transformations from self to all descendants.

        Returns: Dict[link_name, transform_from_self_to_link]
        """
        # Compute joint pose
        if self.name in q_dict:
            q = q_dict[self.name]
            batch_size = q.shape[0]

            rot_angles_vals = self.rot_angles()
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
                trans=torch.reshape(self.trans(), (1, 3)).repeat(batch_size, 1),
                device=self._device,
            )

        else:
            joint_pose = self.joint_pose

        # Compute forward kinematics of children
        pose_dict = {self.name: self.pose}
        for child in self._children:
            pose_dict.update(child.forward_kinematics(q_dict))

        # Apply joint pose
        # TODO: add center of mass to pose if getting body transforms (vs link transforms)
        return {
            body_name: joint_pose.multiply_transform(pose_dict[body_name])
            for body_name in pose_dict
        }

    # Get/set
    def update_joint_state(self, q):
        batch_size = q.shape[0]

        # joint_ang_vel = qd @ self.joint_axis
        # self.joint_vel = SpatialMotionVec(
        #     torch.zeros_like(joint_ang_vel), joint_ang_vel
        # )

        rot_angles_vals = self.rot_angles()
        roll = rot_angles_vals[0, 0]
        pitch = rot_angles_vals[0, 1]
        yaw = rot_angles_vals[0, 2]

        fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # when we update the joint angle, we also need to update the transformation
        self.joint_pose.set_translation(
            torch.reshape(self.trans(), (1, 3)).repeat(batch_size, 1)
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
            "joint_id": self.joint_id,
            "joint_axis": self.joint_axis,
            "joint_limits": self.joint_limits,
            "joint_pose": self.joint_pose,
        }.items()])+"\n"