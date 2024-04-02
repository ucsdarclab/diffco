# Description: This file contains the interfaces for different robots and environments
# They either read robot info and provides fkine function, or read obstacle info.
# Consider add parent classes for robot and environment interfaces, respectively.

# Copyright (c) Facebook, Inc. and its affiliates.
"""
Differentiable robot model class
====================================
TODO
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os

try:
    import rospy
    from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander, MoveItCommanderException
    from moveit_msgs.msg import RobotState, DisplayRobotState, PlanningScene, RobotTrajectory, ObjectColor
    from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point
    from rosgraph.names import ns_join
except ImportError:
    print("ROS-related imports failed. This is expected if not running in a ROS environment. "
          "Otherwise, try source your ROS setup.bash file or check your ROS installation.")

import torch

from .robot_interface_base import RobotInterfaceBase

class ROSRobotEnv(RobotInterfaceBase):
    def __init__(self, ns='', robot_topic=None, env_dict=None, name='', device='cpu'):
        rospy.init_node(f'node_{ns}_{__name__}')
        self.scene = PlanningSceneInterface(ns=ns)
        self.robot = RobotCommander(ns_join(ns, 'robot_description')) if robot_topic is None else RobotCommander(robot_topic)
        self.group = MoveGroupCommander("right_arm", robot_description=ns_join(ns, 'robot_description'), ns=ns)
        self.scene._scene_pub = rospy.Publisher(ns_join(ns, 'planning_scene'),
                                           PlanningScene,
                                           queue_size=0)

        self.sv = StateValidity(ns=ns)
        # set_environment(self.robot, self.scene)

        # masterModifier = ShelfSceneModifier()
        self.sceneModifier = PlanningSceneModifier(envDict['obsData'])
        self.sceneModifier.setup_scene(self.scene, self.robot, self.group)

        self.rs_man = RobotState()
        robot_state = self.robot.get_current_state()
        self.rs_man.joint_state.name = robot_state.joint_state.name
        self.filler_robot_state = list(robot_state.joint_state.position)
        self.joint_ranges = torch.FloatTensor(
            [3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])
        
        self.gt_checker = self

    def collision(self, q): #, print_depth=False):
        # returns true if robot state is in collision, false if robot state is collision free
        if states.ndim == 1: 
            states = states[None, :]
        states = self.unnormalizer(states)
        for state in states:
            self.filler_robot_state[10:17] = state # moveit_scrambler(state)
            self.rs_man.joint_state.position = tuple(self.filler_robot_state)
            collision_free = self.sv.getStateValidity(
                self.rs_man, group_name="right_arm") #, print_depth=print_depth)
            if not collision_free:
                return True
        return False

    def reset_pose(self, pose_dict, env_name=None):
        self.sceneModifier.delete_obstacles()
        self.sceneModifier.permute_obstacles(pose_dict)

    def normalizer(self, states):
        return moveit_unscrambler(states) / self.joint_ranges
    
    def unnormalizer(self, states):
        return moveit_scrambler(states * self.joint_ranges)

    def compute_forward_kinematics_all_links(self, q, return_collision=False):
        return super().compute_forward_kinematics_all_links(q, return_collision)
    

class PlanningSceneModifier():
    def __init__(self, obstacles, port=0):
        self._obstacles = obstacles

        self.port = port

        self._scene = None
        self._robot = None

    def setup_scene(self, scene, robot, group):
        self._scene = scene
        self._robot = robot
        self._group = group

    def permute_obstacles(self, pose_dict):
        for name in pose_dict.keys():
            pose = PoseStamped()
            pose.header.frame_id = self._robot.get_planning_frame()
            pose.pose.position.x = pose_dict[name][0]
            pose.pose.position.y = pose_dict[name][1]
            pose.pose.position.z = pose_dict[name][2] + self._obstacles[name]['z_offset']

            # Keep the orientations
            if 'orientation' not in pose_dict[name] and self._obstacles[name]['orientation'] is not None:
                pose.pose.orientation.x = self._obstacles[name]['orientation'][0]
                pose.pose.orientation.y = self._obstacles[name]['orientation'][1]
                pose.pose.orientation.z = self._obstacles[name]['orientation'][2]
                pose.pose.orientation.w = self._obstacles[name]['orientation'][3]

            if self._obstacles[name]['is_mesh']:
                # _logger.info(self._obstacles[name]['mesh_file'])
                self._scene.add_mesh(name, pose, filename=self._obstacles[name]['mesh_file'], size=self._obstacles[name]['dim'])
            else:
                self._scene.add_box(name, pose, size=self._obstacles[name]['dim'])

        # rospy.sleep(1)
        # _logger.info(self._scene.get_known_object_names())

    def delete_obstacles(self):
        #scene.remove_world_object("table_center")
        for name in self._obstacles.keys():
            self._scene.remove_world_object(name)