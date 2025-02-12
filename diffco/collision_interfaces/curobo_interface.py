from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import os
from collections import defaultdict
import time

import torch

from .robot_interface_base import RobotInterfaceBase

try:
    import curobo
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel as curobo_CudaRobotModel
    from curobo.geom.sdf.world import WorldCollision, CollisionBuffer, CollisionQueryBuffer
    from curobo.geom.sdf.world_mesh import WorldPrimitiveCollision, WorldMeshCollision
    from curobo.curobolib.geom import SelfCollisionDistance
except ImportError:
    print("curobo not found. Please install curobo to use the cuRobo interface")


class CuRoboCollisionWorldEnv:
    '''
    A thin wrapper around curobo.geom.sdf.world.WorldCollision to provide a interface for CuRoboInterface
    '''
    def __init__(self, curobo_collision_world: 'Union[WorldCollision, WorldMeshCollision, WorldPrimitiveCollision]'):
        self.curobo_collision_world = curobo_collision_world
    
    def get_sphere_collision(self, *args, **kwargs):
        return self.curobo_collision_world.get_sphere_collision(*args, **kwargs)


class CuRoboRobot(RobotInterfaceBase):
    def __init__(
            self, 
            cuda_robot_model: 'curobo_CudaRobotModel',
            name: str = "curobo_robot",
            unique_position_link_names: Optional[List[str]] = None,
            world_coll_activation_distance: float = 0.0,
            ):
        super().__init__(name=name, device=cuda_robot_model.tensor_args.device)
        self.cuda_robot_model = cuda_robot_model
        self.joint_limits = cuda_robot_model.get_joint_limits().position.transpose(0, 1)
        self._n_dofs = cuda_robot_model.get_dof()
        self._batch_size = cuda_robot_model._batch_size
        self.tensor_args = cuda_robot_model.tensor_args

        if unique_position_link_names is None:
            self.unique_position_link_names = cuda_robot_model.link_names
            print(f'cuda_robot_model.link_names: {cuda_robot_model.link_names}')
        else:
            self.unique_position_link_names = unique_position_link_names

        self._collision_query_buffer = CollisionQueryBuffer()
        self.world_coll_activation_distance = self.tensor_args.to_device([world_coll_activation_distance])
        self.env_query_idx = None
        self.weight = self.tensor_args.to_device([1.0])
        self.self_collision_kin_config = self.cuda_robot_model.get_self_collision_config()
        self.self_collision_weight = self.tensor_args.to_device([1.0])
        self.return_loss = False
        
        

    def rand_configs(self, num_cfgs):
        return torch.rand(num_cfgs, self._n_dofs, device=self._device) * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]
    
    def _update_self_collision_batch_size(self, robot_spheres):
        # from curobo/src/curobo/rollout/cost/self_collision_cost.py
        
        # Assuming n stays constant
        # TODO: use collision buffer here?

        if self._batch_size is None or self._batch_size != robot_spheres.shape:
            b, h, n, k = robot_spheres.shape
            self._out_distance = torch.zeros(
                (b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._out_vec = torch.zeros(
                (b, h, n, k), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self._batch_size = robot_spheres.shape
            self._sparse_sphere_idx = torch.zeros(
                (b, h, n), device=self.tensor_args.device, dtype=torch.uint8
            )
    
    def collision(
            self, 
            q, 
            other: Optional[CuRoboCollisionWorldEnv]=None,
        ):
        '''
        Assumes q = [batch_size x n_dofs]
        '''
        robot_spheres = self.cuda_robot_model.forward(q,)[-1]
        robot_spheres = robot_spheres.unsqueeze(0) if len(robot_spheres.shape) == 3 else robot_spheres

        if other is not None:
            self._collision_query_buffer.update_buffer_shape(
                robot_spheres.shape, self.tensor_args, other.curobo_collision_world.collision_types
            )
            robot_world_collision = other.get_sphere_collision(
                robot_spheres, 
                self._collision_query_buffer, 
                self.weight,
                env_query_idx=self.env_query_idx,
                activation_distance=self.world_coll_activation_distance,
                return_loss=False
            ) # Positive means in collision
            robot_world_collision = robot_world_collision.max(dim=-1).values
        
        self._update_self_collision_batch_size(robot_spheres)
        
        self_collision = SelfCollisionDistance.apply(
            self._out_distance,
            self._out_vec,
            self._sparse_sphere_idx,
            robot_spheres,
            self.self_collision_kin_config.offset,
            self.weight,
            self.self_collision_kin_config.collision_matrix,
            self.self_collision_kin_config.thread_location,
            self.self_collision_kin_config.thread_max,
            self.self_collision_kin_config.checks_per_thread,
            self.self_collision_kin_config.experimental_kernel,
            self.return_loss,
        )
        if other is not None:
            return (robot_world_collision[0] > 0) | (self_collision[0] > 0)
        else:
            return self_collision[0] < 0
    

    def forward_kinematics(self, q, link_names=None, return_collision=False):
        '''
        returns only positions unique position links for FKDiffCo.
        TODO: maybe in the future return rotations as well if needed.
        '''
        link_names = self.unique_position_link_names if link_names is None else link_names
        shape_tuple = q.shape
        q = q.view(-1, shape_tuple[-1]).contiguous()
        link_poses = self.cuda_robot_model.get_link_poses(q, link_names=link_names)  # .contiguous()
        # return link_poses
        link_pos = link_poses.position.reshape(shape_tuple[:-1] + link_poses.position.shape[1:])  # .contiguous()
        return link_pos
                           
    def compute_forward_kinematics_all_links(self, q, return_collision=False):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement compute_forward_kinematics_all_links, "
                                    "which is expected to return a dictionary of link names to poses. "
                                    "Instead it directly uses curobo's tensorized forward kinematics")