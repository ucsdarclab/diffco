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
from collections import defaultdict

import torch

from yourdfpy import URDF
from trimesh import transformations as tf
import trimesh
import fcl
from fcl import defaultCollisionCallback
import numpy as np

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


class URDFRobotCollisionManager(trimesh.collision.CollisionManager):
    def __init__(self, urdf_robot: "URDFRobot", acm_cfgs=None):
        super().__init__()
        self.urdf_robot = urdf_robot
        
        # use self.geom_to_parent_link to check if the CollisionGeometry's of a link are allowed to collide with another link
        self.link_collision_objects, self.geom_name_to_parent_link = self._scene_to_collision(self.urdf_robot.robot.collision_scene)
        
        self._allowed_collisions = self._create_allowed_collision_matrix(cfgs=None) # disable collision between adjacent links first
        if acm_cfgs is not None:
            self._allowed_collisions = self._create_allowed_collision_matrix(cfgs=acm_cfgs)

    def _scene_to_collision(self, c_scene):
        """
        Convert objects in trimesh.Scene fcl CollisionObject's
        """
        link_collision_objects = defaultdict(list)
        # use this to check if the CollisionGeometry's of a link are allowed to collide with another link
        geom_name_to_parent_link = dict()
        for geometry_node_name in c_scene.graph.nodes_geometry:
            T, geometry = c_scene.graph[geometry_node_name]
            mesh = c_scene.geometry[geometry]
            cobj = self.add_object(name=geometry_node_name, mesh=mesh, transform=T)
            parent_link_name = c_scene.graph.transforms.parents[geometry_node_name]
            link_collision_objects[parent_link_name].append(geometry_node_name)
            geom_name_to_parent_link[geometry_node_name] = parent_link_name
        
        return link_collision_objects, geom_name_to_parent_link

    def _get_fcl_obj(self, mesh):
        if isinstance(mesh, trimesh.primitives.Box):
            return fcl.Box(*mesh.extents.astype(np.float32))
        elif isinstance(mesh, trimesh.primitives.Sphere):
            return fcl.Sphere(np.float32(mesh.primitive.radius))
        elif isinstance(mesh, trimesh.primitives.Cylinder):
            return fcl.Cylinder(np.float32(mesh.primitive.radius), np.float32(mesh.primitive.height))
        elif isinstance(mesh, trimesh.primitives.Capsule):
            return fcl.Capsule(np.float32(mesh.primitive.radius), np.float32(mesh.primitive.height))
        return super()._get_fcl_obj(mesh)


    def _create_allowed_collision_matrix(self, cfgs=None):
        """
        Create an allowed collision matrix for the robot
        """
        acm = set()
        for body in self.urdf_robot._bodies:
            if body._parent is None:
                continue
            parent_name = body._parent.name
            child_name = body.name
            acm.add((parent_name, child_name))
            acm.add((child_name, parent_name))
        
        if cfgs is not None:
            fk = self.urdf_robot.compute_forward_kinematics_all_links(cfgs, return_collision=True)
            num_cfgs = cfgs.shape[0]
            for i in range(num_cfgs):
                cur_colliding_link_pairs = set()
                self.set_fkine({k: [(t[0], r[0]) for t, r in v] for k, v in fk.items()})
                is_collision, contacts_data = self.in_collision_internal(return_data=True)
                for contact in contacts_data:
                    name1, name2 = tuple(contact.names)
                    parent_link1 = self.geom_name_to_parent_link[name1]
                    parent_link2 = self.geom_name_to_parent_link[name2]
                    cur_colliding_link_pairs.add((parent_link1, parent_link2))
                    cur_colliding_link_pairs.add((parent_link2, parent_link1))
                if i == 0:
                    always_colliding = cur_colliding_link_pairs
                else:
                    always_colliding = always_colliding.intersection(cur_colliding_link_pairs)
            acm = acm.union(always_colliding)
        return acm



    def _collision_callback(self, o1, o2, cdata):
        return defaultCollisionCallback(o1, o2, cdata)
        # default_done = defaultCollisionCallback(o1, o2, cdata)
        
        # if o1.geom in self.collision_object_id_to_parent_link and o2.geom in self.collision_object_id_to_parent_link:    
        #     link_name1 = self.collision_object_id_to_parent_link[o1.geom]
        #     link_name2 = self.collision_object_id_to_parent_link[o2.geom]
        #     if (link_name1, link_name2) in self._allowed_collisions or (link_name2, link_name1) in self._allowed_collisions:
        #         print('allowed collision', o1, o2)
        #         return cdata.done
        # else:
        #     pass
        # return
    
    def set_fkine(self, fk_dict: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]):
        for link_name, pieces_pose in fk_dict.items():
            for piece_pose, cobj_name in zip(
                pieces_pose, self.link_collision_objects[link_name]):
                piece_trans, piece_rot = piece_pose
                piece_rot = piece_rot.numpy()
                t = np.eye(4, dtype=piece_rot.dtype)
                t[:3, :3] = piece_rot
                t[:3, 3] = piece_trans.numpy()
                self.set_transform(cobj_name, t)
                
    
    def in_collision_internal(self, return_names=False, return_data=True):
        """
        Check if any pair of objects in the manager collide with one another.

        Parameters
        ----------
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data :  bool
          If true, a list of ContactData is returned as well

        Returns
        -------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names in alphabetical order indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(
                request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True)
            )

        self._manager.collide(cdata, self._collision_callback)
                
        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []
        if return_names or return_data:
            result = False # override result unless there are disallowed collisions
            for contact in cdata.result.contacts:
                names = (self._extract_name(contact.o1), self._extract_name(contact.o2))

                # filter out allowed collisions
                if self._allowed_collisions is not None:
                    parent_link1 = self.geom_name_to_parent_link[names[0]]
                    parent_link2 = self.geom_name_to_parent_link[names[1]]
                    if (parent_link1, parent_link2) in self._allowed_collisions or (parent_link2, parent_link1) in self._allowed_collisions:
                        continue
                
                result = True

                if return_names:
                    objs_in_collision.add(tuple(sorted(names)))
                if return_data:
                    contact_data.append(trimesh.collision.ContactData(names, contact))

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result

    def in_collision_other(self, other_manager, return_names=False, return_data=False):
        """
        Check if any object from this manager collides with any object
        from another manager.

        Parameters
        -------------------
        other_manager : CollisionManager
          Another collision manager object
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data : bool
          If true, a list of ContactData is returned as well

        Returns
        -------------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names (first from this manager,
          second from the other_manager) indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(
                request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True)
            )
        self._manager.collide(other_manager._manager, cdata, self._collision_callback)
        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []
        if return_names or return_data:
            for contact in cdata.result.contacts:
                reverse = False
                names = (
                    self._extract_name(contact.o1),
                    other_manager._extract_name(contact.o2),
                )
                if names[0] is None:
                    names = (
                        self._extract_name(contact.o2),
                        other_manager._extract_name(contact.o1),
                    )
                    reverse = True

                if return_names:
                    objs_in_collision.add(names)
                if return_data:
                    if reverse:
                        names = tuple(reversed(names))
                    contact_data.append(trimesh.collision.ContactData(names, contact))

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result



class URDFRobot:
    def __init__(
            self, 
            urdf_path, 
            name='', 
            device="cpu", 
            setup_acm=True,
            load_visual_meshes=False,):
        self.robot = URDF.load(
            urdf_path, 
            build_scene_graph=True,
            build_collision_scene_graph=True,
            load_meshes=load_visual_meshes,
            load_collision_meshes=True,
            force_collision_mesh=False)
        self.name = name
        self._device = torch.device(device)

        self._n_dofs = 0
        self._controlled_joints = []
        self._mimic_joints = defaultdict(list)
        self._bodies = []

        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for link_idx, link in enumerate(self.robot.link_map.values()):
            # Initialize body object
            rigid_body_params = self.get_body_parameters_from_urdf(link_idx, link)
            body = RigidBody(
                rigid_body_params=rigid_body_params, device=self._device
            )

            # Joint properties
            body.dof_idx = None
            if rigid_body_params["joint_type"] != "fixed":
                if body.joint_mimic is None:
                    body.dof_idx = self._n_dofs
                    self._n_dofs += 1
                    self._controlled_joints.append(link_idx)
                else:
                    self._mimic_joints[
                        self.robot.joint_map[body.joint_mimic.joint].child
                    ].append(body.name)

            # Add to data structures
            self._bodies.append(body)
            self._name_to_idx_map[body.name] = link_idx

        # Once all bodies are loaded, connect each body to its parent
        for body in self._bodies:
            if body.joint_name == 'base_joint':
                continue
            parent_body_name = self.robot.joint_map[body.joint_name].parent
            parent_body_idx = self._name_to_idx_map[parent_body_name]
            body.set_parent(self._bodies[parent_body_idx])
            self._bodies[parent_body_idx].add_child(body)
        
        # Calculate joint limits
        self.joint_limits = torch.zeros((self._n_dofs, 2), device=self._device)
        for i, body_idx in enumerate(self._controlled_joints):
            joint = self.robot.joint_map[self._bodies[body_idx].joint_name]
            self.joint_limits[i, 0] = joint.limit.lower
            self.joint_limits[i, 1] = joint.limit.upper
        
        # Collision manager maintains a fcl.DynamicAABBTreeCollisionManager,
        # and a dictionary between link names and its list of fcl collision objects
        # so that their transforms can be easily updated by forward kinematics function
        if setup_acm:
            acm_cfgs = self.rand_configs(100)
        else:
            acm_cfgs = None
        self.collision_manager = URDFRobotCollisionManager(self, acm_cfgs=acm_cfgs)

    def rand_configs(self, num_cfgs):
        return torch.rand(num_cfgs, self._n_dofs, device=self._device) * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]
    
    def collision(self, q, other=None, show=False):
        fk = self.compute_forward_kinematics_all_links(q, return_collision=True)
        batch_size = q.shape[0]
        collision_labels = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        for i in range(batch_size):
            pieces_pose = {
                link_name: [
                    (batch_piece_trans[i], batch_piece_rot[i])
                        for batch_piece_trans, batch_piece_rot in batch_pieces_pose if i < len(batch_piece_rot)
                ] for link_name, batch_pieces_pose in fk.items()
            } # type: Dict[str, List[Tuple[torch.Tensor[3], torch.Tensor[3, 3]]]]
            self.collision_manager.set_fkine(pieces_pose)

            if other is None:
                collision_labels[i], contacts_data  = self.collision_manager.in_collision_internal(return_data=True)
            else:
                collision_labels[i] = self.collision_manager.in_collision_other(other.collision_manager) or \
                    self.collision_manager.in_collision_internal(return_data=True)[0]
                
            if show:
                print(f"in collision?: {collision_labels[i]}")
                print(f"contact data: {[c.names for c in contacts_data]}")
                self.robot.update_cfg(q[i].numpy())
                
                scene = self.robot.collision_scene if self.robot.scene is None else self.robot.scene
                points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]) - [0.5, 0.5, 0]
                scene.camera_transform = scene.camera.look_at(points)
                scene.show()
                
        
        return collision_labels
    

    @tensor_check
    def compute_forward_kinematics_all_links(
        self, q: torch.Tensor,
        return_collision: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link
            return_collision: whether to return collision geometry transforms

        Returns: translation and rotation of the link frame

        """
        # Create joint state dictionary
        q_dict = {}
        for i, body_idx in enumerate(self._controlled_joints):
            body_name = self._bodies[body_idx].name
            q_dict[body_name] = q[:, i].unsqueeze(1)
            for mimic_body_name in self._mimic_joints[body_name]:
                q_dict[mimic_body_name] = q_dict[body_name]

        # Call forward kinematics on root node
        pose_dict = self._bodies[0].forward_kinematics(q_dict, return_collision)

        return {
            link: [(p.translation(), p.rotation()) for p in pose_dict[link]]
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

    def get_body_parameters_from_urdf(self, link_idx, link):
        body_params = {}
        body_params["link_idx"] = link_idx
        body_params["link_name"] = link.name

        # find joint whose "child" of this body according to urdf
        joint_name = self.find_joint_of_body(link.name)
        if joint_name is None:
            joint_rot_angles = torch.zeros(3, device=self._device)
            joint_trans = torch.zeros(3, device=self._device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self._device)
            joint_mimic = None
        else:
            joint = self.robot.joint_map[joint_name]
            joint_rot_angles = torch.tensor(
                tf.euler_from_matrix(joint.origin[:3, :3], axes='sxyz'), 
                dtype=torch.float32, device=self._device
            )
            joint_trans = torch.tensor(
                joint.origin[:3, 3], dtype=torch.float32, device=self._device
            )
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, device=self._device)
            joint_axis = torch.zeros((1, 3), device=self._device)
            joint_mimic = getattr(joint, "mimic", None)
            if joint_type != "fixed":
                joint_limits = {
                    "effort": joint.limit.effort,
                    "lower": joint.limit.lower,
                    "upper": joint.limit.upper,
                    "velocity": joint.limit.velocity,
                }
                try:
                    joint_damping = joint.dynamics.damping
                    if isinstance(joint_damping, str):
                        joint_damping = float(joint_damping)
                    joint_damping = torch.tensor(
                        [joint_damping],
                        dtype=torch.float32,
                        device=self._device,
                    )
                except AttributeError:
                    joint_damping = torch.zeros(1, device=self._device)
                joint_axis = torch.tensor(
                    joint.axis, dtype=torch.float32, device=self._device
                ).reshape(1, 3)

        body_params["joint_rot_angles"] = joint_rot_angles
        body_params["joint_trans"] = joint_trans
        body_params["joint_name"] = joint_name
        body_params["joint_type"] = joint_type
        body_params["joint_limits"] = joint_limits
        body_params["joint_damping"] = joint_damping
        body_params["joint_axis"] = joint_axis
        body_params["joint_mimic"] = joint_mimic

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
        
        if link.collisions is not None:
            collision_origins = []
            for collision in link.collisions:
                if collision.origin is None:
                    origin = torch.eye(
                        4, 
                        dtype=torch.float32,
                        device=self._device)
                else:
                    origin = torch.tensor(
                        collision.origin,
                        dtype=torch.float32,
                        device=self._device,
                    ).reshape((4, 4))
                collision_origins.append(origin)
            body_params["collision_origins"] = collision_origins
        
        if link.visuals is not None:
            visual_origins = []
            for visual in link.visuals:
                if visual.origin is None:
                    origin = torch.eye(
                        4, 
                        dtype=torch.float32,
                        device=self._device)
                else:
                    origin = torch.tensor(
                        visual.origin,
                        dtype=torch.float32,
                        device=self._device,
                    ).reshape((4, 4))
                visual_origins.append(origin)
            body_params["visual_origins"] = visual_origins

        return body_params



class KUKAiiwa(URDFRobot):
    def __init__(self, device=None, load_visual_meshes=False):
        rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.urdf_path, self.name, device=device, setup_acm=True, load_visual_meshes=load_visual_meshes)


class FrankaPanda(URDFRobot):
    def __init__(self, load_gripper=False, device=None, load_visual_meshes=False):
        rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf" if not load_gripper else "panda_description/urdf/panda.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "urdf_franka_panda"
        super().__init__(self.urdf_path, self.name, device=device, setup_acm=True, load_visual_meshes=load_visual_meshes)


class TwoLinkRobot(URDFRobot):
    def __init__(self, device=None, load_visual_meshes=False):
        rel_urdf_path = "2link_robot.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "urdf_2d_robot"
        super().__init__(self.urdf_path, self.name, device=device, setup_acm=True, load_visual_meshes=load_visual_meshes)


class TrifingerEdu(URDFRobot):
    def __init__(self, device=None, load_visual_meshes=False):
        rel_urdf_path = "trifinger_edu_description/trifinger_edu.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "trifinger_edu"
        super().__init__(self.urdf_path, self.name, device=device, setup_acm=True, load_visual_meshes=load_visual_meshes)
    

class URDFEnv:
    def __init__(self, urdf_path):
        raise NotImplementedError
    