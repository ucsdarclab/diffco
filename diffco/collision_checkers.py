#
# These checkers use underlying interfaces to parse robot descriptions and environment representations, 
# parse arguments such as whether to use DiffCo/DiffCoBeta/MultiDiffCo, whether to fit labels or distances, whether uses sigmoid; 
# generate dataset if not provided; 
# 
from typing import Dict, Union
import os

from functools import partial

import diffco
from diffco import model, kernel
from diffco.collision_interfaces import RobotInterfaceBase, URDFRobot, MultiURDFRobot, robot_description_folder, ROSRobotEnv
from diffco.collision_interfaces import ShapeEnv, PCDEnv
from diffco.kernel_perceptrons import Perceptron, DiffCo, DiffCoBeta, MultiDiffCo
import rospy
# from tools.planning_scene_editor import *
# from tools.get_state_validity import StateValidity
import torch
from tqdm import tqdm

# from tools.custom_log import custom_log_manager

class CollisionChecker:
    '''
    This base class defines the interfaces for the robot and (optionally) the environment.
    Also, depends on the robot and environment representations, it defines a ground truth collision checker,
    except when a ground truth collision checker is provided by the user.

    From robot, environment, and planning_scene_topic, provide one combination of the following arguments:
    1. robot: str if you are using a URDF file or RobotInterfaceBase
    2. robot: str if you are using a URDF file or RobotInterfaceBase, and environment: Dict, if you are using a ShapeEnv, or PCDEnv
    3. robot_topic: str, and planning_scene_topic: str, if you want to use a ROS robot interface, which includes information of both robot and environment
    '''
    def __init__(
            self, 
            robot: Union[str, RobotInterfaceBase]=None, 
            robot_base_transform: torch.Tensor=None,
            environment: Union[Dict, ShapeEnv, PCDEnv]=None,
            robot_topic: str=None,
            planning_scene_topic: str=None,
            gt_check_func=None,
            device='cpu',
            ) -> None:
        if isinstance(robot, str):
            if os.path.isfile(robot):
                assert robot_topic is None, 'robot_topic should be None if a URDF file path is provided'
                name = os.path.basename(robot).split('.')[0]
                robot = URDFRobot(
                    robot, 
                    name=name,
                    base_transform=robot_base_transform,
                    device=device,)
            else:
                raise ValueError('Invalid robot URDF file path')
        if robot_topic is not None:
            assert robot is None, 'robot should be None if robot_topic is provided'
            name = robot_topic.split('/')[-1]
            robot = ROSRobotEnv(
                robot_topic=robot_topic,
                planning_scene_topic=planning_scene_topic,
                name=name,
                device=device,
            )
        self.robot = robot
        if environment is not None:
            if isinstance(environment, ShapeEnv):
                pass 
            elif isinstance(environment, Dict):
                environment = ShapeEnv(environment)
            elif isinstance(environment, PCDEnv):
                raise NotImplementedError(f'PCDEnv is not supported in {__class__.__name__} yet')
            else:
                raise ValueError('Invalid environment representation')
        self.environment = environment
        self.device = device

        if gt_check_func is None:
            if isinstance(self.robot, URDFRobot):
                self.gt_check_func = partial(self.robot.collision, other=self.environment)
            else:
                self.gt_check_func = self.robot.collision
        else:
            self.gt_check_func = gt_check_func

    def collision(self, q):
        return self.gt_check_func(q)
    
    def fkine(self, q, return_collision=False, **kwargs):
        '''
        q: torch.Tensor, shape=(num_samples, num_dof)
        return_collision: bool, whether to return poses of collision geometries, or just
            the joint poses of the robot
        '''
        fk_dict = self.robot.compute_forward_kinematics_all_links(q, return_collision=return_collision, **kwargs)
        return fk_dict

    def normalizer(self, unnormalized_q):
        raise NotImplementedError
    
    def unnormalizer(self, normalized_q):
        raise NotImplementedError
    
    def _generate_dataset(self, q, labels, dists, num_samples):
        if q is None:
            q = self.robot.rand_configs(num_samples)
        if labels is None:
            print('Generating labels...')
            labels = self.gt_check_func(q)
            print('Labels generated.')
        if dists is None:
            dists = torch.zeros(num_samples)
        return q, labels, dists

class RBFDiffCo(CollisionChecker):
    '''
    The vanilla DiffCo implementation with RBF kernel, 
    without Forward Kinematics transformation.
    '''
    def __init__(
            self, 
            robot: Union[str, RobotInterfaceBase]=None, 
            robot_base_transform: torch.Tensor=None,
            environment: Union[Dict, ShapeEnv, PCDEnv]=None,
            robot_topic: str=None,
            planning_scene_topic: str=None,
            gt_check_func=None,
            device='cpu',
            kernel_func=None,
            perceptron_class: Perceptron=DiffCo,
            **perceptron_kwargs,
            ) -> None:
        super().__init__(
            robot=robot,
            robot_base_transform=robot_base_transform,
            environment=environment,
            robot_topic=robot_topic,
            planning_scene_topic=planning_scene_topic,
            gt_check_func=gt_check_func,
            device=device,
        )
        if kernel_func is None:
            self.kernel_func = kernel.RQKernel(perceptron_kwargs.pop('gamma', 10))
        else:
            self.kernel_func = kernel_func
        self.perceptron = perceptron_class(
            kernel_func=self.kernel_func,
            **perceptron_kwargs,
        )

    def collision(self, q):
        return self.perceptron.score(q) > 0
    
    def collision_score(self, q):
        return self.perceptron.score(q)

    def normalizer(self, unnormalized_q):
        '''
        Normalize the joint values to [0, 1]
        '''
        return (unnormalized_q-self.robot.joint_limits[:, 0])/(self.robot.joint_limits[:, 1]-self.robot.joint_limits[:, 0])
    
    def unnormalizer(self, normalized_q):
        '''
        Unnormalize the joint values between [0, 1] to the original joint limits
        '''
        return normalized_q*(self.robot.joint_limits[:, 1]-self.robot.joint_limits[:, 0])+self.robot.joint_limits[:, 0]


class ForwardKinematicsDiffCo(CollisionChecker):
    '''
    The DiffCo implementation with Forward Kinematics transformation.
    Recommended for robot manipulators.
    '''
    def __init__(
            self, 
            robot: Union[str, RobotInterfaceBase]=None, 
            robot_base_transform: torch.Tensor=None,
            environment: Union[Dict, ShapeEnv, PCDEnv]=None,
            robot_topic: str=None,
            planning_scene_topic: str=None,
            gt_check_func=None,
            device='cpu',
            perceptron_class=DiffCo,
            **perceptron_kwargs,
            ) -> None:
        super().__init__(
            robot=robot,
            robot_base_transform=robot_base_transform,
            environment=environment,
            robot_topic=robot_topic,
            planning_scene_topic=planning_scene_topic,
            gt_check_func=gt_check_func,
            device=device,
        )
        self.unique_position_link_names = []
        for link_body in self.robot._bodies:
            if torch.any(link_body.joint_trans() != 0):
                self.unique_position_link_names.append(link_body.name)
        print(f'Unique position link names: {self.unique_position_link_names}')
        self.kernel_func = kernel.FKKernel(
            partial(self.tensorized_fkine, return_collision=False),
            kernel.RQKernel(perceptron_kwargs.pop('gamma', 10)),
        )
        self.perceptron = perceptron_class(
            kernel_func=self.kernel_func,
            **perceptron_kwargs,
        )
        
        self.q_verify = None
        self.labels_verify = None
        self.safety_bias = 0
        self.perceptron_trained = False

    
    def tensorized_fkine(self, q, return_collision=False):
        fk_dict = self.fkine(q, return_collision=return_collision)
        # Stack the positions of every piece of every link. By default only joint poses are stacked,
        # i.e., one piece for each link in self.unique_position_link_names
        fk_tensor = torch.stack([pos for link_name in self.unique_position_link_names for pos, _ in fk_dict[link_name]], dim=-1)
        return fk_tensor

    def train(self, q=None, labels=None, dists=None, num_samples=5000, verify_ratio=0.1):
        '''
        Used to train and update the DiffCo model.
        '''
        q, labels, dists = self._generate_dataset(q, labels, dists, num_samples)
        labels = (2*labels-1).type(q.dtype)
        verify_indices = torch.randperm(len(q))[:int(verify_ratio*len(q))]
        verify_mask = torch.zeros(len(q), dtype=torch.bool)
        verify_mask[verify_indices] = True
        q_train, q_verify = q[~verify_mask], q[verify_mask]
        labels_train, labels_verify = labels[~verify_mask], labels[verify_mask]
        dists_train, dists_verify = dists[~verify_mask], dists[verify_mask]

        self.perceptron.train(q_train, labels_train, max_iteration=len(q_train), distance=dists_train)
        inference_kernel_func = kernel.Polyharmonic(k=1, epsilon=1)
        self.perceptron.fit_poly(kernel_func=inference_kernel_func, target='label', fkine=self.tensorized_fkine)
        if not self.perceptron_trained:
            self.q_verify = q_verify

        self.safety_bias = self._calculate_safety_bias(self.q_verify)
        print(f'Safety bias: {self.safety_bias}')
        # Verification needs self.safety_bias
        if not self.perceptron_trained:
            verify_acc, verify_tpr, verify_tnr = self._accuracy_verify(self.q_verify, labels_verify)
        
        self.perceptron_trained = True
        return verify_acc, verify_tpr, verify_tnr

    def _calculate_safety_bias(self, q_verify):
        scores = self.perceptron.poly_score(q_verify)[:, 0]
        min_score = scores.min()
        max_score = scores.max()
        min_polar_abs = min(min_score.abs(), max_score.abs())
        safety_bias = min_polar_abs/3
        print(f'min_polar_score/3 = {min_polar_abs/3}')
        return safety_bias

    def _accuracy_verify(self, q_verify, labels_verify):
        scores_verify = self.perceptron.poly_score(q_verify)
        preds_verify = scores_verify > 0
        biased_preds_verify = scores_verify + self.safety_bias > 0
        preds_verify = 2 * preds_verify - 1
        biased_preds_verify = 2 * biased_preds_verify - 1
        preds_verify = preds_verify.reshape_as(labels_verify)
        biased_preds_verify = biased_preds_verify.reshape_as(labels_verify)
        test_acc = torch.sum(preds_verify == labels_verify, dtype=torch.float32)/len(preds_verify)
        test_tpr = torch.sum(preds_verify[labels_verify == 1] == 1, dtype=torch.float32) / (labels_verify == 1).sum()
        test_tnr = torch.sum(preds_verify[labels_verify == -1] == -1, dtype=torch.float32) / (labels_verify == -1).sum()
        print('Test acc: {:.4f}, TPR {:.4f}, TNR {:.4f}'.format(test_acc, test_tpr, test_tnr))
        if test_acc < 0.9:
            print('test acc is only {:.4f}'.format(test_acc))
        test_acc = torch.sum(biased_preds_verify == labels_verify, dtype=torch.float32)/len(biased_preds_verify)
        test_tpr = torch.sum(biased_preds_verify[labels_verify == 1] == 1, dtype=torch.float32) / (labels_verify == 1).sum()
        test_tnr = torch.sum(biased_preds_verify[labels_verify == -1] == -1, dtype=torch.float32) / (labels_verify == -1).sum()
        print('Biased Test acc: {:.4f}, TPR {:.4f}, TNR {:.4f}'.format(test_acc, test_tpr, test_tnr))
        if test_acc < 0.9:
            print('Biased Test acc is only {:.4f}'.format(test_acc))
        return test_acc, test_tpr, test_tnr
    
    def collision(self, q):
        return self.collision_score(q) > 0
    
    def collision_score(self, q, bias: Union[float, torch.Tensor]=None):
        bias = self.safety_bias if bias is None else bias
        return self.perceptron.poly_score(q) + bias

    def normalizer(self, unnormalized_q):
        return (unnormalized_q-self.robot.joint_limits[:, 0])/(self.robot.joint_limits[:, 1]-self.robot.joint_limits[:, 0])
    
    def unnormalizer(self, normalized_q):
        return normalized_q*(self.robot.joint_limits[:, 1]-self.robot.joint_limits[:, 0])+self.robot.joint_limits[:, 0]

class HybridForwardKinematicsDiffCo(ForwardKinematicsDiffCo):
    def __init__(
            self, 
            robot: Union[str, RobotInterfaceBase]=None, 
            robot_base_transform: torch.Tensor=None,
            environment: Union[Dict, ShapeEnv, PCDEnv]=None,
            robot_topic: str=None,
            planning_scene_topic: str=None,
            gt_check_func=None,
            device='cpu',
            lazy_line_check=False,
            perceptron_class: Perceptron=DiffCo,
            **perceptron_kwargs,
            ) -> None:
        super().__init__(
            robot=robot,
            robot_base_transform=robot_base_transform,
            environment=environment,
            robot_topic=robot_topic,
            planning_scene_topic=planning_scene_topic,
            gt_check_func=gt_check_func,
            device=device,
            perceptron_class=perceptron_class,
            **perceptron_kwargs,
        )
        self.lazy_line_check = lazy_line_check

    def collision(self, q):
        unbias_scores = self.collision_score(q, bias=0)
        collision_labels = unbias_scores + self.safety_bias > 0
        if self.lazy_line_check:
            max_score, max_i = unbias_scores.max(dim=0)
            collision_labels[max_i] = self.gt_check_func(q[max_i])
        else:
            uncertain_q_mask = torch.logical_and(unbias_scores + self.safe_bias > 0, unbias_scores - self.safe_bias < 0)
            uncertain_q = q[uncertain_q_mask]
            collision_labels[uncertain_q_mask] = self.gt_check_func(uncertain_q)
        return collision_labels

class OptimisticBaxterChecker(HybridForwardKinematicsDiffCo):
    def in_collision(self, states, optimistic=False):
        if optimistic:
            if states.ndim==1:
                states = states[None, :]
            # raw_state = states
            states = self.gt_checker.unnormalizer(states)
            scores = self.checker.rbf_score(states)[:, 0]
            max_score, max_i = scores.max(dim=0)
            return max_score - self.safety_bias > 0
        else:
            return super().in_collision(states)
