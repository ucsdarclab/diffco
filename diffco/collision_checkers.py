#
# These checkers use underlying interfaces to parse robot descriptions and environment representations, 
# parse arguments such as whether to use DiffCo/DiffCoBeta/MultiDiffCo, whether to fit labels or distances, whether uses sigmoid; 
# generate dataset if not provided; 
# 
from typing import Dict, Union, Optional
import os
import time

from functools import partial

import diffco
from diffco import model, kernel
from diffco.collision_interfaces import RobotInterfaceBase, URDFRobot, MultiURDFRobot, robot_description_folder, ROSRobotEnv, CuRoboRobot, CuRoboCollisionWorldEnv
from diffco.collision_interfaces import ShapeEnv, PCDEnv
from diffco.kernel_perceptrons import Perceptron, DiffCo, DiffCoBeta, MultiDiffCo
try:
    import rospy
except ImportError:
    pass
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
            robot: Optional[Union[str, RobotInterfaceBase]]=None, 
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
            if isinstance(environment, (ShapeEnv, CuRoboCollisionWorldEnv)):
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
            if isinstance(self.robot, RobotInterfaceBase):
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
    
    def _generate_dataset(self, q, labels, dists, num_samples, fix_joints=None, fix_joint_values=None, verbose=False):
        if q is None:
            q = self.robot.rand_configs(num_samples)
        if fix_joints is not None:
            q[:, fix_joints] = torch.tensor(fix_joint_values, dtype=q.dtype, device=q.device)
        num_samples = len(q)
        if labels is None:
            if verbose:
                print('Generating labels...')
                start_time = time.time()
            labels = self.gt_check_func(q)
            if verbose:
                print(f'Labels generated in {time.time()-start_time:.2f}s')
        else:
            labels = (labels > 0).type(q.dtype)
        if dists is None:
            dists = torch.zeros(num_samples, dtype=q.dtype, device=q.device)
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
        if kernel_func is None:
            self.kernel_func = kernel.RQKernel(perceptron_kwargs.pop('gamma', 10))
        else:
            self.kernel_func = kernel_func
        self.perceptron = perceptron_class(
            kernel_func=self.kernel_func,
            **perceptron_kwargs,
        )

    def fit(
            self, 
            q=None, labels=None, dists=None, 
            update=False, exist_mask=None,
            num_samples=5000, verify_ratio=0.1, 
            verbose=False, **get_dataset_kwargs):
        '''
        Used to train and update the DiffCo model.
        When verify_ratio is 0, the model trains without a follow-up verification,
        which is the default behavior when *updating* the model.
        When verify_ratio is True, the model trains with the full dataset and 
        verifies the model with self.q_verify.
        When 0 < verify_ratio < 1, the model trains with a portion of the dataset and
        verifies the model with the rest of the dataset.
        '''
        get_dataset_kwargs['verbose'] = not self.perceptron_trained
        q, labels, dists = self._generate_dataset(q, labels, dists, num_samples, **get_dataset_kwargs)

        num_samples = len(q)
        labels = (2*labels-1).type(q.dtype)
        if 0 < verify_ratio < 1:
            num_verify = int(verify_ratio*num_samples)
            verify_indices = torch.randperm(len(q))[:num_verify]
            verify_mask = torch.zeros(len(q), dtype=torch.bool)
            verify_mask[verify_indices] = True
            q_train, q_verify = q[~verify_mask], q[verify_mask]
            labels_train, labels_verify = labels[~verify_mask], labels[verify_mask]
            if verbose:
                print(f'Positive verify labels: {(labels_verify == 1).sum()}, Negative verify labels: {(labels_verify == -1).sum()}')
                print(f'label_verify: {labels_verify}')
            dists_train, dists_verify = dists[~verify_mask], dists[verify_mask]
        elif verify_ratio:
            raise ValueError(f'verify_ratio should be in (0, 1), got {verify_ratio}')
        else:
            q_train = q
            labels_train = labels
            dists_train = dists
            q_verify = self.robot.rand_configs(100)

        self.perceptron.train(
            q_train, labels_train, 
            update=update, exist_mask=exist_mask,
            max_iteration=len(q_train), distance=dists_train, verbose=verbose)
        inference_kernel_func = kernel.Polyharmonic(k=1, epsilon=1)
        self.perceptron.fit_poly(kernel_func=inference_kernel_func, target='label')

        self.safety_bias = self._calculate_safety_bias(q_verify)
        # Verification needs self.safety_bias
        if verify_ratio:
            verify_acc, verify_tpr, verify_tnr = self.verify(q_verify, labels_verify)
            self.q_verify = q_verify
        else:
            verify_acc, verify_tpr, verify_tnr = None, None, None
        
        self.perceptron_trained = True
        return verify_acc, verify_tpr, verify_tnr

    def update(self, q=None, labels=None, dists=None, 
               exploit_std=0.3, num_samples=100, num_exploit_samples=None, num_explore_samples=None,
               verify=False, verbose=False):
        '''
        Used to update the DiffCo model.
        Done: change the DiffCo perceptron class so it can update without re-initailize the model.
        '''
        num_exploit_samples = num_samples if num_exploit_samples is None else num_exploit_samples
        num_explore_samples = num_samples if num_explore_samples is None else num_explore_samples
        if q is None:
            if num_exploit_samples > len(self.perceptron.support_points):
                exploit_sample_mul = (num_exploit_samples // len(self.perceptron.support_points)) + \
                    (num_exploit_samples % len(self.perceptron.support_points) > 0)
                selected_indices = torch.arange(len(self.perceptron.support_points))
            else:
                exploit_sample_mul = 1
                selected_indices = torch.randperm(len(self.perceptron.support_points))[:num_exploit_samples]
            selected_support_points = self.perceptron.support_points[selected_indices]
            device = selected_support_points.device
            dtype = selected_support_points.dtype
            exploit_samples = torch.randn(exploit_sample_mul, len(selected_support_points), self.robot._n_dofs, dtype=dtype, device=device) * exploit_std + selected_support_points[None]
            exploit_samples = torch.clamp(exploit_samples, min=self.robot.joint_limits[:, 0], max=self.robot.joint_limits[:, 1])
            exploit_samples = exploit_samples.reshape(-1, self.robot._n_dofs)

            explore_samples = self.robot.rand_configs(num_explore_samples)
            q = torch.cat([exploit_samples, explore_samples, self.perceptron.support_points], dim=0)
            exist_mask = torch.zeros(len(q), dtype=torch.bool, device=device)
            exist_mask[-len(self.perceptron.support_points):] = True
        return self.fit(
            q, labels, dists, 
            update=True,
            exist_mask=exist_mask,
            verify_ratio=verify, verbose=verbose)

    def verify(self, q_verify=None, labels_verify=None, num_samples=None, verbose=False):
        if q_verify is None:
            if num_samples is not None:
                q_verify = self.robot.rand_configs(num_samples)
                self.q_verify = q_verify
            elif self.q_verify is not None:
                q_verify = self.q_verify
            else:
                raise ValueError('self.q_verify or num_samples should be provided')
        scores_verify = self.perceptron.poly_score(q_verify)
        # scores_verify = self.perceptron.score_original(q_verify)
        preds_verify = scores_verify > 0
        biased_preds_verify = scores_verify + self.safety_bias > 0
        preds_verify = 2 * preds_verify - 1
        biased_preds_verify = 2 * biased_preds_verify - 1

        if labels_verify is None:
            labels_verify = self.gt_check_func(q_verify)
            labels_verify = (2*labels_verify-1).type(q_verify.dtype)

        preds_verify = preds_verify.reshape_as(labels_verify)
        biased_preds_verify = biased_preds_verify.reshape_as(labels_verify)
        n_total, n_pos, n_neg = len(preds_verify), (labels_verify == 1).sum(), (labels_verify == -1).sum()
        test_acc = torch.sum(preds_verify == labels_verify, dtype=torch.float32) / n_total
        test_tpr = torch.sum(preds_verify[labels_verify == 1] == 1, dtype=torch.float32) / n_pos
        test_tnr = torch.sum(preds_verify[labels_verify == -1] == -1, dtype=torch.float32) / n_neg
        print(f'Positive labels: {n_pos.item()}, Negative labels: {n_neg.item()}')
        print(f'Test acc: {test_acc:.4f}, TPR {test_tpr:.4f}, TNR {test_tnr:.4f}')
        if verbose and test_acc < 0.9:
            print(f'test acc is only {test_acc:.4f}')
        test_acc = torch.sum(biased_preds_verify == labels_verify, dtype=torch.float32)/len(biased_preds_verify)
        test_tpr = torch.sum(biased_preds_verify[labels_verify == 1] == 1, dtype=torch.float32) / (labels_verify == 1).sum()
        test_tnr = torch.sum(biased_preds_verify[labels_verify == -1] == -1, dtype=torch.float32) / (labels_verify == -1).sum()
        print(f'Biased Test acc: {test_acc:.4f}, TPR {test_tpr:.4f}, TNR {test_tnr:.4f}')
        if verbose and test_acc < 0.9:
            print(f'Biased Test acc is only {test_acc:.4f}')
        return test_acc, test_tpr, test_tnr

    def collision(self, q):
        return self.collision_score(q) > 0
    
    def collision_score(self, q, bias: Union[float, torch.Tensor]=None):
        '''
        assume q is a tensor of shape (..., num_dof)
        '''
        bias = self.safety_bias if bias is None else bias
        shape_q = q.shape
        raw_scores = self.perceptron.poly_score(q.reshape(-1, shape_q[-1]))
        raw_scores = raw_scores.reshape(shape_q[:-1]+raw_scores.shape[1:])
        return raw_scores + bias

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


class ForwardKinematicsDiffCo(RBFDiffCo, CollisionChecker):
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
        CollisionChecker.__init__(
            self,
            robot=robot,
            robot_base_transform=robot_base_transform,
            environment=environment,
            robot_topic=robot_topic,
            planning_scene_topic=planning_scene_topic,
            gt_check_func=gt_check_func,
            device=device,
        )
        self.unique_position_link_names = []
        # TODO: remove base joints as its position is fixed
        if isinstance(self.robot, MultiURDFRobot):
            for robot_idx, link_body_list in enumerate(self.robot._bodies):
                for link_body in link_body_list:
                    if torch.any(link_body.joint_trans() != 0):
                        self.unique_position_link_names.append((robot_idx, link_body.name))
            self.tensorized_fkine = self.tensorized_fkine_multi_robot
        elif isinstance(self.robot, CuRoboRobot):
            self.tensorized_fkine = self.robot.forward_kinematics
        else:
            for link_body in self.robot._bodies:
                if torch.any(link_body.joint_trans() != 0):
                    self.unique_position_link_names.append(link_body.name)
            self.tensorized_fkine = self.tensorized_fkine_single_robot
        print(f'Unique position link names: {self.unique_position_link_names}')
        self.kernel_func = kernel.RQKernel(perceptron_kwargs.pop('gamma', 10))
        self.kernel_transform = partial(self.tensorized_fkine, return_collision=False)
        self.perceptron = perceptron_class(
            kernel_func=self.kernel_func,
            transform=self.kernel_transform,
            **perceptron_kwargs,
        )
        
        self.q_verify = None
        self.labels_verify = None
        self.safety_bias = 0
        self.perceptron_trained = False

    def tensorized_fkine_multi_robot(self, q, return_collision=False):
        unsqueezed = False
        if q.ndim == 1:
            q = q[None]
            unsqueezed = True
        fk_dicts = self.fkine(q, return_collision=return_collision)
        fk_tensors = torch.stack([pos for robot_idx, link_name in self.unique_position_link_names for pos, _ in fk_dicts[robot_idx][link_name]], dim=-1)
        if unsqueezed:
            fk_tensors = fk_tensors[0]

        return fk_tensors
    
    def tensorized_fkine_single_robot(self, q, return_collision=False):
        # start_time = time.time()
        fk_dict = self.fkine(q, return_collision=return_collision)
        # Stack the positions of every piece of every link. By default only joint poses are stacked,
        # i.e., one piece for each link in self.unique_position_link_names
        fk_tensor = torch.stack([pos for link_name in self.unique_position_link_names for pos, _ in fk_dict[link_name]], dim=-1)
        # end_time = time.time()
        # print(f'tensor FK time: {end_time-start_time:.6f}s')
        return fk_tensor
    
    def _uniform_sample_on_transformed_manifold(self, transform, num_samples):
        
        # from torch.func import jacfwd, vmap, jacrev
        # func_jacobian = vmap(jacfwd(transform, argnums=0), in_dims=0)
        # jacobian = torch.compile(jacobian)
        # This does not work with cuRobo's FK function as it does not
        # override setup_context staticmethod; using a naive implementation
        def jacobian(q):
            # q = q.unsqueeze(1)
            q = q.clone().detach().requires_grad_(True)
            bs = q.shape[0]
            pos = transform(q).reshape(bs, -1)
            output_dim = pos.shape[-1]
            jac = torch.zeros(bs, output_dim, q.shape[-1], device=q.device, dtype=q.dtype)
            for i in range(output_dim):
                q.grad = None
                pos.backward(torch.eye(output_dim, device=q.device)[i][None].expand_as(pos), retain_graph=True)
                jac[:, i] = q.grad
            # jac = torch.autograd.grad(pos, q, torch.eye(q.shape[-1]), create_graph=True)[0]
            return jac
        
        rand_q = self.robot.rand_configs(num_samples)
        rand_jac = jacobian(rand_q)
        # func_jac = func_jacobian(rand_q).reshape(num_samples, -1, rand_jac.shape[-1])
        # assert torch.allclose(rand_jac, func_jac, atol=1e-6), f'Jacobian mismatch: {rand_jac.max()}, {func_jac.max()}'
        rand_jac = rand_jac.reshape(num_samples, -1, rand_jac.shape[-1])
        print(f'rand_q: {rand_q.shape}')
        print(f'rand_jac: {rand_jac.shape}')
        if rand_jac.shape[-2] > rand_jac.shape[-1]:
            rand_jac = rand_jac.transpose(-2, -1)
        jac_det = torch.linalg.det(
            torch.matmul(rand_jac, rand_jac.transpose(-2, -1)) + 1e-4 * torch.eye(rand_jac.shape[-2], device=rand_jac.device)).sqrt()
        max_jac_det = 1.1 * jac_det.max()

        cnt_valid_q = 0
        valid_q = []
        while cnt_valid_q < num_samples:
            rej_u = torch.rand(len(rand_q), device=rand_q.device, dtype=rand_q.dtype)
            print(f'rej_u: {rej_u.shape}, max_jac_det: {max_jac_det}, jac_det: ({jac_det.min()}, {jac_det.max()})')
            mask = jac_det > rej_u * max_jac_det
            accepted_q = rand_q[mask]
            valid_q.append(accepted_q)
            cnt_valid_q += len(accepted_q)
            print(f'Valid q: {cnt_valid_q}/{num_samples}')
            if cnt_valid_q >= num_samples:
                break

            rand_q = self.robot.rand_configs(num_samples)
            rand_jac = jacobian(rand_q)
            rand_jac = rand_jac.reshape(num_samples, -1, rand_jac.shape[-1])
            if rand_jac.shape[-2] > rand_jac.shape[-1]:
                rand_jac = rand_jac.transpose(-2, -1)
            jac_det = torch.linalg.det(
                torch.matmul(rand_jac, rand_jac.transpose(-2, -1)) + 1e-4 * torch.eye(rand_jac.shape[-2], device=rand_jac.device)).sqrt()
        q = torch.cat(valid_q, dim=0)[:num_samples]

        return q

    def _generate_dataset(
            self, 
            q, labels, dists, 
            num_samples, 
            verbose=False, 
            sample_transform=None,
            **kwargs):
        transform = None
        if sample_transform == 'fkine':
            transform = self.tensorized_fkine
        elif callable(sample_transform):
            transform = sample_transform
        elif sample_transform is not None:
            raise ValueError(f'Invalid sample_transform: {sample_transform}')
        
        
        if transform is not None:
            q = self._uniform_sample_on_transformed_manifold(transform, num_samples)
            num_samples = len(q)
        return super()._generate_dataset(q, labels, dists, num_samples, verbose=verbose, **kwargs)
    
    def collision_score(
            self, 
            q: Optional[torch.Tensor]=None, 
            bias: Union[float, torch.Tensor]=None,
            q_link_pos: Optional[torch.Tensor]=None
        ):
        '''
        assume q is a tensor of shape (..., num_dof) or
        q_link_pos is a tensor of shape (..., num_links, 3)
        '''
        bias = self.safety_bias if bias is None else bias

        if q is not None:
            shape_q = q.shape
            raw_scores = self.perceptron.poly_score(point=q.reshape(-1, shape_q[-1]))
            raw_scores = raw_scores.reshape(shape_q[:-1]+raw_scores.shape[1:])
        elif q_link_pos is not None:
            shape_q_link_pos = q_link_pos.shape
            raw_scores = self.perceptron.poly_score(transformed_point=q_link_pos.reshape(-1, *shape_q_link_pos[-2:]))
            raw_scores = raw_scores.reshape(shape_q_link_pos[:-2]+raw_scores.shape[1:])
        return raw_scores + bias

    def _calculate_safety_bias(self, q_verify):
        scores = self.perceptron.poly_score(q_verify)[:, 0]
        min_score = scores.min()
        max_score = scores.max()
        min_polar_abs = min(min_score.abs(), max_score.abs())
        safety_bias = min_polar_abs/3
        return safety_bias

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
