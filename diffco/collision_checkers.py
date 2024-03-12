#
# These checkers use underlying interfaces to parse robot descriptions and environment representations, 
# parse arguments such as whether to use DiffCo/DiffCoBeta/MultiDiffCo, whether to fit labels or distances, whether uses sigmoid; 
# generate dataset if not provided; 
# 

import diffco
from diffco import model, kernel
import rospy
from rosgraph.names import ns_join
# from tools.planning_scene_editor import *
# from tools.get_state_validity import StateValidity
import torch
from tqdm import tqdm

# from tools.custom_log import custom_log_manager

class CollisionChecker:
    def normalizer(self, states):
        raise NotImplementedError
    
    def unnormalizer(self, states):
        raise NotImplementedError

class FCLBaxterChecker(CollisionChecker):
    def __init__(self, envDict, ns=''):
        rospy.init_node(f'node_{ns}_{__name__}')
        self.scene = PlanningSceneInterface(ns=ns)
        self.robot = RobotCommander(robot_description=ns_join(ns, 'robot_description'), ns=ns)
        self.group = MoveGroupCommander("right_arm", robot_description=ns_join(ns, 'robot_description'), ns=ns)
        self.scene._scene_pub = rospy.Publisher(ns_join(ns, 'planning_scene'),
                                           PlanningScene,
                                           queue_size=0)

        # global sv
        # global filler_robot_state
        # global rs_man

        self.sv = StateValidity(ns=ns)
        set_environment(self.robot, self.scene)

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

    def in_collision(self, states): #, print_depth=False):
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


class ProxyBaxterChecker(CollisionChecker):
    def __init__(self, envDict=None, gt_checker=None, ns='', gamma=10, beta=1.0, safety_bias=0, 
        num_init_points=10000, k=1, epsilon=1, verify_ratio=0.1, reuse_dataset=False, 
        lazy_line_checking=False):
        self.envDict = envDict
        if gt_checker is None:
            self.gt_checker = FCLBaxterChecker(envDict, ns=ns)
        else:
            self.gt_checker = gt_checker
        self.robot = model.BaxterRightArmFK()
        self.checker = diffco.DiffCo(envDict, kernel_func=kernel.FKKernel(
            self.robot.fkine, kernel.RQKernel(gamma)), beta=beta)
        self.gamma = gamma
        self.beta = beta
        self.num_init_points = num_init_points
        self.num_verify_points = int(num_init_points * verify_ratio)
        self.num_train_points = num_init_points - self.num_verify_points
        self.k = k
        self.epsilon = epsilon
        self.safety_bias = safety_bias
        self.lazy_line_checking = lazy_line_checking

        self._cuda = False

        self.reuse_dataset = reuse_dataset

        self._logger = custom_log_manager.get_logger(__class__.__name__)

    def _retrain(self, env_name=None):
        save_dataset = False
        if self.reuse_dataset:
            dataset_path = f"tmp/diffco_dataset_env={env_name}.pkl"
            if os.path.isfile(dataset_path):
                self._logger.info(f"Reusing dataset from {dataset_path}...")
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                    cfgs = dataset['cfg']
                    labels = dataset['label']
            else:
                save_dataset = True
        if not self.reuse_dataset or save_dataset:
            cfgs = torch.rand((self.num_init_points, self.robot.dof), dtype=torch.float)
            cfgs = cfgs*(self.robot.limits[:, 1]-self.robot.limits[:, 0])+self.robot.limits[:, 0]
            labels = torch.zeros(self.num_init_points, dtype=torch.float)
            norm_cfgs = self.gt_checker.normalizer(cfgs)
            for i, cfg in enumerate(tqdm(norm_cfgs, desc='Getting labels')):
                in_col = self.gt_checker.in_collision(cfg)
                labels[i] = 1 if in_col else -1
            if save_dataset:
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                with open(dataset_path, 'wb') as f:
                    dataset = {
                        'cfg': cfgs,
                        'label': labels
                    }
                    pickle.dump(dataset, f)
                    self._logger.info(f'diffco dataset saved to {dataset_path}.')
        train_cfgs = cfgs[:self.num_train_points]
        train_labels = labels[:self.num_train_points]
        self.verify_cfgs = cfgs[-self.num_verify_points:]
        self.verify_labels = labels[-self.num_verify_points:]
        self.checker.train(train_cfgs, train_labels, self.num_train_points)
        self.checker.fit_poly(kernel.Polyharmonic(self.k, self.epsilon), target='label', fkine=self.robot.fkine)
    
    def _accuracy_verify(self):
        verify_preds = [
            self.in_collision(self.gt_checker.normalizer(cfg)) for cfg in self.verify_cfgs]
        verify_preds = torch.FloatTensor(verify_preds) * 2 - 1
        test_acc = torch.sum(verify_preds == self.verify_labels, dtype=torch.float32)/len(verify_preds.view(-1))
        test_tpr = torch.sum(verify_preds[self.verify_labels==1] == 1, dtype=torch.float32) / len(verify_preds[self.verify_labels==1])
        test_tnr = torch.sum(verify_preds[self.verify_labels==-1] == -1, dtype=torch.float32) / len(verify_preds[self.verify_labels==-1])
        self._logger.info('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
        if test_acc < 0.9:
            self._logger.warning('test acc is only {}'.format(test_acc))
    
    def _calculate_safety_bias(self):
        scores = self.checker.rbf_score(self.verify_cfgs)[:, 0]
        min_score = scores.min()
        max_score = scores.max()
        min_polar_abs = min(min_score.abs(), max_score.abs())
        self.safety_bias = min(self.safety_bias, min_polar_abs/3)
        self._logger.info(f'Safety bias set to {self.safety_bias}, min_polar_score/3 = {min_polar_abs/3}')
    
    def cuda(self):
        self._cuda = True

    def reset_pose(self, pose_dict, env_name=None):
        self.gt_checker.reset_pose(pose_dict)
        self.robot = model.BaxterRightArmFK()
        self.checker = diffco.DiffCo(self.envDict, kernel_func=kernel.FKKernel(
            self.robot.fkine, kernel.RQKernel(self.gamma)), beta=self.beta)
        self._retrain(env_name)
        if self._cuda:
            self.checker.cuda()
            self.robot.cuda()
        self._calculate_safety_bias()
        self._accuracy_verify()
    
    def in_collision(self, state):
        # Input is the output of MPNet
        if states.ndim==1:
            states = states[None, :]
        states = self.gt_checker.unnormalizer(states)
        scores = self.checker.rbf_score(states)[:, 0]
        max_score = scores.max()
        return max_score + self.safety_bias > 0


class HybridBaxterChecker(ProxyBaxterChecker):
    def in_collision(self, states):
        # Input is the output of MPNet
        if states.ndim==1:
            states = states[None, :]
        raw_state = states
        states = self.gt_checker.unnormalizer(states)
        scores = self.checker.rbf_score(states)[:, 0]
        max_score, max_i = scores.max(dim=0)
        if max_score + self.safety_bias < 0:
            return False
        elif max_score - self.safety_bias > 0:
            return True
        else:
            if self.lazy_line_checking:
                return self.gt_checker.in_collision(raw_state[max_i])
            else:
                return self.gt_checker.in_collision(raw_state)

class OptimisticBaxterChecker(HybridBaxterChecker):
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
