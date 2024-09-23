import numpy as np
from scipy.ndimage.measurements import label
import torch
from tqdm import tqdm
from time import time
from . import kernel
from .Obstacles import Obstacle
from .DiffCo import CollisionChecker
from .Obstacles import FCLObstacle
import fcl


class FCLChecker(CollisionChecker):
    def __init__(self, obstacles, robot, robot_manager=None, obs_managers=None, label_type=None, num_class=None):
        super().__init__(obstacles)
        self.robot = robot
        self.robot_manager = robot_manager
        self.obs_managers = obs_managers
        self.label_type = label_type
        self.num_class = num_class

        if self.robot_manager is None:
            rand_cfg = torch.rand(robot.dof)
            rand_cfg = rand_cfg * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
            robot_links = robot.update_polygons(rand_cfg)
            robot_manager = fcl.DynamicAABBTreeCollisionManager()
            robot_manager.registerObjects(robot_links)
            robot_manager.setup()
            self.robot_manager = robot_manager
        
        if self.obs_managers is None:
            assert label_type == 'binary' or label_type == 'instance' or \
                (label_type == 'class' and num_class is not None), \
                (f'When obs_managers is not provided one need to provide label type: \
                label_type={label_type}, num_class={num_class}')

            fcl_obs = [FCLObstacle(*param) for param in obstacles]
            fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

            if label_type == 'binary':
                obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
                obs_managers[0].registerObjects(fcl_collision_obj)
                obs_managers[0].setup()
            elif label_type == 'instance':
                obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in fcl_obs]
                for mng, cobj in zip(obs_managers, fcl_collision_obj):
                    mng.registerObjects([cobj])
            elif label_type == 'class':
                obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in range(num_class)]
                obj_by_cls = [[] for _ in range(num_class)]
                for obj in fcl_obs:
                    obj_by_cls[obj.category].append(obj.cobj)
                for mng, obj_group in zip(obs_managers, obj_by_cls):
                    mng.registerObjects(obj_group)
            else:
                raise NotImplementedError('Unsupported label_type {}'.format(label_type))
            
            for mng in obs_managers:
                mng.setup()
            
            self.obs_managers = obs_managers
        
        self.num_class = len(obs_managers)
    
    def predict(self, X, distance=True):
        if X.ndim == 1:
            X = X[None, :]
        labels = torch.FloatTensor(len(X), len(self.obs_managers))
        dists = torch.FloatTensor(len(X), len(self.obs_managers)) if distance else None
        req = fcl.CollisionRequest(num_max_contacts=1000 if distance else 1, enable_contact=distance)
        for i, cfg in enumerate(X):
            self.robot.update_polygons(cfg)
            self.robot_manager.update()
            # assert len(self.robot_manager.getObjects()) == self.robot.dof
            for cat, obs_mng in enumerate(self.obs_managers):
                rdata = fcl.CollisionData(request = req)
                self.robot_manager.collide(obs_mng, rdata, fcl.defaultCollisionCallback)
                in_collision = rdata.result.is_collision
                labels[i, cat] = 1 if in_collision else -1
                if distance:
                    ddata = fcl.DistanceData()
                    self.robot_manager.distance(obs_mng, ddata, fcl.defaultDistanceCallback)
                    depths = torch.FloatTensor([c.penetration_depth for c in rdata.result.contacts])
                    dists[i, cat] = depths.abs().max() if in_collision else -ddata.result.min_distance
        if distance:
            return labels, dists
        else:
            return labels
    
    def score(self, X):
        return self.predict(X, distance=True)[1]

class Simple1DDynamicChecker(CollisionChecker):
    def __init__(self, obstacles, robot):
        super().__init__(obstacles)
        self.robot = robot
    
    def predict(self, X, distance=True):
        if X.ndim == 1:
            X = X[None, :]
        # labels = torch.FloatTensor(len(X))
        # dists = torch.FloatTensor(len(X)) if distance else None
        X = self.robot.unnormalize(X)
        res = [obs.is_collision(X, distance=distance) for obs in self.obstacles]
        labels, dists = tuple(zip(*res))
        labels = (torch.vstack(labels).sum(dim=1) > 0) * 2 - 1
        if not distance:
            return labels
        dists = torch.max(torch.hstack(dists), dim=1).values
        # for i, (cfg, t) in enumerate(zip(X, ts)):
        #     res = [obs.is_collision(cfg[0], t) for obs in self.obstacles]
        #     in_collision = any([r[0] for r in res])
        #     labels[i] = 1 if in_collision else -1
        #     if distance:
        #         dists[i] = max([r[1] for r in res])
        return labels, dists

