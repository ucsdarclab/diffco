import numpy as np
from scipy.ndimage.measurements import label
import torch
from tqdm import tqdm
from time import time
from . import kernel
from .Obstacles import Obstacle
from .DiffCo import CollisionChecker
import fcl


class FCLChecker(CollisionChecker):
    def __init__(self, obstacles, robot, robot_manager, obs_managers):
        super().__init__(obstacles)
        self.robot = robot
        self.robot_manager = robot_manager
        self.obs_managers = obs_managers
        self.num_class = len(obs_managers)
    
    def predict(self, X, distance=True):
        labels = torch.FloatTensor(len(X), len(self.obs_managers))
        dists = torch.FloatTensor(len(X), len(self.obs_managers)) if distance else None
        req = fcl.CollisionRequest(num_max_contacts=1000 if distance else 1, enable_contact=distance)
        for i, cfg in enumerate(X):
            self.robot.update_polygons(cfg)
            self.robot_manager.update()
            assert len(self.robot_manager.getObjects()) == self.robot.dof
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