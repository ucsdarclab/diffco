from time import time
import matplotlib.patheffects as path_effects
import sys
from diffco import Simple1DDynamicChecker
from diffco import utils
from diffco.Obstacles import Simple1DDynamicObstacle
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import PointRobot1D
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()

class ObstacleMotion:
    def predict(self, t):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class LinearMotion(ObstacleMotion):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def predict(self, t):
        return self.A * t + self.B

def main():
    # ========================== Data generqation =========================
    # !! Unlike previous scripts, starting from this one (temporal1d_data_generation.py)
    # I will try to normalize configurations (and time) to be between [0, 1]
    env_name = '1obs_linear'
    label_type = 'binary'  # [instance, class, binary]
    num_class = 2
    DOF = 1

    obstacles = {
        '1obs_linear': [(1, LinearMotion(A=1, B=1)), ]
    }
    obstacles = obstacles[env_name]
    t_range = torch.FloatTensor([0, 10])

    temporal_obs = [Simple1DDynamicObstacle(
        *obs) for obs in obstacles]

    robot = PointRobot1D(limits=[[0, 10]])

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
    cfgs = torch.rand((num_init_points, DOF), dtype=torch.float32)
    ts = torch.rand((num_init_points, 1), dtype=torch.float32)

    gt_checker = Simple1DDynamicChecker(temporal_obs)

    st = time()
    labels, dists = gt_checker.predict(
        cfgs * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0],
        ts * (t_range[1]-t_range[0]) + t_range[0])
    end = time()
    print('mean: {} secs.'.format((end-st)/num_init_points))
    print('{} collisions, {} free'.format(
        torch.sum(labels == 1), torch.sum(labels == -1))) 
    dataset = {'data': cfgs, 'time': ts, 'label': labels, 'dist': dists,
               'obs': obstacles, 'robot': robot.__class__, 'rparam': [robot.limits],
               't_range': t_range}
    torch.save(dataset, 'data/temp1d_{}dof_{}.pt'.format(DOF, env_name))


if __name__ == "__main__":
    main()
