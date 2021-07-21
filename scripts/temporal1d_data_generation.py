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
from diffco.Obstacles import ObstacleMotion, LinearMotion, SineMotion
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()

def main():
    # ========================== Data generqation =========================
    # !! Unlike previous scripts, starting from this one (temporal1d_data_generation.py)
    # I will try to normalize configurations (and time) to be between [0, 1]
    env_name = '1obs_sine'
    label_type = 'binary'  # [instance, class, binary]
    num_class = 2
    DOF = 1

    obstacles = {
        '1obs_linear': [(1, LinearMotion(A=1, B=1)), ],
        '1obs_sine': [(1, SineMotion(A=1, alpha=1, beta=0, bias=5))]
    }
    obstacles = obstacles[env_name]
    
    temporal_obs = [Simple1DDynamicObstacle(
        *obs) for obs in obstacles]

    joint_limits = torch.FloatTensor([0, 20])
    t_range = torch.FloatTensor([0, 10])

    robot = PointRobot1D(limits=torch.vstack([joint_limits, t_range]))

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
    cfgs = torch.rand((num_init_points, DOF+1), dtype=torch.float32)

    gt_checker = Simple1DDynamicChecker(temporal_obs, robot)

    st = time()
    labels, dists = gt_checker.predict(cfgs, distance=True)
    end = time()
    print('mean: {} secs.'.format((end-st)/num_init_points))
    print('{} collisions, {} free'.format(
        torch.sum(labels == 1), torch.sum(labels == -1)))
    dataset = {'data': cfgs, 'label': labels, 'dist': dists,
               'obs': obstacles, 'robot': robot.__class__, 'rparam': [robot.limits]}
    torch.save(dataset, 'data/temp1d_{}dof_{}.pt'.format(DOF, env_name))


if __name__ == "__main__":
    main()
