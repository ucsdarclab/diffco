import argparse

import numpy as np
import torch
from diffco.model import RevolutePlanarRobot

from generate_batch_data_2d import generate_data_planar_manipulators

predefined_obstacles = {
    '2circle': [
        ('circle', (3, 2), 2),
        ('circle', (-2, 3), 0.5),
    ],
    '1rect_1circle': [
        ('rect', (4, 3), (2, 2)),
        ('circle', (-4, -3), 1)],
    '2rect': [
        ('rect', (4, 3), (2, 2)),
        ('rect', (-4, -3), (2, 2)),
    ],
    '1rect': [
        ('rect', (3, 2), (2, 2)),
    ],
    '3circle': [
        ('circle', (0, 4.5), 1),
        ('circle', (-2, -3), 2),
        ('circle', (-2, 2), 1.5),
    ],
    '1rect_1circle_7d': [
        ('circle', (-2, 3), 1),
        ('rect', (3, 2), (2, 2)),
    ],
    '2class_1': [
        ('rect', (5, 0), (2, 2), 0),
        ('circle', (-3, 6), 1, 1),
        ('rect', (-5, 2), (2, 1.5), 1),
        ('circle', (-5, -2), 1.5, 1),
        ('circle', (-3, -6), 1, 1),
    ],
    '2class_2': [
        ('rect', (0, 3), (16, 0.5), 1),
        ('rect', (0, -3), (16, 0.5), 0),
    ],
    '1rect_active': [
        ('rect', (-7, 3), (2, 2)),
    ],
    '3circle_7d': [
        ('circle', (-2, 2), 1),
        ('circle', (-3, 3), 1),
        ('circle', (-6, -3), 1),
    ],
    '2instance_big': [
        ('rect', (5, 4), (4, 4), 0),
        ('circle', (-5, -4), 2, 1),
    ],
    '7d_narrow': [],
    '3d_halfnarrow': [],
}

def setup_7d_narrow():
    lb = np.array([-8, 1.0], dtype=float)
    ub = np.array([8, 8], dtype=float)
    for i in range(150):
        pos = np.random.rand(2,)*(ub-lb)+lb
        pos = pos.tolist()
        size = (1, 1)
        predefined_obstacles['7d_narrow'].append(('rect', pos, size))
    
    lb = np.array([-8, -8], dtype=float)
    ub = np.array([8, -1.0], dtype=float)
    for i in range(150):
        pos = np.random.rand(2,)*(ub-lb)+lb
        pos = pos.tolist()
        size = (1, 1)
        predefined_obstacles['7d_narrow'].append(('rect', pos, size))

def setup_3dhalfnarrow():
    lb = np.array([-8, 1.0], dtype=float)
    ub = np.array([8, 8], dtype=float)
    for i in range(150):
        pos = np.random.rand(2,)*(ub-lb)+lb
        pos = pos.tolist()
        size = (1, 1)
        predefined_obstacles['3d_halfnarrow'].append(('rect', pos, size))

setup_7d_narrow()
setup_3dhalfnarrow()

def main(
        env_name: str = '3d_halfnarrow',
        folder: str = 'data/landscape',
        label_type: str = 'binary',
        num_class: int = 2,
        dof: int = 3,
        num_init_points: int = 8000,
        width: float = 0.3,
        link_length: float = 1.0,
        generate_random_cfgs: bool = True,
        random_seed: int = 2021) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    obstacles = predefined_obstacles[env_name]
    robot = RevolutePlanarRobot(link_length, width, dof)
    generate_data_planar_manipulators(robot, folder, obstacles, label_type=label_type,
        num_class=num_class, num_points=num_init_points, env_id=env_name, vis=True,
        generate_random_cfgs=generate_random_cfgs)


if __name__ == "__main__":
    desc = '2D data generation'
    parser = argparse.ArgumentParser(description=desc)
    env_choices = predefined_obstacles.keys()
    parser.add_argument('--env', dest='env_name', help='2D environment', choices=env_choices, default='3d_halfnarrow')
    parser.add_argument('-o', '--output-dir', dest='folder', default='data/landscape')
    parser.add_argument('-l', '--label-type', choices=['instance', 'class', 'binary'], default='binary')
    parser.add_argument('--num-classes', dest='num_class', default=2, type=int)
    parser.add_argument('--dof', help='degrees of freedom', choices=[2, 3, 7], default=3, type=int)
    parser.add_argument('--num-init-points', type=int, default=8000)
    parser.add_argument('--width', help='link width', type=float, default=0.3)
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--no-random-cfgs', dest='generate_random_cfgs', action='store_false', default=True)
    args = parser.parse_args()
    main(**vars(args))
