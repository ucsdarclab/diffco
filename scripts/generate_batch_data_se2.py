import os

import matplotlib as mpl
import numpy as np
import torch
from diffco import utils
from diffco.model import RigidPlanarBody
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from numpy.random import rand, randint

from generate_batch_data_2d import build_dataset


def create_plots(robot, obstacles, cfg=None, label=None):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111) #, projection='3d'

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks([-4, 0, 4])
    # ax.set_yticks([-4, 0, 4])
    for obs in obstacles:
        cat = obs[3] if len(obs) >= 4 else 1
        # print('{}, cat {}, {}'.format(obs[0], cat, obs))
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], color=cmaps[cat](0.5))) #path_effects=[path_effects.withSimplePatchShadow()], 
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], 
            color=cmaps[cat](0.5))) #, path_effects=[path_effects.withSimplePatchShadow()]
    
    # Plot robot
    if cfg is None:
        cfg = torch.rand(1, robot.dof, dtype=torch.float32)
        cfg = cfg * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        label = torch.zeros(len(cfg))
    cfg = cfg.reshape(-1, 3) 
    label = label.reshape(-1, 1)
    for q, l in zip(cfg, label):
        points = robot.fkine(q)[0]
        for p, trans in zip(robot.parts, points):
            if p[0] == 'circle':
                ax.add_patch(Circle(trans, p[2], color=cmaps[cat](0.5))) #path_effects=[path_effects.withSimplePatchShadow()], 
            elif p[0] == 'rect':
                lower_left = torch.FloatTensor([-float(p[2][0])/2, -float(p[2][1])/2])
                R = utils.rot_2d(q[2:3])[0]
                lower_left_position = R@lower_left + trans
                rect_patch = Rectangle([0, 0], p[2][0], p[2][1], color='grey' if l < 0 else 'red',) #
                tf = mpl.transforms.Affine2D().rotate(q[2].item()).translate(*lower_left_position) + ax.transData
                rect_patch.set_transform(tf)
                ax.add_patch(rect_patch) #, path_effects=[path_effects.withSimplePatchShadow()]

def generate_obstacles_for_rigid_body(obs_num: int) -> list:
    obstacles = []
    types = ['rect', 'circle']
    # link_length = robot.link_length[0].item()
    for i in range(obs_num):
        obs_t = types[randint(2)]
        if types[0] in obs_t: # rectangle, size = 0.5-3.5, pos = -7~7
            s = rand(2) * 3 + 0.5
            if obs_num <= 2:
                p = rand(2) * 10 - 5
            else:
                p = rand(2) * 14 - 7
        elif types[1] in obs_t: # circle, size = 0.25-2, pos = -7~7
            s = rand() * 1.75 + 0.25
            if obs_num <= 2:
                p = rand(2) * 10 - 5
            else:
                p = rand(2) * 14 - 7
        obstacles.append((obs_t, p, s))
    return obstacles

def generate_data_rigid_body(
        robot,
        folder: str,
        obs_num: int,
        label_type: str = 'binary',
        num_class: int = None,
        num_points: int = 8000,
        env_id: str = '',
        vis: bool = True):
    """Generate dataset for a 2D rigid body robot.
    """
    obstacles = generate_obstacles_for_rigid_body(obs_num)
    robot, cfgs, labels, dists = build_dataset(robot, obstacles, num_points, num_class, label_type, env_id)

    dataset = {
        'data': cfgs, 'label': labels, 'dist': dists, 'obs': obstacles, 
        'robot': robot.__class__, 'rparam': [robot.parts]
    }
    os.makedirs(folder, exist_ok=True)
    torch.save(dataset, os.path.join(
        folder, 'se2_{}obs_{}_{}.pt'.format(obs_num, label_type, env_id)))

    if vis:
        create_plots(robot, obstacles, cfg=cfgs[:100], label=labels[:100] )# torch.FloatTensor([2.5, -5, np.pi/6]))
        plt.savefig(os.path.join(
            folder, 'se2_{}obs_{}_{}.png'.format(obs_num, label_type, env_id)))
        plt.close()
    
    return 

if __name__ == "__main__":
    batch_name = 'exp1'

    label_type = 'binary' #[instance, class, binary]
    seed = 1917

    # type, offset, dimensions
    parts = [
        ('rect', (-1, 0), (0.3, 2)), 
        ('rect', (1, 0), (0.3, 2)), 
        ('rect', (0, 1), (2, 0.3)), 
        ]
    robot = RigidPlanarBody(parts)

    folder_name = os.path.join('data', 'se2_{}'.format(batch_name))
    if os.path.isdir(folder_name):
        ans = input('Folder {} exists. Continue? (Y/n)'.format(folder_name))
        if 'y' in ans or 'Y' in ans:
            pass
        else:
            exit(1)
    else:
        os.makedirs(folder_name)
    print('Start writing data in {}'.format(folder_name))

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    num_envs_per_num_obs = 1
    seq_num_obs = [10] # [5, 10] # [1, 2, 5, 10, 20]
    for num_obs in seq_num_obs:
        for env_id in range(num_envs_per_num_obs):
            # ==== may be used to regenerate some data with no collision ====
            # file_name = os.path.join(
            # folder_name, '2d_{}dof_{}obs_{}_{}.pt'.format(robot.dof, num_obs, label_type, env_id))
            # if os.path.isfile(file_name):
            #     with open(file_name) as f:
            #         labels = torch.load(f)['labels']
            #         if torch.sum(labels==1) != 0:
            #             continue
            # ==============================================================
            generate_data_rigid_body(robot, folder_name, num_obs, num_points=8000, env_id='{:02d}'.format(env_id), vis=True)
