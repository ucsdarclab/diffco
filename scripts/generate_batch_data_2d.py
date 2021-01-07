import sys
# sys.path.append('/home/yuheng/DiffCo/')
import os
from time import time

import numpy as np
from numpy.random import rand, randint
import torch
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects

from diffco import DiffCo
from diffco import kernel
from matplotlib import pyplot as plt
from diffco.model import RevolutePlanarRobot
from diffco import utils
from diffco.Obstacles import FCLObstacle

def create_plots(robot, obstacles, cfg=None):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111) #, projection='3d'

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
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
    points = robot.fkine(cfg)[0]
    points = torch.cat([torch.zeros(1, points.shape[1]), points], dim=0)
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot(points[:, 0], points[:, 1], color='silver', lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    joint_plot, = ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', markersize=lw)

def generate_one(robot, obs_num, folder, label_type='binary', num_class=None, num_points=8000, env_id='', vis=True):
    obstacles = []
    types = ['rect', 'circle']
    link_length = robot.link_length[0].item()
    for i in range(obs_num):
        obs_t = types[randint(2)]
        if types[0] in obs_t: # rectangle, size = 0.5-3.5, pos = -7~7
            while True:
                s = rand(2) * 3 + 0.5
                if obs_num <= 2:
                    p = rand(2) * 10 - 5
                else:
                    p = rand(2) * 14 - 7
                if any(p-s/2 > link_length) or any(p+s/2 < -link_length): # the near-origin area is clear
                    break
        elif types[1] in obs_t: # circle, size = 0.25-2, pos = -7~7
            while True:
                s = rand() * 1.75 + 0.25
                if obs_num <= 2:
                    p = rand(2) * 10 - 5
                else:
                    p = rand(2) * 14 - 7
                if np.linalg.norm(p) > s+link_length:
                    break
        obstacles.append((obs_t, p, s))
    
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    # geom2instnum = {id(g): i for i, (_, g) in enumerate(fcl_obs)}

    cfgs = torch.rand((num_points, robot.dof), dtype=torch.float32)
    cfgs = cfgs * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
    if label_type == 'binary':
        labels = torch.zeros(num_points, 1, dtype=torch.float)
        dists = torch.zeros(num_points, 1, dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        obs_managers[0].registerObjects(fcl_collision_obj)
        obs_managers[0].setup()
    elif label_type == 'instance':
        labels = torch.zeros(num_points, len(obstacles), dtype=torch.float)
        dists = torch.zeros(num_points, len(obstacles), dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in fcl_obs]
        for mng, cobj in zip(obs_managers, fcl_collision_obj):
            mng.registerObjects([cobj])
    elif label_type == 'class':
        labels = torch.zeros(num_points, num_class, dtype=torch.float)
        dists = torch.zeros(num_points, num_class, dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in range(num_class)]
        obj_by_cls = [[] for _ in range(num_class)]
        for obj in fcl_obs:
            obj_by_cls[obj.category].append(obj.cobj)
        for mng, obj_group in zip(obs_managers, obj_by_cls):
            mng.registerObjects(obj_group)
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    for mng in obs_managers:
        mng.setup()
    req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
    
    times = []
    st = time()
    for i, cfg in enumerate(cfgs):
        st1 = time()
        robot.update_polygons(cfg)
        robot_manager.update()
        assert len(robot_manager.getObjects()) == robot.dof
        for cat, obs_mng in enumerate(obs_managers):
            rdata = fcl.CollisionData(request = req)
            robot_manager.collide(obs_mng, rdata, fcl.defaultCollisionCallback)
            in_collision = rdata.result.is_collision
            ddata = fcl.DistanceData()
            robot_manager.distance(obs_mng, ddata, fcl.defaultDistanceCallback)
            depths = torch.FloatTensor([c.penetration_depth for c in rdata.result.contacts])

            labels[i, cat] = 1 if in_collision else -1
            dists[i, cat] = depths.abs().max() if in_collision else -ddata.result.min_distance
        end1 = time()
        times.append(end1-st1)
    end = time()
    times = np.array(times)
    print('std: {}, mean {}, avg {}'.format(times.std(), times.mean(), (end-st)/len(cfgs)))
    
    in_collision = (labels == 1).sum(1) > 0
    if label_type == 'binary':
        labels = labels.squeeze_(1)
        dists = dists.squeeze_(1)
    print('env_id {}, {} collisons, {} free'.format(
        env_id, torch.sum(in_collision==1), torch.sum(in_collision==0)))
    if torch.sum(in_collision==1) == 0:
        print('0 Collision. You may want to regenerate env {}obs{}'.format(obs_num, env_id))

    dataset = {
        'data': cfgs, 'label': labels, 'dist': dists, 'obs': obstacles, 
        'robot': robot.__class__, 'rparam': [robot.link_length, robot.link_width, robot.dof, ]
    }
    torch.save(dataset, os.path.join(
        folder, '2d_{}dof_{}obs_{}_{}.pt'.format(robot.dof, obs_num, label_type, env_id)))

    if vis:
        create_plots(robot, obstacles, cfg=None)
        plt.savefig(os.path.join(
            folder, '2d_{}dof_{}obs_{}_{}.png'.format(robot.dof, obs_num, label_type, env_id)))
        plt.close()
    
    return 
        


if __name__ == "__main__":
    batch_name = 'exp1'

    label_type = 'binary' #[instance, class, binary]
    DOF, link_length, seed = 7, 1, 1919
    # DOF, link_length, seed = 3, 2, 1918
    # DOF, link_length, seed = 2, 3.5, 1917
    
    width = 0.3
    robot = RevolutePlanarRobot(link_length, width, DOF)

    folder_name = os.path.join('data', '2d_{}dof_{}'.format(robot.dof, batch_name))
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

    num_envs_per_num_obs = 10
    env_id = 0
    seq_num_obs = [1]#[1, 2, 5, 10, 20]
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
            generate_one(robot, num_obs, folder_name, num_points=8000, env_id='{:02d}'.format(env_id), vis=True)
