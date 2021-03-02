import os
from diffco import DiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import rand, randint
import torch
from diffco.model import RigidBody
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
import matplotlib as mpl
from diffco import utils
from diffco.Obstacles import FCLObstacle
from time import time
import trimesh

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

def generate_one(robot, obs_num, folder, label_type='binary', num_class=None, num_points=8000, env_id='', vis=True):
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
    print('env_id {}, {} collisions, {} free'.format(
        env_id, torch.sum(in_collision==1), torch.sum(in_collision==0)))
    if torch.sum(in_collision==1) == 0:
        print('0 Collision. You may want to regenerate env {}obs{}'.format(obs_num, env_id))

    dataset = {
        'data': cfgs, 'label': labels, 'dist': dists, 'obs': obstacles, 
        'robot': robot.__class__, 'rparam': [robot.parts]
    }
    torch.save(dataset, os.path.join(
        folder, 'se3_{}obs_{}_{}.pt'.format(obs_num, label_type, env_id)))

    if vis:
        create_plots(robot, obstacles, cfg=cfgs[:100], label=labels[:100] )# torch.FloatTensor([2.5, -5, np.pi/6]))
        plt.savefig(os.path.join(
            folder, 'se3_{}obs_{}_{}.png'.format(obs_num, label_type, env_id)))
        plt.close()
    
    return

def generate_home(robot, obs_model_path, folder, label_type='binary', num_class=None, num_points=8000, env_id='', vis=True, transform=None):
    # Only using label_type = binary 
    mesh = trimesh.load(obs_model_path, force='mesh')
    if transform is not None:
        mesh.apply_transform(transform)
    bvh_mesh = trimesh.collision.mesh_to_BVH(mesh)
    bounds = mesh.bounds
    robot.limits[:3] = 1.3*torch.from_numpy(bounds.copy()).T # setting limits to room dimensions
    fcl_obs = [FCLObstacle('mesh', [0, 0, 0], geom=bvh_mesh)]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

    cfgs = torch.rand((num_points, robot.dof), dtype=torch.float32)
    cfgs = cfgs * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
    if label_type == 'binary':
        labels = torch.zeros(num_points, 1, dtype=torch.float)
        dists = torch.zeros(num_points, 1, dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        obs_managers[0].registerObjects(fcl_collision_obj)
    elif label_type == 'instance':
        labels = torch.zeros(num_points, len(fcl_obs), dtype=torch.float)
        dists = torch.zeros(num_points, len(fcl_obs), dtype=torch.float)
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
    print('env_id {}, {} collisions, {} free'.format(
        env_id, torch.sum(in_collision==1), torch.sum(in_collision==0)))
    if torch.sum(in_collision==1) == 0:
        print('0 Collision. You may want to regenerate env {}'.format(env_id))

    dataset = {
        'data': cfgs, 'label': labels, 'dist': dists, 'obs': obs_model_path, 
        'robot': robot.__class__, 'rparam': [robot.body_path, robot.keypoints, robot.limits]
    }
    torch.save(dataset, os.path.join(
        folder, 'se3_{}_{}.pt'.format(label_type, env_id)))


if __name__ == "__main__":
    batch_name = 'exp1'

    label_type = 'binary' #[instance, class, binary]
    seed = 1917

    transformation_for_home_environment = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    # type, offset, dimensions
    robot = RigidBody('data/Home_robot.dae', transform=transformation_for_home_environment)

    folder_name = os.path.join('data', 'se3_{}'.format(batch_name))
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

    generate_home(robot, 'data/Home_env.dae', folder_name, num_points=8000, env_id='home', 
        transform=transformation_for_home_environment)