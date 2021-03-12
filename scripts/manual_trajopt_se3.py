from pickle import TRUE
import sys
import json
from diffco import DiffCo, MultiDiffCo, DiffCoBeta
from diffco import kernel
from diffco.FCLChecker import FCLChecker
from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib import animation
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from diffco import utils
from diffco.Obstacles import FCLObstacle
from time import time
from scipy.optimize import minimize
import trimesh
from tqdm import trange
from diffco.utils import save_ompl_path
import fcl
from scipy.spatial.transform import Rotation

def original_traj_optimize(robot, dist_est, start_cfg, target_cfg, history=False):
    # There is a slightly different version in speed_compare.py,
    # which allows using SLSQP instead of Adam, allows
    # inputting an initial solution other than straight line,
    # and is better modularly written.
    # That one with SLSQP is more recommended to use.
    N_WAYPOINTS = 20
    NUM_RE_TRIALS = 10
    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 10
    collision_weight = 10
    safety_margin = torch.FloatTensor([-12, -1.2])#([-8.0, -0.8]) #
    lr = 5e-1
    seed = 1234
    torch.manual_seed(seed)

    lowest_cost_solution = None
    lowest_cost = np.inf
    lowest_cost_trial = None
    lowest_cost_step = None
    best_valid_solution = None
    best_valid_cost = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []

    found = False
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)

        for step in range(UPDATE_STEPS):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-0.3**2, min=0).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data[:, 2] = utils.wrap2pi(p.data[:, 2])
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_cost:
                lowest_cost = loss.data.numpy()
                lowest_cost_solution = p.data.clone()
                lowest_cost_step = step
                lowest_cost_trial = trial_time
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_cost:
                    best_valid_cost = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
                print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    diff.item(), dif_weight,
                    loss.item()))
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    if not found:
        print('Did not find a valid solution after {} trials!\
            Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_cost_solution
        solution_step = lowest_cost_step
        solution_trial = lowest_cost_trial
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    return solution, path_history, solution_trial, solution_step

def adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    history = options['history']

    dif_weight = 1 # This should NOT be changed
    max_move_weight = 10 # 100000 for centimeter
    collision_weight = 30 # 100000
    joint_limit_weight = 10 # 100000
    safety_margin = options['safety_margin']
    max_speed = options['max_speed']
    lr = 0.1
    seed = options['seed']
    torch.manual_seed(seed)

    lowest_loss_solution = None
    lowest_loss = np.inf
    lowest_loss_obj = np.inf
    lowest_loss_trial = None
    lowest_loss_step = None
    best_valid_solution = None
    best_valid_obj = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []
    cnt_check = 0

    found = False
    start_t = time()
    for trial_time in trange(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS)).double()
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        # p_pos = init_path[:, :3].requires_grad_(True)
        # p_ang = init_path[:, 3:].requires_grad_(True)
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        # opt = torch.optim.SGD([p], lr=lr)
        # opt = torch.optim.Adam([p_pos], lr=lr)
        # opt2 = torch.optim.Adam([p_ang], lr=lr)

        for step in range(MAXITER):
            opt.zero_grad()
            # opt2.zero_grad()
            # p = torch.cat([p_pos, p_ang], dim=1)
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            cnt_check += len(p) # Counting collision checks
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-max_speed**2, min=0).sum() #! change the max_move cost to environment
            joint_limit_cost = (
                torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            constraint_loss = collision_weight * collision_score\
                + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            # p.grad[:, [0,1,3,4,5]] = 0.0
            # p_pos.grad[[0, -1]] = 0.0
            # p_ang.grad[[0, -1]] = 0.0
            opt.step()
            # opt2.step()
            p.data[:, 3:] = utils.wrap2pi(p.data[:, 3:]) #! Wrap around specific angular dimensions
            # p_ang.data[:] = utils.wrap2pi(p_ang.data[:])
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_loss:
                lowest_loss = loss.data.numpy()
                lowest_loss_solution = p.data.clone()
                lowest_loss_step = step
                lowest_loss_trial = trial_time
                lowest_loss_obj = objective_loss.data.numpy()
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_obj:
                    best_valid_obj = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            if constraint_loss <= 1e-2 or step % (MAXITER/5) == 0 or step == MAXITER-1:
                print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, joint_limit={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    joint_limit_cost.item(), joint_limit_weight,
                    diff.item(), dif_weight,
                    loss.item()))
            if constraint_loss <= 1e-2 and torch.norm(p.grad) < 1e-4:
            # if constraint_loss <= 1e-2 and torch.norm(p_pos.grad) < 1e-4 and torch.norm(p_ang.grad) < 1e-4:
                break
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    end_t = time()
    if not found:
        # print('Did not find a valid solution after {} trials!\
            # Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_loss_solution
        solution_step = lowest_loss_step
        solution_trial = lowest_loss_trial
        solution_obj = lowest_loss_obj
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
        solution_obj = best_valid_obj
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': solution_obj.item(),
        'time': end_t - start_t,
        'success': found,
        'seed': seed,
        'solution': solution.numpy().tolist()
    }
    control_points = robot.fkine(solution)
    print(torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-max_speed**2, min=0).data.numpy())
    return rec

def givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    safety_margin = options['safety_margin']
    max_speed = options['max_speed']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check, obj, max_move_cost, collision_cost, joint_limit_cost, call_cnt
    global var_p_max_move, var_p_collision, var_p_limit, var_p_cost
    global latest_p_max_move, latest_p_collision, latest_p_limit, latest_p_cost
    cnt_check = 0
    call_cnt = 0

    def pre_process(p):
        global var_p
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        p[:, 3:] = utils.wrap2pi(p[:, 3:])
        var_p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0).requires_grad_(True)
        return var_p

    def con_max_move(p):
        global max_move_cost, var_p_max_move, latest_p_max_move
        var_p_max_move = pre_process(p)
        latest_p_max_move = var_p_max_move.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_max_move)
        max_move_cost = -torch.clamp_((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-max_speed**2, min=0).sum()
        return max_move_cost.data.numpy()
    def grad_con_max_move(p):
        if all(p == latest_p_max_move):
            pass
        else:
            con_max_move(p)
            # print(ValueError('p is not the same as the lastest passed p'))
        var_p_max_move.grad = None
        max_move_cost.backward()
        if var_p_max_move.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        return var_p_max_move.grad[1:-1].numpy().reshape(-1)

    def con_collision_free(p):
        global cnt_check, collision_cost, var_p_collision, latest_p_collision
        var_p_collision = pre_process(p)
        latest_p_collision = var_p_collision.data[1:-1].numpy().reshape(-1)
        cnt_check += len(p)
        collision_cost = torch.sum(-torch.clamp_(dist_est(var_p_collision[1:-1])-safety_margin, min=0))
        return collision_cost.data.numpy()
    def grad_con_collision_free(p):
        if all(p == latest_p_collision):
            pass
        else:
            con_collision_free(p)
            # print(ValueError('p is not the same as the lastest passed p'))
        var_p_collision.grad = None
        collision_cost.backward()
        if var_p_collision.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        return var_p_collision.grad[1:-1].numpy().reshape(-1)

    def con_joint_limit(p):
        global joint_limit_cost, var_p_limit, latest_p_limit
        var_p_limit = pre_process(p)
        latest_p_limit = var_p_limit.data[1:-1].numpy().reshape(-1)
        joint_limit_cost = -torch.sum(torch.clamp_(robot.limits[:, 0]-var_p_limit, min=0)\
             + torch.clamp_(var_p_limit-robot.limits[:, 1], min=0))
        return joint_limit_cost.data.numpy()
    def grad_con_joint_limit(p):
        if all(p == latest_p_limit):
            pass
        else:
            con_joint_limit(p)
            # print(ValueError('p is not the same as the lastest passed p'))
            # raise ValueError('p is not the same as the lastest passed p')
        var_p_collision.grad = None
        joint_limit_cost.backward()
        if var_p_collision.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        return var_p_collision.grad[1:-1].numpy().reshape(-1)

    def cost(p):
        global obj, var_p_cost, latest_p_cost
        var_p_cost = pre_process(p)
        latest_p_cost = var_p_cost.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_cost)
        obj = (control_points[1:]-control_points[:-1]).pow(2).sum()
        return obj.data.numpy()
    def grad_cost(p):
        if np.allclose(p, latest_p_cost):
            pass
        else:
            cost(p)
            # print(p, latest_p_cost, np.linalg.norm(p-latest_p_cost))
            # print(ValueError('p is not the same as the lastest passed p'))
        var_p_cost.grad = None
        obj.backward()
        if var_p_cost.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        return var_p_cost.grad[1:-1].numpy().reshape(-1)

    start_t = time()
    success = False
    res = None
    for trial_time in trange(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            # init_path = (torch.rand(N_WAYPOINTS, robot.dof, dtype=torch.float64))*np.pi*2-np.pi
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        tmp_res = minimize(cost, init_path[1:-1].reshape(-1).numpy(), jac=grad_cost,
            method='slsqp',
            constraints=[
                {'fun': con_max_move, 'type': 'ineq', 'jac': grad_con_max_move},
                {'fun': con_collision_free, 'type': 'ineq', 'jac': grad_con_collision_free},
                {'fun': con_joint_limit, 'type': 'ineq', 'jac': grad_con_joint_limit}
            ],
            options={'maxiter': MAXITER, 'disp': True, 'verbose': 3})
        if tmp_res.success:
            success = True
            break
        elif res is None or tmp_res.fun < res.fun:
            res = tmp_res
    end_t = time()
    res.x = res.x.reshape([-1, robot.dof])
    res.x = pre_process(res.x)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': res.fun.item(),
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': res.x.data.numpy().tolist()
    }
    return rec

def animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None):
    global opt, start_frame, cnt_down
    FPS = 15

    def init():
        if robot.dof == 2:
            return [link_plot, joint_plot, eff_plot] + cfg_path_plots
        else:
            return link_plot, joint_plot, eff_plot

    def update_traj(i):
        if robot.dof == 2:
            for cfg_path in cfg_path_plots:
                cfg_path.set_data(path_history[i][:, 0], path_history[i][:, 1])
            return cfg_path_plots
        else:
            return link_plot, joint_plot, eff_plot
        
    def plot_robot(q):
        robot_points = robot.fkine(q)[0]
        robot_points = torch.cat([torch.zeros(1, 2), robot_points])
        link_plot.set_data(robot_points[:, 0], robot_points[:, 1])
        joint_plot.set_data(robot_points[:-1, 0], robot_points[:-1, 1])
        eff_plot.set_data(robot_points[-1:, 0], robot_points[-1:, 1])

        return link_plot, joint_plot, eff_plot

    def move_robot(i):
        i = i if i < len(p) else len(p)-1
        with torch.no_grad():
            ret = plot_robot(p[i])
        if robot.dof == 2:
            return list(ret) + cfg_path_plots
        else:
            return ret

    if robot.dof == 2 and path_history:
        UPDATE_STEPS = len(path_history)
        f = lambda i: update_traj(i) if i < UPDATE_STEPS else move_robot(i-UPDATE_STEPS)
        num_frames = UPDATE_STEPS + len(p)
    else:
        f = move_robot
        num_frames=len(p)
    ani = animation.FuncAnimation(
        fig, func=f, 
        frames=num_frames, interval=1000./FPS, 
        blit=True, init_func=init, repeat=False)
    
    if save_dir:
        ani.save(save_dir, fps=FPS)
    else:
        # plt.axis('equal')
        # plt.axis('tight')
        plt.show()

# A function that controls the style of visualization.
def create_plots(robot, obstacle_meshes, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "sans-serif",
    #     "font.sans-serif": ["Helvetica"]})

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks([-4, 0, 4])
    # ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=18)
    for obs_mesh in obstacle_meshes:
        cat = 1
        ax.plot_trisurf(obs_mesh.vertices[:, 0], obs_mesh.vertices[:,1], triangles=obs_mesh.faces, Z=obs_mesh.vertices[:,2])
    # Placeholder of the robot plot

    
    # trans = ax.transData.transform
    # lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    # link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    # joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    # eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    return fig, ax
    
def single_plot(robot, path, fig, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    from copy import copy
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    points_traj = robot.fkine(path)
    points_traj = torch.cat([torch.zeros(len(path), 1, 2), points_traj], dim=1)
    traj_alpha = 0.3
    ends_alpha = 0.5
    robot_color = 'grey'
    start_robot_color = 'green'
    end_robot_color = 'orange'
    
    robot_traj_patches = []
    for q in path:
        
        robot_traj_patches.append(robot_patch)

    # lw = link_plot.get_lw()
    # robot_patches = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=traj_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    # joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    # eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]

    for i in [0, -1]:
        for patch in robot_traj_patches[i]:
            patch.set_alpha(ends_alpha)
            # patch.set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
            # joint_traj[i].set_alpha(ends_alpha)
            # eff_traj[i].set_alpha(ends_alpha)
    for patch in robot_traj_patches[0]:
        patch.set_color(start_robot_color)
    for patch in robot_traj_patches[-1]:
        patch.set_color(end_robot_color)

    # def divide(p): # divide the path into several segments that obeys the wrapping around rule [TODO]
    #     diff = torch.abs(p[:-1]-p[1:])
    #     div_idx = torch.where(diff.max(1) > np.pi)
    #     div_idx = torch.cat([-1, div_idx])
    #     segments = []
    #     for i in range(len(div_idx)-1):
    #         segments.append(p[div_idx[i]+1:div_idx[i+1]+1])
    #     segments.append(p[div_idx[-1]+1:])
    #     for i in range(len(segments)-1):
    #         if torch.sum(torch.abs(segments[i]) > np.pi) == 2:


    # for cfg_path in cfg_path_plots:
    #     cfg_path.set_data(p[:, 0], p[:, 1])

    # ---------Just for making the opening figure------------
    # For a better way wrap around trajectories in angular space
    # see active.py

    # segments = [p[:-3], p[-3:]]
    # d1 = segments[0][-1, 0]-(-np.pi)
    # d2 = np.pi - segments[1][0, 0]
    # dh = segments[1][0, 1] - segments[0][-1, 1]
    # intery = segments[0][-1, 1] + dh/(d1+d2)*d1
    # segments[0] = torch.cat([segments[0], torch.FloatTensor([[-np.pi, intery]])])
    # segments[1] = torch.cat([torch.FloatTensor([[np.pi, intery]]), segments[1]])
    # for cfg_path in cfg_path_plots:
    #     for seg in segments:
    #         cfg_path.axes.plot(seg[:, 0], seg[:, 1], '-o', c='olivedrab', alpha=0.5, markersize=3)
    # ---------------------------------------------

def escape(robot, dist_est, start_cfg):
    N_WAYPOINTS = 200
    safety_margin = -0.3 # -30 for centimeter,  -0.3 for meter
    lr = 0.05
    path_history = []
    init_path = start_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr) # Adam is bad for rot and trans with huge unit range difference
    # opt = torch.optim.SGD([p], lr=lr)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())

        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin
        loss = collision_score
        loss.backward()
        # p.grad[3:] = 0.0
        opt.step()
        print('Collision score: ', collision_score.data.item()+safety_margin)
        print(p.grad.data)
        p.data[3:] = utils.wrap2pi(p.data[3:])
        if collision_score <= 1e-4:
            break
    return torch.stack(path_history, dim=0)

# Commented out lines include convenient code for debugging purposes
def main():
    env_name = 'binary_home_smalldesk_meter'

    dataset = torch.load('data/se3_{}.pt'.format(env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].double()
    dists = dataset['dist'].double()
    # obstacles = dataset['obs']
    obstacle_paths = dataset['obs']
    # obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    print(obstacle_paths)
    robot = dataset['robot'](*dataset['rparam'])
    robot.keypoints = robot.keypoints.double()
    robot.limits /= 1.3
    train_num = 7000 #int(len(cfgs) * 0.9)
    fkine = robot.fkine
    Epsilon = 1 #0.01
    checker = DiffCoBeta(obstacle_paths, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1,\
        rbf_kernel=kernel.Polyharmonic(3, Epsilon))
    checker.train(cfgs[:train_num], dists[:train_num], fkine=fkine, max_iteration=train_num, n_left_out_points=3000, keep_all=True)
    # checker = DiffCo(obstacle_paths, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(100)), beta=1.0)
    # checker = DiffCo(obstacle_paths, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(0.0001)), beta=1.0) # for centimeter environment
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])
    # import pickle
    # with open('results/checker_se3_{}.p'.format(env_name), 'wb') as f:
    #     pickle.dump(checker, f)
    #     print('checker saved: {}'.format(f.name))
    #==== The following can be used alone to save training time in debugging
    # with open('results/checker_se3_{}.p'.format(env_name), 'rb') as f:
    #     checker = pickle.load(f)
    #     print('checker loaded: {}'.format(f.name))

    fitting_target = 'label' # {label, dist, hypo}
    # checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine)#, reg=0.09) # epsilon=Epsilon,
    dist_est = checker.rbf_score
    # checker.fit_full_poly(epsilon=Epsilon, k=1, target=fitting_target, fkine=fkine) #, lmbd=10)
    # dist_est = checker.poly_score
    min_score = dist_est(cfgs[train_num:]).min()
    safety_bias = -0.3 #-1.1 for diffco with label #-0.3 for diffcobeta with distance
    print('MIN_SCORE = {:.6f}'.format(min_score))

    # Check DiffCo test ACC
    test_preds = (dist_est(cfgs[train_num:]).view(-1)-safety_bias > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    # assert(test_acc > 0.8)
    from matplotlib import pyplot as plt
    plt.scatter(dists[train_num:].view(-1), dist_est(cfgs[train_num:]).view(-1))
    plt.show()

    # return # DEBUGGING

    cfg_path_plots = []
    # if robot.dof > 2:
    #     fig, ax, = create_plots(robot, obstacles, dist_est, checker)
    # elif robot.dof == 2:
    #     fig, ax, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)

    

    # Pick a pair of collision-free configurations
    # torch.manual_seed(1213)
    # free_cfgs = cfgs[labels == -1]
    # indices = torch.randint(0, len(free_cfgs), (2, ))
    # while indices[0] == indices[1]:
    #     indices = torch.randint(0, len(free_cfgs), (2, ))
    # start_cfg = free_cfgs[indices[0]] # torch.zeros(robot.dof, dtype=torch.float32) # 
    # target_cfg = free_cfgs[indices[1]] # torch.zeros(robot.dof, dtype=torch.float32) # 

    # Pick one in-collision configuration
    # torch.manual_seed(1237)
    # collided_cfgs = cfgs[dists > dists.max()/2] # cfgs[labels == 1]
    # indice = torch.randint(0, len(collided_cfgs), (1,))[0]
    # start_cfg = collided_cfgs[indice]
    # print("Start from: ", start_cfg)

    # use star and goals from OMPL app
    ompl_start = list(map(float, '18.0 -110.0 67.19 0.0 0.0 0.0 1.0'.split(' ')))
    start_cfg_quat = torch.FloatTensor(ompl_start)
    start_cfg = torch.zeros(6)
    start_cfg[:3] = start_cfg_quat[:3] /100*2
    start_cfg[3:] = torch.from_numpy(Rotation.from_quat(start_cfg_quat[3:]).as_euler('xyz').astype(np.float32))
    ompl_target = list(map(float, '-142.0 -110.0 68.19 0.0 0.0 0.0 1.0'.split(' ')))
    target_cfg_quat = torch.FloatTensor(ompl_target)
    target_cfg = torch.zeros(6)
    target_cfg[:3] = target_cfg_quat[:3] /100*2
    target_cfg[3:] = torch.from_numpy(Rotation.from_quat(target_cfg_quat[3:]).as_euler('xyz').astype(np.float32))

    # This is for doing traj optimization
    # p, path_history, num_trial, num_step = traj_optimize(
    #     robot, dist_est, start_cfg, target_cfg, history=True)
    # with open('results/path_se3_{}.json'.format(env_name), 'w') as f:
    #     json.dump(
    #         {
    #             'path': p.data.numpy().tolist(), 
    #             'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
    #             'trial': num_trial,
    #             'step': num_step
    #         },
    #         f, indent=1)
    #     print('Plan recorded in {}'.format(f.name))

    # this is for doing traj opt using OMPL solution as initialization. You can also use SLSQP to optimize here
    # init_path = []
    # with open('results/path.txt', 'r') as f:
    #     for line in f:
    #         init_path.append([float(x) for x in line.strip().split(' ')])
    # init_path_np = np.array(init_path)
    # init_path = np.zeros((len(init_path_np), 6))
    # init_path[:, :3] = init_path_np[:, :3] /100 *2
    # init_path[:, 3:] = Rotation.from_quat(init_path_np[:, 3:]).as_euler('xyz')
    # init_path = torch.from_numpy(init_path).double()
    # start_cfg, target_cfg = init_path[[0, -1]]

    mesh = trimesh.load(obstacle_paths, force='mesh')
    transformation_for_home_environment = robot.transform * 2 #np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 0, 1]
    # ]) /100
    # transformation_for_home_environment = np.eye(4)
    mesh.apply_transform(transformation_for_home_environment)
    mesh.visual.vertex_colors = trimesh.visual.interpolate(mesh.vertices[:, 1], color_map='viridis')
    bvh_mesh = trimesh.collision.mesh_to_BVH(mesh)
    fcl_obs = [FCLObstacle('mesh', [0, 0, 0], geom=bvh_mesh)]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
    obs_managers[0].registerObjects(fcl_collision_obj)
    robot_links = robot.update_polygons(start_cfg)
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    obs_managers[0].setup()
    fcl_checker = FCLChecker(obstacle_paths, robot, robot_manager, obs_managers)
    # # fcl_preds, fcl_dist = fcl_checker.predict(init_path, distance=True)
    # # print('FCL checking initial solution: {}'.format('All valid!' if torch.all(fcl_preds == -1) else \
    # #     '{} invalid waypoints.'.format(torch.sum(fcl_preds == 1).item())))
    # # fcl_preds, fcl_dist = fcl_checker.predict(cfgs[train_num:], distance=True)
    # # diffco_preds = dist_est(cfgs[train_num:]) # > 0)* 2 - 1
    # # diffco_preds = dist_est(init_path)# > 0)* 2 - 1
    # # print('{}/{} incorrect predictions.'.format((diffco_preds != fcl_preds).sum(), len(diffco_preds)))
    # # print('DiffCo checking initial solution: {}'.format('All valid!' if torch.all(diffco_preds == -1) else \
    # #     '{} invalid waypoints.'.format(torch.sum(diffco_preds == 1).item())))
    # # print(diffco_preds)
    # # from matplotlib import pyplot as plt
    # # # plt.plot(range(len(diffco_preds)), diffco_preds, label="diffco")
    # # # plt.plot(range(len(diffco_preds)), fcl_dist, label="fcl")
    # # plt.scatter(fcl_dist, diffco_preds)
    # # plt.axis('equal')
    # # plt.legend()
    # # plt.show()
    # # utils.view_se3_path(robot, mesh, init_path)
    # # return
    
    options = {
        'N_WAYPOINTS': 40,
        'NUM_RE_TRIALS': 3,
        'MAXITER': 500,
        'safety_margin': safety_bias, # max(1/5*min_score, -1),
        'max_speed': 0.5,
        'seed': 123456,
        'history': False,
    }
    # options['init_solution'] = init_path
    # options['N_WAYPOINTS'] = len(init_path)
    # rec = givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
    rec = adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
    p = torch.FloatTensor(rec['solution'])
    print('Succeeded!' if rec['success'] else 'Not successful')
    with open('results/path_se3_{}.json'.format(env_name), 'w') as f:
        json.dump(
            {
                'path': p.data.numpy().tolist(), 
            },
            f, indent=1)
        print('Plan recorded in {}'.format(f.name))
    save_ompl_path('results/ompl_se3_{}.txt'.format(env_name), p*torch.FloatTensor([100, 100, 100, 1, 1, 1]))
    fcl_preds = fcl_checker.predict(p, distance=False).view(-1)
    print('In collision:', (fcl_preds==1).sum())
    print('Collision free:', (fcl_preds==-1).sum())
    utils.view_se3_path(robot, mesh, p)

    ## This for doing the escaping-from-collision experiment
    # p = escape(robot, dist_est, start_cfg.double())
    # with open('results/esc_se3_{}.json'.format(env_name), 'w') as f:
    #     json.dump({'path': p.data.numpy().tolist(), },f, indent=1)
    #     print('Plan recorded in {}'.format(f.name))
    # with open('results/esc_se3_{}.json'.format(env_name), 'r') as f:
    #     path_dict = json.load(f)
    #     p = torch.FloatTensor(path_dict['path'])
    #     print('Esc plan loaded from {}'.format(f.name))
    # save_ompl_path('results/ompl_escape_se3_{}.txt'.format(env_name), p)
    # fcl_preds = fcl_checker.predict(p, distance=False).view(-1)
    # print('In collision:', (fcl_preds==1).sum())
    # print('Collision free:', (fcl_preds==-1).sum())
    # utils.view_se3_path(robot, mesh, p)

    ## This is for loading previously computed trajectory
    # with open('results/path_se3_{}.json'.format(env_name), 'r') as f:
    #     path_dict = json.load(f)
    #     p = torch.FloatTensor(path_dict['path'])
    #     path_history = [torch.FloatTensor(shot) for shot in path_dict['path_history']] #[p] #
    
    ## This produces an animation for the trajectory (not recommended)
    # vid_name = None #'results/maual_trajopt_se3_{}_fitting_{}_eps_{}_dif_{}_updates_{}_steps_{}.mp4'.format(
    #     # robot.dof, env_name, fitting_target, Epsilon, dif_weight, UPDATE_STEPS, N_STEPS)
    # if robot.dof == 2:
    #     animation_demo(
    #         robot, p, fig, link_plot, joint_plot, eff_plot, 
    #         cfg_path_plots=cfg_path_plots, path_history=path_history, save_dir=vid_name)
    # elif robot.dof == 7:
    #     animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, save_dir=vid_name)

    # (Recommended) This produces a single shot of the planned trajectory
    # single_plot(robot, p, fig, cfg_path_plots=cfg_path_plots, ax=ax)
    # plt.show()
    # plt.savefig('figs/path_se3_{}.png'.format(env_name), dpi=500)
    # plt.savefig('figs/se3_{}_adam_05'.format(env_name), dpi=500) #_equalmargin.png

    # plt.tight_layout()
    # plt.savefig('figs/opening_contourline.png', dpi=500, bbox_inches='tight')
    
def check_configuration_convention():
    p = torch.zeros((10, 6))
    p[:, :3] = torch.FloatTensor([18.0, -130.0, 67.19000244140625])
    deg2rad = np.pi/180.
    p[0, 3] = 30 * deg2rad
    p[0, 4] = 45 * deg2rad
    p[0, 5] = 70 * deg2rad
    p[-1, 3] = 60 * deg2rad
    p[-1, 4] = 80 * deg2rad
    p[-1, 5] = 120 * deg2rad
    p[:, 3:] = torch.from_numpy(np.linspace(p[0, 3:], p[-1, 3:], len(p)))

    print(p[:, 3:] / deg2rad)
    save_ompl_path('results/check_convention.txt', p)




if __name__ == "__main__":
    main()
    # check_configuration_convention()