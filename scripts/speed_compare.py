import sys
import json
import os
from os.path import basename, splitext, join, isdir
# sys.path.append('/home/yuheng/DiffCo/')
from diffco import DiffCo, MultiDiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from diffco import utils
from diffco.Obstacles import FCLObstacle
from scipy.optimize import minimize as fmin
from diffco.FCLChecker import FCLChecker
from time import time
from tqdm import tqdm

def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    if robot.dof > 2:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) #, projection='3d'
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*(num_class + 1)+0.5, 3 * num_class))
        gs = fig.add_gridspec(num_class, num_class+1)
        ax = fig.add_subplot(gs[:, :-1]) #sum([list(range(r*(num_class+1)+1, (r+1)*(num_class+1))) for r in range(num_class)], [])) #, projection='3d'
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2)).type(checker.gains.dtype)
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[cat, -1])

                # score_DiffCo = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains.reshape(len(checker.gains), -1)[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                c_ax.contour(xx, yy, score, levels=[0], linewidths=1, alpha=0.4, ) #-1.5, -0.75, 0, 0.3
                # fig.colorbar(color_mesh, ax=c_ax)
                # sparse_score = score[5:-5:10, 5:-5:10]
                # score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
                # score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
                # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
                # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
                # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
                # c_ax.quiver(xx[5:-5:10, 5:-5:10], yy[5:-5:10, 5:-5:10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
                # cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
                # c_ax.add_patch(cfg_point)
                cfg_path, = c_ax.plot([], [], '-o', c='orange', markersize=3)
                cfg_path_plots.append(cfg_path)

                c_ax.set_aspect('equal', adjustable='box')
                # c_ax.axis('equal')
                c_ax.set_xlim(-np.pi, np.pi)
                c_ax.set_ylim(-np.pi, np.pi)
                c_ax.set_xticks([-np.pi, 0, np.pi])
                c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
                c_ax.set_yticks([-np.pi, 0, np.pi])
                c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
                # c_ax.tick_params(direction='in', reset=True)
                # c_ax.tick_params(which='both', direction='out', length=6, width=2, colors='r',
                #    grid_color='r', grid_alpha=0.5)
            # c_ax.set_ticks('')

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
            color=cmaps[cat](0.5))) #path_effects=[path_effects.withSimplePatchShadow()], 
            # print((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    if robot.dof > 2:
        return fig, ax, link_plot, joint_plot, eff_plot
    elif robot.dof == 2:
        return fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots
    
def single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    from copy import copy
    from matplotlib.lines import Line2D
    points_traj = robot.fkine(p)
    points_traj = torch.cat([torch.zeros(len(p), 1, 2, dtype=points_traj.dtype), points_traj], dim=1)
    traj_alpha = 0.3
    ends_alpha = 0.5
    
    lw = link_plot.get_lw()
    link_traj = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=traj_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    # for link_plot, joint_plot, eff_plot, points in zip(link_traj, joint_traj, eff_traj, points_traj):
    #     link_plot.set_data(points[:, 0], points[:, 1])
    #     joint_plot.set_data(points[:-1, 0], points[:-1, 1])
    #     eff_plot.set_data(points[-1:, 0], points[-1:, 1])
    for i in [0, -1]:
        link_traj[i].set_alpha(ends_alpha)
        link_traj[i].set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
        joint_traj[i].set_alpha(ends_alpha)
        eff_traj[i].set_alpha(ends_alpha)
    link_traj[0].set_color('green')
    link_traj[-1].set_color('orange')
    # ax.add_artist(link_traj[2])

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


    for cfg_path in cfg_path_plots:
        cfg_path.set_data(p[:, 0], p[:, 1])

    # ---------Just for making one particular figure------------
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

    # plt.show()

def adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options): # history=False):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    history = options['history']

    dif_weight = 1 # This should NOT be changed
    max_move_weight = 10
    collision_weight = 10
    joint_limit_weight = 10
    safety_margin = options['safety_margin'] # torch.FloatTensor([-8.0, -0.8]) #
    lr = 5e-1
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
    for trial_time in range(NUM_RE_TRIALS):
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
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)

        for step in range(MAXITER):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            cnt_check += len(p) # Counting collision checks
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2, min=0).sum()
            joint_limit_cost = (
                torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum() # + (control_points[2:]-2*control_points[1:-1] + control_points[:-2]).pow(2).sum()
            # np.clip(1.5*float(i)/UPDATE_STEPS, 0, 1)**2 (float(i)/UPDATE_STEPS) * 
            # torch.clamp(utils.wrap2pi(p[1:]-p[:-1]).abs(), min=0.3).pow(2).sum()
            constraint_loss = collision_weight * collision_score\
                + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data = utils.wrap2pi(p.data)
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
            # if constraint_loss <= 1e-2 or step % (MAXITER/5) == 0 or step == MAXITER-1:
            #     print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
            #         trial_time, step, 
            #         collision_score.item(), collision_weight,
            #         max_move_cost.item(), max_move_weight,
            #         diff.item(), dif_weight,
            #         loss.item()))
            if constraint_loss <= 1e-2 and torch.norm(p.grad) < 1e-4:
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
        # 'cons_violation': None,
        'time': end_t - start_t,
        'success': found,
        'seed': seed,
        'solution': solution.numpy().tolist()
    }
    return rec

def givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS'] # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS'] # 10
    MAXITER = options['MAXITER'] # 200
    safety_margin = options['safety_margin']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check, obj, max_move_cost, collision_cost, joint_limit_cost, latest_p, opt, call_cnt
    global var_p_max_move, var_p_collision, var_p_limit, var_p_cost
    global latest_p_max_move, latest_p_collision, latest_p_limit, latest_p_cost
    cnt_check = 0
    call_cnt = 0

    def pre_process(p):
        global var_p, opt
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        p[:] = utils.wrap2pi(p)
        var_p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0).requires_grad_(True)
        # opt = torch.optim.Adam([var_p]) # just to use zero_grad
        return var_p

    def con_max_move(p):
        # global call_cnt
        # print('max_move {}'.format(call_cnt))
        # call_cnt += 1
        global max_move_cost, var_p_max_move, latest_p_max_move
        var_p_max_move = pre_process(p)
        latest_p_max_move = var_p_max_move.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_max_move)
        max_move_cost = -torch.clamp_((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2, min=0).sum()#\
            # -torch.clamp((utils.wrap2pi(var_p_max_move[1:]-var_p_max_move[:-1])).pow(2).sum(dim=1)-(15*np.pi/180)**2, min=0).sum() # DEBUG. No angle loss before.
        return max_move_cost.data.numpy()
    def grad_con_max_move(p):
        # global call_cnt
        # print('grad_max_move {}'.format(call_cnt))
        # call_cnt += 1
        if all(p == latest_p_max_move):
            # opt.zero_grad()
            var_p_max_move.grad = None
            max_move_cost.backward()
            if var_p_max_move.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_max_move.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def con_collision_free(p):
        # global call_cnt
        # print('collision {}'.format(call_cnt))
        # call_cnt += 1
        global cnt_check, collision_cost, var_p_collision, latest_p_collision
        var_p_collision = pre_process(p)
        latest_p_collision = var_p_collision.data[1:-1].numpy().reshape(-1)
        # print(torch.min(-checker(p)).numpy().dtype)
        cnt_check += len(p)
        collision_cost = torch.sum(-torch.clamp_(dist_est(var_p_collision[1:-1])-safety_margin, min=0))
        return collision_cost.data.numpy()
    def grad_con_collision_free(p):
        # global call_cnt
        # print('grad_collision {}'.format(call_cnt))
        # call_cnt += 1
        if all(p == latest_p_collision):
            # opt.zero_grad()
            var_p_collision.grad = None
            collision_cost.backward()
            if var_p_collision.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_collision.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def con_joint_limit(p):
        # global call_cnt
        # print('joint_limit {}'.format(call_cnt))
        # call_cnt += 1
        global joint_limit_cost, var_p_limit, latest_p_limit
        var_p_limit = pre_process(p)
        latest_p_limit = var_p_limit.data[1:-1].numpy().reshape(-1)
        # print(np.float(torch.all(robot.limits[:, 0] < p) and torch.all(p < robot.limits[:, 1])))
        joint_limit_cost = -torch.sum(torch.clamp_(robot.limits[:, 0]-var_p_limit, min=0)\
             + torch.clamp_(var_p_limit-robot.limits[:, 1], min=0))
        return joint_limit_cost.data.numpy()
    def grad_con_joint_limit(p):
        # global call_cnt
        # print('grad_joint_limit {}'.format(call_cnt))
        # call_cnt += 1
        if all(p == latest_p_limit):
            # opt.zero_grad()
            var_p_collision.grad = None
            joint_limit_cost.backward()
            if var_p_collision.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_collision.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def cost(p):
        global obj, var_p_cost, latest_p_cost
        var_p_cost = pre_process(p)
        latest_p_cost = var_p_cost.data[1:-1].numpy().reshape(-1)
        # p_tensor = torch.from_numpy(p).reshape([-1, 2])
        control_points = robot.fkine(var_p_cost)
        obj = (control_points[1:]-control_points[:-1]).pow(2).sum()
        return obj.data.numpy()
    def grad_cost(p):
        if np.allclose(p, latest_p_cost):
            # opt.zero_grad()
            var_p_cost.grad = None
            obj.backward()
            if var_p_cost.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_cost.grad[1:-1].numpy().reshape(-1)
        else:
            print(p, latest_p_cost, np.linalg.norm(p-latest_p_cost))
            raise ValueError('p is not the same as the lastest passed p')

    start_t = time()
    success = False
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof, dtype=torch.float64))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        # p = init_path.requires_grad_(True)
        res = fmin(cost, init_path[1:-1].reshape(-1).numpy(), jac=grad_cost,
            # method='trust-constr',
            # method='Nelder-Mead',
            method='slsqp',
            constraints=[
                {'fun': con_max_move, 'type': 'ineq', 'jac': grad_con_max_move},
                {'fun': con_collision_free, 'type': 'ineq', 'jac': grad_con_collision_free},
                {'fun': con_joint_limit, 'type': 'ineq', 'jac': grad_con_joint_limit}
            ],
            options={'maxiter': MAXITER, 'disp': False})
        # print(res)
        if res.success:
            success = True
            break
    end_t = time()
    res.x = res.x.reshape([-1, robot.dof])
    # print('Collision constraint: ', con_collision_free(res.x))
    res.x = pre_process(res.x)
    # print(res)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': res.fun.item(),
        # 'cons_violation': None,
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': res.x.data.numpy().tolist()
    }
    return rec

def gradient_free_traj_optimize(robot, checker, start_cfg, target_cfg, options=None): #, history=False):
    N_WAYPOINTS = options['N_WAYPOINTS']
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']
    MAXITER = options['MAXITER']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check
    cnt_check = 0

    def pre_process(p):
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        p[:] = utils.wrap2pi(p)
        p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0)
        return p

    def con_max_move(p):
        p = pre_process(p)
        control_points = robot.fkine(p)
        return -torch.clamp_((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2, min=0).sum().numpy()
    def con_collision_free(p):
        global cnt_check
        p = pre_process(p)
        # print(torch.min(-checker(p)).numpy().dtype)
        cnt_check += len(p)
        return torch.sum(-torch.clamp_(checker(p), min=0)).numpy()
    def con_joint_limit(p):
        p = pre_process(p)
        # print(np.float(torch.all(robot.limits[:, 0] < p) and torch.all(p < robot.limits[:, 1])))
        return -torch.sum(torch.clamp_(robot.limits[:, 0]-p, min=0) + torch.clamp_(p-robot.limits[:, 1], min=0)).numpy()
    def cost(p):
        p_tensor = pre_process(p)
        # p_tensor = torch.from_numpy(p).reshape([-1, 2])
        control_points = robot.fkine(p_tensor)
        diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
        return diff.numpy()

    # def cost_test(x):
    #     return (x**2).sum()
    # def con_test(x):
    #     return (x**3).sum() - 10
    # res = fmin(cost_test, torch.FloatTensor([0.1]), constraints=[
    #     {'fun': con_test, 'type': 'ineq'}],
    #     options={'maxiter': 1000})
    start_t = time()
    success = False
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
            else:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof, dtype=torch.float64))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        # p = init_path.requires_grad_(True)
        res = fmin(cost, init_path[1:-1].reshape(-1).numpy(), 
            # method='trust-constr',
            # method='Nelder-Mead',
            constraints=[
                {'fun': con_max_move, 'type': 'ineq'},
                {'fun': con_collision_free, 'type': 'ineq'},
                {'fun': con_joint_limit, 'type': 'ineq'}
            ],
            options={'maxiter': MAXITER, 'disp': False})
        # print(res)
        if res.success:
            success = True
            break
    end_t = time()
    res.x = res.x.reshape([-1, robot.dof])
    # print('Collision constraint: ', con_collision_free(res.x))
    res.x = utils.wrap2pi(pre_process(res.x))
    # print(res)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': res.fun.item(),
        # 'cons_violation': None,
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': res.x.numpy().tolist()
    }
    return rec

class ExpConfigs(object):
    # A simple class to store experiment configurations. 
    # arguments not mentioned in args will be in their default values
    def __init__(self, args: dict):
        # default values
        self.load_exp = None
        self.include_validate_time = True
        self.use_previous_solution = True
        self.validate_density = 1

        for k, v in args.items():
            assert hasattr(self, k)
            setattr(self, k, v)

def test_one_env(env_name, method, folder, args: ExpConfigs, prev_rec={}):
    assert (args.use_previous_solution and prev_rec != {}) or \
        ((not args.use_previous_solution) and prev_rec == {}), \
            "args.use_previous_solution does not match the existence of prev_rec."

    print(env_name, method, 'Begin')
    # Prepare distance estimator ====================
    dataset = torch.load('{}/{}.pt'.format(folder, env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].double() #.max(1).values
    dists = dataset['dist'].double() #.reshape(-1, 1) #.max(1).values
    obstacles = dataset['obs']
    # obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine

    train_t = time()
    checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check DiffCo test ACC
    # test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    # test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    # test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    # test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    # print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    # if test_acc < 0.9:
    #     print('test acc is only {}'.format(test_acc))

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 1 #0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine)#, reg=0.09) # epsilon=Epsilon,
    # checker.fit_rbf(kernel_func=kernel.MultiQuadratic(Epsilon), target=fitting_target, fkine=fkine)
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine)#, lmbd=80)
    # ========================
    # ONLY for additional training timing exp
    # fcl_obs = [FCLObstacle(*param) for param in obstacles]
    # fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    # obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
    # obs_managers[0].registerObjects(fcl_collision_obj)
    # obs_managers[0].setup()
    # robot_links = robot.update_polygons(cfgs[0])
    # robot_manager = fcl.DynamicAABBTreeCollisionManager()
    # robot_manager.registerObjects(robot_links)
    # robot_manager.setup()
    # for mng in obs_managers:
    #     mng.setup()
    # gt_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)
    # gt_checker.predict(cfgs[:train_num], distance=False)
    # return time() - train_t
    # END ========================
    dist_est = checker.rbf_score
    # dist_est = checker.poly_score
    min_score = dist_est(cfgs[train_num:]).min().item()
    # print('MIN_SCORE = {:.6f}'.format(min_score))
    # ==============================================

    # FCL checker =====================
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

    label_type = 'binary'
    num_class = 1

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
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    for mng in obs_managers:
        mng.setup()
    fcl_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)
    # =================================

    options = {
        'N_WAYPOINTS': 20,
        'NUM_RE_TRIALS': 3,
        'MAXITER': 200,
        'safety_margin': max(1/5*min_score, -0.5),
        'seed': 19961221,
        'history': False
    }

    repair_options = {
        'N_WAYPOINTS': 20,
        'NUM_RE_TRIALS': 1, # just one trial
        'MAXITER': 200,
        'seed': 19961221, # actually not used due to only one trial
        'history': False,
    }

    test_rec = {
        'start_cfg': [],
        'target_cfg': [],
        'cnt_check': [],
        'repair_cnt_check': [],
        'cost': [],
        'repair_cost': [],
        # 'cons_violation': [],
        'time': [],
        'val_time': [],
        'repair_time': [],
        'success': [],
        'repair_success': [],
        'seed': [],
        'solution': [],
        'repair_solution': [],
    }
    
    with open('{}/{}_testcfgs.json'.format(folder, env_name), 'r') as f:
        test_cfg_dataset = json.load(f)
        s_cfgs = torch.FloatTensor(test_cfg_dataset['start_cfgs'])[:10]
        t_cfgs = torch.FloatTensor(test_cfg_dataset['target_cfgs'])[:10]
        assert env_name == test_cfg_dataset['env_name']
    if prev_rec != {}:
        s_cfgs = torch.FloatTensor(test_cfg_dataset['start_cfgs'])[:len(prev_rec['success'])]
        t_cfgs = torch.FloatTensor(test_cfg_dataset['target_cfgs'])[:len(prev_rec['success'])]
        assert len(s_cfgs) == len(prev_rec['success']) and len(t_cfgs) == len(prev_rec['success'])
        rec_s_cfgs = torch.FloatTensor(prev_rec['solution'])[:, 0]
        rec_t_cfgs = torch.FloatTensor(prev_rec['solution'])[:, -1]
        assert torch.all(torch.isclose(rec_s_cfgs, s_cfgs)) and torch.all(torch.isclose(rec_t_cfgs, t_cfgs))
    for test_it, (start_cfg, target_cfg) in tqdm(enumerate(zip(s_cfgs, t_cfgs)), desc='Test Query'):
        options['seed'] += 1 # Otherwise the random initialization will stay the same every problem
        if prev_rec != {}:
            tmp_rec = {k: prev_rec[k][test_it] for k in prev_rec}
        elif method == 'fclgradfree':
            tmp_rec = gradient_free_traj_optimize(robot, lambda cfg: fcl_checker.predict(cfg, distance=False), start_cfg, target_cfg, options=options)
        elif method == 'fcldist':
            tmp_rec = gradient_free_traj_optimize(robot, fcl_checker.score, start_cfg, target_cfg, options=options)
        elif method == 'diffco':
            tmp_rec = adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
        elif method == 'bidiffco':
            tmp_rec = gradient_free_traj_optimize(robot, lambda cfg: 2*(dist_est(cfg)>=0).type(torch.FloatTensor)-1, start_cfg, target_cfg, options=options)
        elif method == 'diffcogradfree':
            with torch.no_grad():
                tmp_rec = gradient_free_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
        elif method == 'givengrad':
            tmp_rec = givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=options)
        else:
            raise NotImplementedError('Method = {} not implemented'.format(method))
        
        # Verification
        # if tmp_rec['success']:
        def con_max_move(p):
            control_points = robot.fkine(p)
            return torch.all((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.5**2 <= 0).item()
        def con_collision_free(p):
            return torch.all(fcl_checker.predict(p, distance=False) < 0).item()
        def con_joint_limit(p):
            return (torch.all(robot.limits[:, 0]-p <= 0) and torch.all(p-robot.limits[:, 1] <= 0)).item()

        def validate(solution):
            veri_cfgs = [utils.anglin(q1, q2, args.validate_density, endpoint=False)\
                for q1, q2 in zip(solution[:-1], solution[1:])]
            veri_cfgs = torch.cat(veri_cfgs, 0)
            collosion_free = con_collision_free(veri_cfgs) # torch.all(fcl_checker.predict(veri_cfgs, distance=False) < 0).item()
            sol_tensor = torch.FloatTensor(solution)
            within_jointlimit = con_joint_limit(sol_tensor)
            within_movelimit = con_max_move(sol_tensor)
            # withinlimit = torch.all(robot.limits[:, 0] <= torch.FloatTensor(tmp_rec['solution'])).item() \
            #     and torch.all(torch.FloatTensor(tmp_rec['solution']) <= robot.limits[:, 1]).item()
            return collosion_free and within_jointlimit and within_movelimit
        
        if 'fcl' in method and args.validate_density == 1: # skip validation if using fcl and density is only 1
            val_t = 0
        else:
            val_t = time()
            tmp_rec['success'] = validate(tmp_rec['solution'])
            val_t = time() - val_t
        tmp_rec['val_time'] = val_t

        for k in tmp_rec:
            test_rec[k].append(tmp_rec[k])
        
        # Repair
        if not tmp_rec['success'] and method != 'fcldist':
            repair_rec = gradient_free_traj_optimize(robot, fcl_checker.score, start_cfg, target_cfg, 
                options={**repair_options, 'init_solution': torch.DoubleTensor(tmp_rec['solution'])})
            # repair_rec['success'] = validate(repair_rec['solution']) # validation not needed
        else:
            repair_rec = {
                'cnt_check': 0,
                'cost': tmp_rec['cost'],
                'time': 0,
                'success': tmp_rec['success'],
                'solution': tmp_rec['solution'],
            }
        for k in ['cnt_check', 'cost', 'time', 'success', 'solution']:
            test_rec['repair_'+k].append(repair_rec[k])
        
        cfg_path_plots = []
        if robot.dof > 2:
            fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
        elif robot.dof == 2:
            fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
        single_plot(robot, torch.FloatTensor(test_rec['repair_solution'][-1]), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
        debug_dir = join('debug', exp_name, method)
        if not isdir(debug_dir):
            os.makedirs(debug_dir)
        plt.savefig(join(debug_dir, 'debug_view_{}.png'.format(test_it)), dpi=500)
        plt.close()

        # break # debugging

    return test_rec

def main(method, exp_name, override=False, args=None):
    # method = 'fclgradfree'
    # method = 'diffco'
    # method = 'givengrad'

    if args.load_exp is not None:
        print('Loading experiment results from {}'.format(args.load_exp))
    data_folder = join('data', exp_name if args.load_exp is None else args.load_exp)
    res_folder = join('results', exp_name)
    restored_res_folder = res_folder if args.load_exp is None else join('results', args.load_exp)
    if not isdir(res_folder):
        os.makedirs(res_folder)
    elif not override:
        ans = input('Overriding {}. Continue?(Y/n)'.format(res_folder))
        if 'y' in ans or 'Y' in ans:
            pass
        else:
            exit(1)
    
    with open(join(res_folder, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)

    from glob import glob
    envs = sorted(glob(join(data_folder, '*.pt'),))

    for env_name in tqdm(envs):
        env_name = splitext(basename(env_name))[0]

        restore_rec_file = os.path.join(restored_res_folder, env_name+'.json')
        if os.path.isfile(restore_rec_file):
            with open(restore_rec_file, 'r') as f:
                all_rec = json.load(f)
            if method in all_rec and args.load_exp is None:
                continue
        else:
            assert args.load_exp is None, \
                'Trying to load experiment {}, but the result file {} does not exist'.format(args.load_exp, restore_rec_file)
            all_rec = {}
        test_rec = test_one_env(env_name, method=method, folder=data_folder, args=args, prev_rec=all_rec[method])
        
        rec_file = os.path.join(res_folder, env_name+'.json')
        all_rec[method] = test_rec
        with open(rec_file, 'w') as f:
            json.dump(all_rec, f, indent=4)

def additional_timing(method, exp_name):
    folder = join('data', exp_name)

    from glob import glob
    envs = sorted(glob(join(folder, '*.pt'),))

    train_ts = {}
    for obsn in [1,2,5,10,20]:
        train_ts[obsn] = []
        for env_name in tqdm(envs):
            if not '_{}obs_'.format(obsn) in env_name:
                continue
            env_name = splitext(basename(env_name))[0]
            t = test_one_env(env_name, method=method, folder=folder)
            train_ts[obsn].append(t)
    print('{}, {}:'.format(m, exp_name))
    for obsn in [1,2,5,10,20]:
        ts = np.array(train_ts[obsn])
        print('{} train times: {} mean {} std {} '.format(obsn, ts, ts.mean(), ts.std()))
    return train_ts



if __name__ == "__main__":
    # exp_name = '2d_2dof_exp1'
    # methods = ['fclgradfree'] #, 'diffco', 'givengrad', 'bidiffco', 'fclgradfree']
    # for m in methods:
    #     st = time()
    #     main(m, exp_name, override=False)
    #     et = time()
    #     print('Method {}, Exp {}, time = {:.3f} secs'.format(m, exp_name, et-st))
    
    exps = ['2d_2dof_exp2', '2d_3dof_exp2', '2d_7dof_exp2'] #['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1'] #, '2d_3dof_exp1'
    load_exps = ['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1']
    import diffco as Fastronpp
    methods = ['diffco', 'givengrad', 'bidiffco'] #'diffco', 'givengrad', 'bidiffco', 'fclgradfree'
    res = {}
    for exp_name, loadexp in zip(exps, load_exps):
        res[exp_name] = {}
        for m in methods:
            st = time()
            args = dict(
                load_exp=loadexp,
                include_validate_time=True,
                use_previous_solution=True,
                validate_density=1,
            )
            args=ExpConfigs(args)
            main(m, exp_name, override=True, args=args)
            # res[exp_name][m] = additional_timing(m, exp_name)
            et = time()
            print('Method {}, Exp {}, time = {:.3f} secs'.format(m, exp_name, et-st))
    # for exp_name in exps:
    #     for m in methods:
    #         print('{}, {}:'.format(m, exp_name))
    #         for obsn in [1,2,5,10,20]:
    #             ts = np.array(res[exp_name][m][obsn])
    #             print('{} train times: {} mean {} std {} '.format(obsn, ts, ts.mean(), ts.std()))