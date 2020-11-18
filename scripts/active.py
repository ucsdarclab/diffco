import sys
import json
sys.path.append('/home/yuheng/FastronPlus-pytorch/')
from Fastronpp import Fastron, MultiFastron, CollisionChecker
from Fastronpp import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from Fastronpp.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from Fastronpp import utils
from Fastronpp.Obstacles import FCLObstacle
from Fastronpp.FCLChecker import FCLChecker

def traj_optimize(robot, dist_est, start_cfg, target_cfg, history=False):
    N_WAYPOINTS = 20
    NUM_RE_TRIALS = 10
    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 10
    collision_weight = 10
    safety_margin = torch.FloatTensor([-1]) #-3, 
    lr = 5e-1

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
    # p = torch.FloatTensor(np.concatenate([np.linspace(start_cfg, (-np.pi, 0), N_STEPS/2), np.linspace((np.pi, 0), target_cfg, N_STEPS/2)], axis=0)).requires_grad_(True)
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0 and False: # Temp
            init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)

        for step in range(UPDATE_STEPS):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            # print(torch.clamp(dist_est(p)-safety_margin, min=0).max(dim=0).values.data)
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:, -1:]-control_points[:-1, -1:]).pow(2).sum(dim=2)-1**2, min=0).sum() \
                + torch.clamp((utils.wrap2pi(p[1:]-p[:-1])).pow(2).sum(dim=1)-(5*np.pi/180)**2, min=0).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            # np.clip(1.5*float(i)/UPDATE_STEPS, 0, 1)**2 (float(i)/UPDATE_STEPS) * 
            # torch.clamp(utils.wrap2pi(p[1:]-p[:-1]).abs(), min=0.3).pow(2).sum()
            constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data = utils.wrap2pi(p.data)
            if history:
                path_history.append(p.data.clone())
            # if loss.data.numpy() < lowest_cost:
            if constraint_loss.data.numpy() < lowest_cost:
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
    return solution, path_history, solution_trial, solution_step # sum(trial_histories, []),


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

'''
def create_plots(robot, obstacles, dist_est, checker, timesteps=3):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    work_axes = []
    c_axes = []
    cfg_path_plots = []
    link_plots = []
    joint_plots = []
    eff_plots = []

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    
    if robot.dof > 2:
        fig = plt.figure(figsize=(3*timesteps, 3))
        for t in range(timesteps):
            ax = fig.add_subplot(1, timesteps, t+1) #, projection='3d'
            work_axes.append(ax)
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*timesteps*num_class+0.5, 3 * (num_class+1)))
        gs = fig.add_gridspec(num_class+1, timesteps * num_class)
        for t in range(timesteps):
            ax = fig.add_subplot(gs[:num_class, t*num_class:(t+1)*num_class]) #sum([list(range(r*(num_class+1)+1, (r+1)*(num_class+1))) for r in range(num_class)], [])) #, projection='3d'
            work_axes.append(ax)

            cfg_path_plots_cur = []

            size = [400, 400]
            yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
            grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
            score_spline = dist_est(grid_points).reshape(size+[num_class])
            c_axes_cur = []
            with sns.axes_style('ticks'):
                for cat in range(num_class):
                    c_ax = fig.add_subplot(gs[-1, t*num_class+cat])

                    # score_fastron = checker.score(grid_points).reshape(size)
                    # score = (torch.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
                    # score = score_spline[:, :, cat]
                    # color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                    # c_support_points = checker.support_points[checker.gains[:, cat] != 0]
                    # c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                    # c_ax.contour(xx, yy, score, levels=[0], linewidths=1, alpha=0.4, ) #-1.5, -0.75, 0, 0.3
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
                    cfg_path_plots_cur.append(cfg_path)

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
                    c_axes_cur.append(c_axes)
                # c_ax.set_ticks('')
            c_axes.append(c_axes_cur)
            cfg_path_plots.append(cfg_path_plots_cur)
            work_axes.append(ax)


    # Plot ostacles
    # ax.axis('tight')
    for ax in work_axes:
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        for obs in obstacles:
            cat = obs[3] if len(obs) >= 4 else 1
            if obs[0] == 'circle':
                ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
            elif obs[0] == 'rect':
                ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
                color=cmaps[cat](0.5)))
                # print((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2))
        
        # Placeholder of the robot plot
        trans = ax.transData.transform
        lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
        link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
        joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
        eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)
        link_plots.append(link_plot)
        joint_plots.append(joint_plot)
        eff_plots.append(eff_plot)

    if robot.dof > 2:
        return fig, work_axes, link_plots, joint_plots, eff_plots
    elif robot.dof == 2:
        return fig, work_axes, link_plots, joint_plots, eff_plots, cfg_path_plots, c_axes
'''
def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]

    if robot.dof > 2:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) #, projection='3d'
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*(num_class), 3 * (num_class+1)))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        gs = fig.add_gridspec(num_class+1, num_class)
        ax = fig.add_subplot(gs[:-1, :]) #sum([list(range(r*(num_class+1)+1, (r+1)*(num_class+1))) for r in range(num_class)], [])) #, projection='3d'
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[-1, cat])

                # score_fastron = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains[:, cat] != 0]
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
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
            color=cmaps[cat](0.5)))
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
    points_traj = torch.cat([torch.zeros(len(p), 1, 2), points_traj], dim=1)
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

    # ---------Just for making a figure------------
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

def escape(robot, dist_est, start_cfg):
    N_WAYPOINTS = 20
    # NUM_RE_TRIALS = 10
    UPDATE_STEPS = 200
    # dif_weight = 1
    # max_move_weight = 10
    # collision_weight = 10
    safety_margin = -0.3 #torch.FloatTensor([-2, -0.2])
    lr = 5e-2
    # seed = 19961221
    # torch.manual_seed(seed)

    # lowest_cost_solution = None
    # lowest_cost = np.inf
    # lowest_cost_trial = None
    # lowest_cost_step = None
    # best_valid_solution = None
    # best_valid_cost = np.inf
    # best_valid_step = None
    # best_valid_trial = None
    
    # trial_histories = []

    # found = False
    # p = torch.FloatTensor(np.concatenate([np.linspace(start_cfg, (-np.pi, 0), N_STEPS/2), np.linspace((np.pi, 0), target_cfg, N_STEPS/2)], axis=0)).requires_grad_(True)
    # for trial_time in range(NUM_RE_TRIALS):
    path_history = []
    # if trial_time == 0:
    #     init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=UPDATE_STEPS))
    # else:
    #     init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
    init_path = start_cfg
    # init_path[-1] = target_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())

        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin #, min=0).sum()
        # print(torch.clamp(dist_est(p)-safety_margin, min=0).max(dim=0).values.data)
        # control_points = robot.fkine(p)
        # max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.0**2, min=0).sum()
        # diff = dif_weight * (control_points[1:]-control_points[:-1]).pow(2).sum()
        # np.clip(1.5*float(i)/UPDATE_STEPS, 0, 1)**2 (float(i)/UPDATE_STEPS) * 
        # torch.clamp(utils.wrap2pi(p[1:]-p[:-1]).abs(), min=0.3).pow(2).sum()
        # constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
        # objective_loss = dif_weight * diff
        loss = collision_score #objective_loss + constraint_loss
        loss.backward()
        # p.grad[[0, -1]] = 0.0
        opt.step()
        p.data = utils.wrap2pi(p.data)
        # if history:
        
        # if loss.data.numpy() < lowest_cost:
        #     lowest_cost = loss.data.numpy()
        #     lowest_cost_solution = p.data.clone()
        #     lowest_cost_step = step
        #     lowest_cost_trial = trial_time
        if collision_score <= 1e-4:
            # if objective_loss.data.numpy() < best_valid_cost:
            #     best_valid_cost = objective_loss.data.numpy()
            #     best_valid_solution = p.data.clone()
            #     best_valid_step = step
            #     best_valid_trial = trial_time
            break
        # if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
        #     print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
        #         trial_time, step, 
        #         collision_score.item(), collision_weight,
        #         max_move_cost.item(), max_move_weight,
        #         diff.item(), dif_weight,
        #         loss.item()))
        # trial_histories.append(path_history)
        
        # if best_valid_solution is not None:
        #     found = True
        #     break
    # if not found:
    #     print('Did not find a valid solution after {} trials!\
    #         Giving the lowest cost solution'.format(NUM_RE_TRIALS))
    #     solution = lowest_cost_solution
    #     solution_step = lowest_cost_step
    #     solution_trial = lowest_cost_trial
    # else:
    #     solution = best_valid_solution
    #     solution_step = best_valid_step
    #     solution_trial = best_valid_trial
    # path_history = trial_histories[solution_trial] # Could be empty when history = false
    # if not path_history:
    #     path_history.append(solution)
    # else:
    #     path_history = path_history[:(solution_step+1)]
    return torch.stack(path_history, dim=0)# sum(trial_histories, []),

def main():
    DOF = 2
    env_name = '1rect_active' # '2rect' # '1rect_1circle' '1rect' 'narrow' '2instance'

    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data']
    labels = dataset['label'].reshape(-1, 1) #.max(1).values
    dists = dataset['dist'].reshape(-1, 1) #.max(1).values
    obstacles = dataset['obs']
    obstacles = [list(o) for o in obstacles]
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine
    # checker = Fastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker = MultiFastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check Fastron test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.9)

    fitting_target = 'dist' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine) # epsilon=Epsilon,
    # checker.fit_rbf(kernel_func=kernel.MultiQuadratic(Epsilon), target=fitting_target, fkine=fkine)
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine, lmbd=10)
    dist_est = checker.rbf_score
    # dist_est = checker.score
    # dist_est = checker.poly_score
    print('MIN_SCORE = {:.6f}'.format(dist_est(cfgs[train_num:]).min()))

    #=================================================================================================================================
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

    label_type = 'binary'
    num_class = 1

    T = 11
    nu = 5 #5
    kai = 4000
    sigma = 0.3
    seed = 19961202
    torch.manual_seed(seed)

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
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
    gt_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)
    
    positions = torch.FloatTensor(np.linspace(obstacles[0][1], [4, 3], T))
    start_cfg = torch.zeros(robot.dof, dtype=torch.float32) # free_cfgs[indices[0]] # 
    target_cfg = torch.zeros(robot.dof, dtype=torch.float32) # free_cfgs[indices[1]] # 
    start_cfg[0] = np.pi/2 # -np.pi/16
    start_cfg[1] = -np.pi/6
    target_cfg[0] = 0 # -np.pi/2 # -15*np.pi/16
    target_cfg[1] = np.pi/7

    for t, trans in zip(range(T), positions):
        fcl_collision_obj[0].setTransform(fcl.Transform(
                    # Rotation.from_rotvec([0, 0, angle]).as_quat()[[3,0,1,2]], 
                    [trans[0], trans[1], 0]))
        for obs_mng in obs_managers:
            obs_mng.update()

        exploit_samples = torch.randn(nu, len(checker.gains), robot.dof) * sigma + checker.support_points
        exploit_samples = utils.wrap2pi(exploit_samples).reshape(-1, robot.dof)

        explore_samples = torch.rand(kai, robot.dof) * 2*np.pi - np.pi

        cfgs = torch.cat([exploit_samples, explore_samples, checker.support_points])
        labels, dists = gt_checker.predict(cfgs)
        print('Collision {}, Free {}\n'.format((labels == 1).sum(), (labels==-1).sum()))

        gains = torch.cat([torch.zeros(len(exploit_samples)+len(explore_samples), checker.num_class), checker.gains]) #None # 
        #TODO: bug: not calculating true hypothesis for new points
        added_hypothesis = checker.score(cfgs[:-len(checker.support_points)])
        hypothesis = torch.cat([added_hypothesis, checker.hypothesis]) # torch.cat([torch.zeros(len(exploit_samples)+len(explore_samples), checker.num_class), checker.hypothesis]) # None # 
        # kernel_matrix = torch.zeros(len(cfgs), len(cfgs)) #None # 
        # kernel_matrix[-len(checker.kernel_matrix):, -len(checker.kernel_matrix):] = checker.kernel_matrix

        checker.train(cfgs, labels, gains=gains, hypothesis=hypothesis, distance=dists) #, kernel_matrix=kernel_matrix
        print('Num of support points {}'.format(len(checker.support_points)))
        checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine, reg=0.1)

        print('t = {}'.format(t))
        if t % 1 == 0 and not torch.any(checker.predict(torch.stack([start_cfg, target_cfg], dim=0)) == 1):

            obstacles[0][1] = (trans[0], trans[1])
            cfg_path_plots = []
            if robot.dof > 2:
                fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
            elif robot.dof == 2:
                fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
            
            # Begin optimization
            # p, path_history, num_trial, num_step = traj_optimize(
            #     robot, dist_est, start_cfg, target_cfg, history=False)
            # with open('results/path_2d_{}dof_{}.json'.format(robot.dof, env_name), 'w') as f:
            #     json.dump(
            #         {
            #             'path': p.data.numpy().tolist(), 
            #             'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
            #             'trial': num_trial,
            #             'step': num_step
            #         },
            #         f, indent=1)
            #     print('Plan recorded in {}'.format(f.name))
            # p = escape(robot, dist_est, start_cfg)
            # with open('results/esc_2d_{}dof_{}.json'.format(robot.dof, env_name), 'w') as f:
            #     json.dump({'path': p.data.numpy().tolist(), },f, indent=1)
            #     print('Plan recorded in {}'.format(f.name))
            # with open('results/path_2d_{}dof_{}.json'.format(robot.dof, env_name), 'r') as f:
            #     path_dict = json.load(f)
            #     p = torch.FloatTensor(path_dict['path'])
            #     path_history = [torch.FloatTensor(shot) for shot in path_dict['path_history']] #[p] #
            
            #animation
            # vid_name = None #'results/maual_trajopt_2d_{}dof_{}_fitting_{}_eps_{}_dif_{}_updates_{}_steps_{}.mp4'.format(
            #     # robot.dof, env_name, fitting_target, Epsilon, dif_weight, UPDATE_STEPS, N_STEPS)
            # if robot.dof == 2:
            #     animation_demo(
            #         robot, p, fig, link_plot, joint_plot, eff_plot, 
            #         cfg_path_plots=cfg_path_plots, path_history=path_history, save_dir=vid_name)
            # elif robot.dof == 7:
            #     animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, save_dir=vid_name)

            # single shot
            p = torch.zeros(20, robot.dof)
            single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
            plt.show()
            # plt.savefig('figs/path_2d_{}dof_{}.png'.format(robot.dof, env_name), dpi=500)
            # plt.savefig('figs/active_2d_{}dof_{}_{}'.format(robot.dof, env_name, t), dpi=500) #_equalmargin.png
    
    




if __name__ == "__main__":
    main()