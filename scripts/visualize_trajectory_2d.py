import torch
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects

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
        blit=False, init_func=init, repeat=False)
    writer = animation.PillowWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    
    if save_dir:
        # ani.save(save_dir, fps=FPS)
        ani.save(save_dir, writer=writer)
    else:
        # plt.axis('equal')
        # plt.axis('tight')
        plt.show()

# A function that controls the style of visualization.
def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    if robot.dof > 2:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111) #, projection='3d'
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*(num_class + 1), 3 * num_class))
        gs = fig.add_gridspec(num_class, num_class+1)
        ax = fig.add_subplot(gs[:, :-1])
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]), indexing='ij')
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[cat, -1])

                # score_DiffCo = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                checker.gains = checker.gains.reshape(len(checker.gains), num_class)
                c_support_points = checker.support_points[checker.gains[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                contour_plot = c_ax.contour(xx, yy, score, levels=[-18, -10, 0, 3.5 if cat==0 else 2.5], linewidths=1, alpha=0.4, colors='k') #-1.5, -0.75, 0, 0.3
                ax.clabel(contour_plot, inline=1, fmt='%.1f', fontsize=8)
                # Comment these out if you want colorbars, grad arrows for debugging purposes
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
                c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=18)
                c_ax.set_yticks([-np.pi, 0, np.pi])
                c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=18)

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=18)
    for obs in obstacles:
        print(f'Plotting obstacle {obs}')
        cat = obs[3] if len(obs) >= 4 else 0
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
            color=cmaps[cat](0.5)))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot([], [], color='silver', alpha=1.0, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
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

    for i in [0, -1]:
        link_traj[i].set_alpha(ends_alpha)
        link_traj[i].set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
        # joint_traj[i].set_alpha(ends_alpha)
        eff_traj[i].set_alpha(ends_alpha)
    link_traj[0].set_color('green')
    link_traj[-1].set_color('orange')

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