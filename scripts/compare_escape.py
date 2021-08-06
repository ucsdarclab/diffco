
import os
import json

import numpy as np
import torch

from diffco import DiffCo, MultiDiffCo
from diffco import kernel
from diffco import utils

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects

from escape import *

# A function that controls the style of visualization.
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
        fig = plt.figure(figsize=(3*(num_class + 1), 3 * num_class))
        gs = fig.add_gridspec(num_class, num_class+1)
        ax = fig.add_subplot(gs[:, :-1])
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
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
        cat = obs[3] if len(obs) >= 4 else 1
        print('{}, cat {}, {}'.format(obs[0], cat, obs))
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
            color=cmaps[cat](0.5)))
    
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
    # joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
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

# Commented out lines include convenient code for debugging purposes
def main(env_name, DOF, total_num_cfgs, key=None):
    # DOF = 2
    # env_name = # '2instance'

    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].double()
    dists = dataset['dist'].double()
    obstacles = dataset['obs']
    obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    print(obstacles)
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine

    if key is not None:
        env_name = f'{env_name}_{key}'
    checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check DiffCo test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.9)

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 1 #0.01
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine)#, reg=0.09) # epsilon=Epsilon,
    dist_est = checker.rbf_score
    min_score = dist_est(cfgs[train_num:]).min().item()
    print('MIN_SCORE = {:.6f}'.format(min_score))

    # return # DEBUGGING

    cfg_path_plots = []
    if robot.dof > 2:
        fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
    elif robot.dof == 2:
        fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)


    path_dir = 'results/escape'
    os.makedirs(path_dir, exist_ok=True)

    optim_esc_options = {
        'N_WAYPOINTS': 20,
        'safety_margin': min_score/5,
        'lr': 5e-2,
        'record_freq': None, # only last configuration
        'post_transform': utils.wrap2pi,
        'optimizer': torch.optim.Adam
    }

    cnt_valid = 0
    while cnt_valid < total_num_cfgs:
        cfg = resampling_escape(robot)
        cfg = optim_escape(robot, dist_est, cfg, optim_esc_options)
        # if fcl_checker(cfg):


    pathname = os.path.join(path_dir, 'esc_2d_{}dof_{}.json'.format(robot.dof, env_name))
    with open(pathname, 'w') as f:
        json.dump({'path': p.data.numpy().tolist(), },f, indent=1)
        print('Plan recorded in {}'.format(f.name))

    ## This is for loading previously computed trajectory
    with open(pathname, 'r') as f:
        optim_rec = json.load(f)
        # p = torch.FloatTensor(path_dict['solution'])
        # path_history = [torch.FloatTensor(shot) for shot in path_dict['path_history']] #[p] #
    
    ## This produces an animation for the trajectory
    # vid_name = None #'results/maual_trajopt_2d_{}dof_{}_fitting_{}_eps_{}_dif_{}_updates_{}_steps_{}.mp4'.format(
    #     # robot.dof, env_name, fitting_target, Epsilon, dif_weight, UPDATE_STEPS, N_STEPS)
    # if robot.dof == 2:
    #     animation_demo(
    #         robot, p, fig, link_plot, joint_plot, eff_plot, 
    #         cfg_path_plots=cfg_path_plots, path_history=path_history, save_dir=vid_name)
    # elif robot.dof == 7:
    #     animation_demo(robot, p, fig, link_plot, joint_plot, eff_plot, save_dir=vid_name)

    # (Recommended) This produces a single shot of the planned trajectory
    optim_rec['solution'] = optim_rec['solution'][0:1] + optim_rec['solution'][1:-2:2]+optim_rec['solution'][-2:]
    single_plot(robot, torch.FloatTensor(optim_rec['solution']), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
    plt.tight_layout()
    fig_dir = 'figs/safetybias'
    os.makedirs(fig_dir, exist_ok=True)
    # plt.show()
    plt.savefig(os.path.join(fig_dir, '_new_2d_{}dof_{}.png'.format(robot.dof, env_name)), dpi=500)
    # plt.savefig(os.path.join(fig_dir, '_new_2d_{}dof_{}_equalmargin'.format(robot.dof, env_name)), dpi=500) #_equalmargin.png

    # plt.savefig('figs/opening_contourline.png', dpi=500, bbox_inches='tight')
    
    




if __name__ == "__main__":
    main('2class_2', 7, key=None) #'equalmargin'
    # main('2class_1', 3, key='equalmargin')