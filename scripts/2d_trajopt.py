import argparse
import os
import sys
import json
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
from diffco import utils, CollisionChecker
from diffco.Obstacles import FCLObstacle
from trajectory_optim import adam_traj_optimize
from distest_error_vis import fit_checker, get_estimator, train_checker, train_test_split, unpack_dataset, test_checker, autogenerate_dataset

# def traj_optimize(robot, dist_est, start_cfg, target_cfg, history=False):
#     # There is a slightly different version in speed_compare.py,
#     # which allows using SLSQP instead of Adam, allows
#     # inputting an initial solution other than straight line,
#     # and is better modularly written.
#     # That one with SLSQP is more recommended to use.
#     N_WAYPOINTS = 20
#     NUM_RE_TRIALS = 10
#     UPDATE_STEPS = 200
#     dif_weight = 1
#     max_move_weight = 10
#     collision_weight = 10
#     safety_margin = torch.FloatTensor([-12, -1.2])#([-8.0, -0.8]) #
#     lr = 5e-1
#     seed = 19961221
#     torch.manual_seed(seed)

#     lowest_cost_solution = None
#     lowest_cost = np.inf
#     lowest_cost_trial = None
#     lowest_cost_step = None
#     best_valid_solution = None
#     best_valid_cost = np.inf
#     best_valid_step = None
#     best_valid_trial = None
    
#     trial_histories = []

#     found = False
#     for trial_time in range(NUM_RE_TRIALS):
#         path_history = []
#         if trial_time == 0:
#             init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
#         else:
#             init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
#         init_path[0] = start_cfg
#         init_path[-1] = target_cfg
#         p = init_path.requires_grad_(True)
#         opt = torch.optim.Adam([p], lr=lr)

#         for step in range(UPDATE_STEPS):
#             opt.zero_grad()
#             collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
#             control_points = robot.fkine(p)
#             max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-0.3**2, min=0).sum()
#             diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
#             constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
#             objective_loss = dif_weight * diff
#             loss = objective_loss + constraint_loss
#             loss.backward()
#             p.grad[[0, -1]] = 0.0
#             opt.step()
#             p.data = utils.wrap2pi(p.data)
#             if history:
#                 path_history.append(p.data.clone())
#             if loss.data.numpy() < lowest_cost:
#                 lowest_cost = loss.data.numpy()
#                 lowest_cost_solution = p.data.clone()
#                 lowest_cost_step = step
#                 lowest_cost_trial = trial_time
#             if constraint_loss <= 1e-2:
#                 if objective_loss.data.numpy() < best_valid_cost:
#                     best_valid_cost = objective_loss.data.numpy()
#                     best_valid_solution = p.data.clone()
#                     best_valid_step = step
#                     best_valid_trial = trial_time
#             if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
#                 print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
#                     trial_time, step, 
#                     collision_score.item(), collision_weight,
#                     max_move_cost.item(), max_move_weight,
#                     diff.item(), dif_weight,
#                     loss.item()))
#         trial_histories.append(path_history)
        
#         if best_valid_solution is not None:
#             found = True
#             break
#     if not found:
#         print('Did not find a valid solution after {} trials!\
#             Giving the lowest cost solution'.format(NUM_RE_TRIALS))
#         solution = lowest_cost_solution
#         solution_step = lowest_cost_step
#         solution_trial = lowest_cost_trial
#     else:
#         solution = best_valid_solution
#         solution_step = best_valid_step
#         solution_trial = best_valid_trial
#     path_history = trial_histories[solution_trial] # Could be empty when history = false
#     if not path_history:
#         path_history.append(solution)
#     else:
#         path_history = path_history[:(solution_step+1)]
#     return solution, path_history, solution_trial, solution_step


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


def main(
        dataset_filepath: str = None,
        checker_type: CollisionChecker = MultiDiffCo,
        start_cfg: torch.Tensor = None,
        target_cfg: torch.Tensor = None,
        num_waypoints: int = 12,
        safety_margin: torch.Tensor = None,
        cache: bool = False,
        random_seed: int = 19961221):
    if dataset_filepath is None:
        dataset_filepath = autogenerate_dataset(3, 5, 'class', '2class_1', random_seed=random_seed)
    robot, cfgs, labels, dists, obstacles = unpack_dataset(dataset_filepath)
    cfgs = cfgs.double()
    labels = labels.double()
    dists = dists.double()
    obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    fkine = robot.fkine

    # Train on 75% of the data or 10,000 instances, whichever is smaller
    train_indices, test_indices = train_test_split(len(cfgs), min(int(0.75 * len(cfgs)), 10000))
    if labels.dim() > 1 and checker_type != MultiDiffCo:
        raise ValueError(f'If data is nonbinary you must use MultiDiffCo, not {checker_type}')
    description = os.path.splitext(os.path.basename(dataset_filepath))[0]  # Remove the .pt extension
    checker = train_checker(checker_type, cfgs[train_indices], labels[train_indices],
        dists[train_indices], fkine, obstacles, description)
    test_checker(checker, checker.score, cfgs[test_indices], labels[test_indices])
    fit_checker(checker, fitting_epsilon=1, fitting_target='label', fkine=fkine)
    dist_est = get_estimator(checker, scoring_method='rbf_score')
    print('MIN_SCORE = {:.6f}'.format(dist_est(cfgs[test_indices]).min()))

    cfg_path_plots = []
    if robot.dof > 2:
        fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
    elif robot.dof == 2:
        fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)

    free_cfgs = cfgs[(labels == -1).all(axis=1)]
    indices = np.random.default_rng(random_seed).choice(len(free_cfgs), 2, replace=False)
    if start_cfg is None:
        start_cfg = free_cfgs[indices[0]]
    else:
        assert len(start_cfg) == free_cfgs.shape[-1]
    if target_cfg is None:
        target_cfg = free_cfgs[indices[1]]
    else:
        assert len(target_cfg) == free_cfgs.shape[-1]

    path_dir = 'results/safetybias'
    os.makedirs(path_dir, exist_ok=True)
    traj_optim_cached_filepath = os.path.join(path_dir, f'path_{description}.json')
    if cache and os.path.exists(traj_optim_cached_filepath):
        with open(traj_optim_cached_filepath, 'r') as f:
            optim_rec = json.load(f)
            # p = torch.FloatTensor(path_dict['solution'])
            # path_history = [torch.FloatTensor(shot) for shot in path_dict['path_history']] #[p] #
    else:
        if safety_margin is None:
            safety_margin = torch.zeros(labels.shape[-1])
        else:
            assert labels.shape[-1] == len(safety_margin)
        optim_options = {
            'N_WAYPOINTS': num_waypoints,
            'NUM_RE_TRIALS': 10,
            'MAXITER': 200,
            'safety_margin': safety_margin,
            'max_speed': 0.3,
            'seed': random_seed,
            'history': False
        }
        optim_rec = adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=optim_options)
        if cache:
            with open(traj_optim_cached_filepath, 'w') as f:
                json.dump(optim_rec,
                    # {
                    #     'path': p.data.numpy().tolist(), 
                    #     'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
                    #     'trial': num_trial,
                    #     'step': num_step
                    # },
                    f, indent=1)
                print('Plan recorded in {}'.format(f.name))

    ## This for doing the escaping-from-collision experiment
    # p = escape(robot, dist_est, start_cfg)
    # with open('results/esc_2d_{}dof_{}.json'.format(robot.dof, env_name), 'w') as f:
    #     json.dump({'path': p.data.numpy().tolist(), },f, indent=1)
    #     print('Plan recorded in {}'.format(f.name))
    
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
    single_plot(robot, torch.FloatTensor(optim_rec['solution']), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
    plt.tight_layout()
    fig_dir = 'figs/safetybias'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'_new_{description}.png'), dpi=500)
    # plt.savefig(os.path.join(fig_dir, '_new_2d_{}dof_{}_equalmargin'.format(robot.dof, env_name)), dpi=500) #_equalmargin.png

    # plt.savefig('figs/opening_contourline.png', dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    desc = 'Tool for generating optimized trajectories for 2D workspaces.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-d', '--dataset', dest='dataset_filepath', help='Dataset filepath')
    parser.add_argument('--checker', dest='checker_type', help='Collision checker class',
        choices=['diffco', 'multidiffco'], default='multidiffco')
    parser.add_argument('--start-cfg', nargs='*', type=float, help='Start configuration')
    parser.add_argument('--target-cfg', nargs='*', type=float, help='Final configuration')
    parser.add_argument('--num-waypoints', type=int, default=12, help='Number of waypoints')
    parser.add_argument('--safety-margin', nargs='*', type=float, help='Safety margin')
    parser.add_argument('--cache', action='store_true', default=False)
    parser.add_argument('--random-seed', type=int, default=19961221)
    args = parser.parse_args()

    if args.checker_type == 'diffco':
        args.checker_type = DiffCo
    elif args.checker_type == 'multidiffco':
        args.checker_type = MultiDiffCo
    
    if args.start_cfg:
        args.start_cfg = torch.Tensor(args.start_cfg)
    if args.target_cfg:
        args.target_cfg = torch.Tensor(args.target_cfg)
    if args.safety_margin:
        args.safety_margin = torch.Tensor(args.safety_margin)

    main(**vars(args))
