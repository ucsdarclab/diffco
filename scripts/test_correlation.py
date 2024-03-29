import argparse
import os
import pickle
from typing import Callable, Tuple, Union

import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
import torch
from diffco import CollisionChecker, DiffCo, MultiDiffCo, kernel
from diffco.model import Model
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage
from diffco.routines import train_checker, test_checker, train_test_split, fit_checker, generate_unified_grid
from diffco.routines import unpack_dataset, autogenerate_2d_dataset, load_pretrained_checker, get_estimator

sns.set()


def create_plots(
        robot: Model,
        obstacles: list,
        dist_est: Callable,
        gt_grid: torch.Tensor,
        use3d: bool = False,
        view_3d: str = 'surface',
        render_arrows: bool = True,
        render_configuration_point: bool = True) -> Tuple[torch.Tensor, list]:
    """Create plots for comparing FCL and DiffCo configuration spaces.
    
    Args:
        robot (Model): The 2 DOF robot from the dataset.
        obstacles (list): The obstacles from the dataset.
        dist_est (Callable): The distance estimator function.
        gt_grid (torch.Tensor): The dists calculated by FCL.
        use3d (bool): Flag for generating a 3D plot.
        view_3d (str): Type of 3D plot to generate. Must be one of 'surface',
            'wireframe', or 'contour'. Defaults to 'surface'.
        render_arrows (bool): Flag for rendering arrows on plot.
        render_configuration_point (bool): Flag for rendering current state in
            configuration space.
    """
    # Adjust figsize according to your specific arrangement
    fig = plt.figure(figsize=(4*2+0.5, 4 * 1))
    # fig = plt.figure(figsize=(3*2+0.5, 3 * 1)) # temp
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.set_xticklabels(['', '', ''])
    ax.set_yticklabels(['', '', ''])
    from matplotlib.cm import get_cmap
    cmap = get_cmap('RdBu_r')
    for obs in obstacles:
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()]))#, color=cmaps[1](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], color=cmap(0.8)))#, color=cmaps[0](0.5)))
            print((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    # q = torch.FloatTensor([-np.pi/8, -np.pi/4])#-np.pi/4])
    q = torch.ones(robot.dof) * np.pi/6 * (torch.randint(0, 3, [robot.dof])-1)
    points = robot.fkine(q)[0]
    points = torch.cat([torch.zeros(1, 2), points], dim=0)
    
    link_plot, = ax.plot(points[:, 0], points[:, 1], color='orange', alpha=1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    joint_plot, = ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', markersize=lw)

    size = [400, 400]
    grid_points = generate_unified_grid(*size)
    xx, yy = grid_points.reshape(*size, 2).unbind(dim=2)
    est_grid = dist_est(grid_points).reshape(size)
    gt_grid = gt_grid.reshape(size)
    # gt_grid = torch.from_numpy(ndimage.gaussian_filter(gt_grid, 15))
    # est_grid = torch.from_numpy(ndimage.gaussian_filter(est_grid, 3))

    c_axes = []
    with sns.axes_style('ticks'):
        for i, d in enumerate([gt_grid, est_grid], 1):
        # for i, d in enumerate([gt_grid], 1): # for clustering
            c_ax = fig.add_subplot(gs[0, i], projection='3d' if use3d else None)

            if use3d:
                if view_3d == 'wireframe':
                    c_ax.plot_wireframe(xx.numpy(), yy.numpy(), d.numpy(), alpha=1, #vmin=-d.max(), vmax=d.max(), 
                        rstride=10, cstride=10, linewidth=0.1, antialiased=False, edgecolor='black') # , 'RdBu_r' 
                elif view_3d == 'surface':
                    surf = c_ax.plot_surface(xx.numpy(), yy.numpy(), d.numpy(), alpha=1, #vmin=-d.max(), vmax=d.max(), 
                        rstride=10, cstride=10, linewidth=0.1, antialiased=True, cmap='Greys_r', edgecolor='black') # , 'RdBu_r' cmap='Greys_r', 
                    # surf.set_edgecolor('none')
                elif view_3d == 'contour':
                    c_ax.contour3D(xx, yy, d, 50, linewidths=1, alpha=1)
                else:
                    raise ValueError(view_3d)
                c_ax.view_init(60, 225)
                c_ax.set_facecolor('white')
                c_ax.w_xaxis.set_pane_color(ax.get_facecolor())
                c_ax.w_yaxis.set_pane_color(ax.get_facecolor())
                c_ax.w_zaxis.set_pane_color(ax.get_facecolor())
                c_ax.set_aspect('auto', adjustable='box')
            else:
                color_mesh = c_ax.pcolormesh(xx, yy, d, cmap='RdBu_r', vmin=-torch.abs(d).max(), vmax=torch.abs(d).max())#, alpha=0.5) # binary shading='gouraud', 
                c_ax.contour(xx, yy, d, levels=[0], linewidths=1, alpha=0.4) #-1.5, -0.75, 0, 0.3
                c_ax.set_aspect('equal', adjustable='box')

            if render_arrows:
                sparse_stride = 20
                sparse_score = d[5:-5:sparse_stride, 5:-5:sparse_stride]
                score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
                score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
                if use3d:
                    raise NotImplementedError('Arrows not rendered properly in 3D. Set render_arrows=False to continue.')
                    score_grad = np.stack([score_grad_x, score_grad_y, -(score_grad_x**2+score_grad_y**2)], axis=2)
                    score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
                    score_grad_x, score_grad_y, score_grad_z = [score_grad[:, :, dim] for dim in range(3)]
                    c_ax.quiver(xx[5:-5:sparse_stride, 5:-5:sparse_stride], yy[5:-5:sparse_stride, 5:-5:sparse_stride], sparse_score+0.05, 
                        score_grad_x, score_grad_y, score_grad_z,
                        color='red') #width=1e-2, headwidth=2, headlength=5
                else:
                    score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
                    score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True) / 10
                    score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
                    c_ax.quiver(xx[5:-5:sparse_stride, 5:-5:sparse_stride], yy[5:-5:sparse_stride, 5:-5:sparse_stride], score_grad_x, score_grad_y, 
                        color='red', width=1e-2, headwidth=2, headlength=5, pivot='mid')

            if render_configuration_point:
                if use3d:
                    nearest_gridpoint = np.argmin(np.square(grid_points-q).sum(axis=1))
                    q_score = d.reshape(-1, 1)[nearest_gridpoint]
                    circle_patch = Circle([q[0].item(), q[1].item()], np.pi/20, ec='k', fc="orange", alpha=1)
                    c_ax.add_patch(circle_patch)
                    from mpl_toolkits.mplot3d import art3d
                    art3d.pathpatch_2d_to_3d(circle_patch, z=q_score, zdir="z")
                else:
                    c_ax.scatter([q[0].item()], [q[1].item()], marker='o', s=40, c='orange', edgecolors='black')
            # fig.colorbar(color_mesh, ax=c_ax)

            c_ax.set_xlim(-np.pi, np.pi)
            c_ax.set_ylim(-np.pi, np.pi)
            c_ax.set_xticks([-np.pi, 0, np.pi])
            c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
            c_ax.set_yticks([-np.pi, 0, np.pi])
            c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
            c_axes.append(c_ax)
            c_ax.grid(False)
    
    return est_grid.reshape(-1), c_axes

def FastronClustering(cfgs, fkine, c_ax):
    from sklearn.cluster import KMeans
    numClusters = 3
    colorarr = ['b','g','r','c','m','y','k','w']
    kmeans = KMeans(n_clusters=numClusters).fit(fkine(cfgs).reshape(len(cfgs), -1))
    size = [400, 400]
    yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    preds = kmeans.predict(fkine(grid_points).reshape(len(grid_points), -1)).reshape(size)
    preds = torch.LongTensor(preds)
    
    c_ax.pcolormesh(xx, yy, preds, cmap='Set1', shading='gouraud', vmin=-0.5, vmax=numClusters-0.5,  alpha=0.05)
    c_ax.grid(False, visible=False)

def train():
    pass

def main(
        task: str,
        dataset_filepath: str,
        checker_type: CollisionChecker,
        lmbda: int,
        use_fk: bool,
        kernel_type: kernel.KernelFunc,
        fit_full_poly: bool,
        fitting_target: str,
        fitting_epsilon: float,
        scoring_method: str,
        safety_margin: int,
        pretrained_checker: str = None,
        random_seed: int = None) -> None:
    """Main entry point for the script. Trains/loads checker, calculates
    correlation, and compares different configurations.

    Args:
        task (str): The task to perform. Must be 'correlate' or 'compare'.
        dataset_filepath (str): Path to dataset.
        checker_type (CollisionChecker): The collision checker class.
        lmbda (int): Argument passed to RQKernel when training a new collision
            checker.
        use_fk (bool): Flag for using forward kinematics or not.
        kernel_type (KernelFunc): The type of kernel function to use when
            fit_full_poly is False. Currently supported kernel types are
            Polyharmonic and MultiQuadratic.
        fit_full_poly (bool): When True, uses the collision checkers
            fit_full_poly fitting function. When False, uses the fit_poly
            fitting function.
        fitting_target (str): The fitting target. Must be one of the following:
            'label', 'dist', or 'hypo'.
        fitting_epsilon (float): Argument passed to the checker's fit function.
        scoring_method (str): Scoring method for the collision checker.
            Supported scoring methods are 'rbf_score', 'poly_score', and
            'score'.
        safety_margin (int): Amount to offset test predictions by when running
            the scoring method.
        pretrained_checker (str): Path to a pretrained collision checker. If
            provided, the training phase is skipped and the pretrained checker
            is used instead (may provide a speedup).
        random_seed (int): Random seed used to reproduce the same results,
            useful for debugging. Defaults to None.
    """
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    if dataset_filepath is None:
        if task == 'correlate':
            dataset_filepath = autogenerate_2d_dataset(3, 150, 'binary', '3d_halfnarrow',
                link_length=2.5, random_seed=random_seed)
        elif task == 'compare':
            dataset_filepath = autogenerate_2d_dataset(2, 1, 'binary', '1rect', 160000, 3, False, random_seed)
        else:
            raise ValueError(task)
    robot, cfgs, labels, dists, obstacles = unpack_dataset(dataset_filepath)
    description = os.path.splitext(os.path.basename(dataset_filepath))[0]  # Remove the .pt extension
    fkine = robot.fkine if use_fk else None

    # Train on 75% of the data or 10,000 instances, whichever is smaller
    train_indices, test_indices = train_test_split(len(cfgs), min(int(0.75 * len(cfgs)), 10000))
    if pretrained_checker:
        checker = load_pretrained_checker(pretrained_checker)
    else:
        checker = train_checker(checker_type, cfgs[train_indices], labels[train_indices],
            dists[train_indices], fkine, obstacles, description, lmbda)
    fit_checker(checker, kernel_type, fit_full_poly, fitting_target, fitting_epsilon, fkine)
    dist_est = get_estimator(checker, scoring_method)
    test_checker(checker, dist_est, cfgs[test_indices], labels[test_indices], safety_margin)

    if task == 'correlate':
        correlation_filename = f'{description}_{fitting_target}_{"woFK" if checker.fkine is None else "withFK"}.png'
        gt_grid = dists[test_indices]
        est_grid = dist_est(cfgs[test_indices])
        correlation(gt_grid, est_grid, correlation_filename)
        test_error(gt_grid, est_grid)
    elif task == 'compare':
        assert robot.dof == 2, f"Expected 2 degrees of freedom, got {robot.dof}"
        comparison_filename = f'{description}_{fitting_target}.png'
        compare(checker, robot, obstacles, cfgs, dists, dist_est, comparison_filename)
    else:
        raise ValueError(task)

def correlation(gt_grid: torch.Tensor, est_grid: torch.Tensor, output_filename: str) -> None:
    """Calculate and plot correlation.

    Args:
        gt_grid (torch.Tensor): The ground truth grid.
        est_grid (torch.Tensor): The output from the distance estimator.
        output_filename (str): The desired filename of the output figure.
    """
    fig = plt.figure(figsize=(5, 5)) # temp
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    # gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(111)
    # ax.set_aspect('equal', adjustable='box')
    # ax.axis('equal')
    # ax.set_xlim((-4, 4))
    # ax.set_ylim((-3, 3))
    
    ax.scatter(gt_grid, est_grid, s=5)
    xlim_max = torch.FloatTensor(ax.get_xlim()).abs().max()
    ax.set_xlim(-xlim_max, xlim_max)
    ylim_max = torch.FloatTensor(ax.get_ylim()).abs().max()
    ax.set_ylim(-ylim_max, ylim_max)
    ax.axhline(0, linestyle='-', color='gray', alpha=0.5)
    ax.axvline(0, linestyle='-', color='gray', alpha=0.5)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')

    save_figure(os.path.join('figs/correlation', output_filename), dpi=300)


def test_error(gt_grid: torch.Tensor, est: torch.Tensor) -> None:
    """Calculate error for the test set.
    
    Args:
        gt_grid (torch.Tensor): The ground truth data.
        est (torch.Tensor): The output from the collision checker estimator for
            the data from the test set.
    """
    est = est / est.std() * gt_grid.std()
    print('{:.4f}, {:.4f}, {:.4f}'.format(
        (est-gt_grid).mean(), (est-gt_grid).std(), gt_grid.std()))


def compare(
        checker: CollisionChecker,
        robot: Model,
        obstacles: list,
        cfgs: torch.Tensor,
        dists: torch.Tensor,
        dist_est: Callable,
        output_filename: str,
        use_3d: bool = False,
        view_3d: str = 'surface',
        render_arrows: bool = False,
        render_configuration_point: bool = False) -> None:
    """Create 3 figures to compare the workspace and two configuration spaces
    for a 2 DOF robot.
    
    Args:
        checker (CollisionChecker): The trained collision checker.
        robot (Model): The 2 DOF robot from the dataset.
        obstacles (list): The obstacles from the dataset.
        cfgs (torch.Tensor): The data.
        dists (torch.Tensor): The dists.
        dist_est (Callable): The distance estimator function.
        output_filename (str): The desired filename of the output figure.
        use_3d (bool): Flag for generating a 3D plot.
        view_3d (str): Type of 3D plot to generate. Must be one of 'surface',
            'wireframe', or 'contour'. Defaults to 'surface'.
        render_arrows (bool): Flag for rendering arrows on plot.
        render_configuration_point (bool): Flag for rendering current state in
            configuration space.
    """
    checker.gains = checker.gains.reshape(-1, 1)
    est, c_axes = create_plots(robot, obstacles, dist_est, dists, use_3d, view_3d, render_arrows,
        render_configuration_point)
    c_support_points = checker.support_points
    c_axes[1].scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
    plt.tight_layout()
    save_figure(os.path.join('figs/comparison', output_filename), dpi=500)


def save_figure(filepath: str, dpi: int = 300, verbose: bool = True):
    """Save current matplotlib figure at desired filepath, creating
    subdirectories if necessary.
    
    Args:
        filepath (str): The desired filepath for the figure (will overwrite if
            there is an existing file).
        dpi (int): The image resolution. Defaults to 300.
        verbose (bool): Flag for printing out the filepath. Defaults to True.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=dpi)
    if verbose:
        print(f'Saved figure at {filepath!r}')


def plot_r_squared():
    raise NotImplementedError
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(est_grid.numpy().reshape(-1), gt_grid.numpy().reshape(-1))
    print('{}DOF, environment {}, with FK {}, r-squared: {}'.format(DOF, env_name, checker.fkine is not None, r_value**2))
    ax.text(xlim_max/4, -7*ylim_max/8, '$\\mathrm{R}^2='+('{:.4f}$'.format(r_value**2)), fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    ax.set_title('{} original supports, {} random samples'.format(checker.num_origin_supports, checker.n_left_out_points))

    plt.show()
    plt.savefig('figs/correlation/training_{}dof_{}_{}_{}ransample_rsquare.png'.format(DOF, env_name, 'hybriddiffco', checker.n_left_out_points))
    return r_value ** 2


def timing():
    raise NotImplementedError
    times = []
    st = time()
    for i, cfg in enumerate(cfgs):
        st1 = time()
        dist_est(cfg)
        end1 = time()
        times.append(end1-st1)
    dist_est(cfgs)
    end = time()
    times = np.array(times)
    print('std: {}, mean {}, avg {}'.format(times.std(), times.mean(), (end-st)/len(cfgs)))


def decomposition():
    raise NotImplementedError
    env_name_gt = env_name if 'compare' in env_name else env_name+'_for_compare'
    gt_grid = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['label']
    est, c_axes = create_plots(robot, obstacles, dist_est, gt_grid)
    DiffCoClustering(cfgs, fkine, c_axes[0])
    plt.show()


if __name__ == "__main__":
    desc = 'Tool for calculating and plotting correlation between DiffCo and FCL libraries.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('task', choices=['correlate', 'compare'])
    parser.add_argument('-c', '--checker', dest='checker_type', help='Collision checker class',
        choices=['DiffCo', 'MultiDiffco'], default='DiffCo')
    parser.add_argument('--pretrained-checker', help='path to pretrained collision checker', type=str, default=None)
    parser.add_argument('-d', '--dataset', dest='dataset_filepath', help='Dataset filepath')
    parser.add_argument('-l', '--lambdas', dest='lmbda', help='# of lambdas for DiffCo kernel', type=int, default=10)
    parser.add_argument('--no-fk', dest='use_fk', action='store_false', default=True)
    parser.add_argument('--fitting-target', choices=['label', 'dist', 'hypo'], default='label')
    parser.add_argument('--fitting-epsilon', type=float, default=0.01)
    parser.add_argument('-k', '--kernel-type', choices=['polyharmonic', 'multiquadratic'], default='polyharmonic')
    parser.add_argument('--fit-full-poly', action='store_true', default=False)
    parser.add_argument('--scoring-method', choices=['poly_score', 'full_poly_score', 'score'], default='poly_score')
    parser.add_argument('--safety_margin', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=2021)
    args = parser.parse_args()

    # Set checker
    if args.checker_type == 'DiffCo':
        args.checker_type = DiffCo
    elif args.checker_type == 'MultiDiffCo':
        args.checker_type = MultiDiffCo
    
    # Set kernel func
    if args.kernel_type == 'polyharmonic':
        args.kernel_type = kernel.Polyharmonic
    elif args.kernel_type == 'multiquadratic':
        args.kernel_type = kernel.MultiQuadratic

    main(**vars(args))
