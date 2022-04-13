'''
This is a script that comprises several small experiments/demos in the paper.
The best to use this is to comment/uncomment certain lines depending on the purpose.
'''

import argparse
import os
import pickle

import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
import torch
from diffco import DiffCo, MultiDiffCo, kernel, CollisionChecker
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage

sns.set()


def create_plots(robot, obstacles, dist_est, gt_grid, use3d=False):
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
    yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
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
                from matplotlib import cm
                # c_ax.plot_wireframe(xx.numpy(), yy.numpy(), d.numpy(), alpha=1, #vmin=-d.max(), vmax=d.max(), 
                #     rstride=10, cstride=10, linewidth=0.1, antialiased=False, edgecolor='black') # , 'RdBu_r' 
                surf = c_ax.plot_surface(xx.numpy(), yy.numpy(), d.numpy(), alpha=1, #vmin=-d.max(), vmax=d.max(), 
                    rstride=10, cstride=10, linewidth=0.1, antialiased=True, cmap='Greys_r', edgecolor='black') # , 'RdBu_r' cmap='Greys_r', 
                # c_ax.plot_wireframe(xx.numpy(), yy.numpy(), d.numpy(), rstride=5, cstride=5, color='black')
                # surf._facecolors2d=surf._facecolor3d
                # surf._facecolors3d = 'face'
                # surf.set_edgecolor('none')
                # c_ax.contour3D(xx, yy, d, 50, linewidths=1, alpha=1)
                # c_ax.plot_surface(xx[::199, ::199], yy[::199, ::199].numpy(), np.zeros_like(yy[::199, ::199].numpy()), alpha=0.3) #, vmin=-torch.abs(d).max(), vmax=torch.abs(d).max()) #cmap='RdBu_r', 
            else:
                color_mesh = c_ax.pcolormesh(xx, yy, d, cmap='RdBu_r', vmin=-torch.abs(d).max(), vmax=torch.abs(d).max())#, alpha=0.5) # binary shading='gouraud', 
                c_ax.contour(xx, yy, d, levels=[0], linewidths=1, alpha=0.4) #-1.5, -0.75, 0, 0.3

            # ============arrows and configuration point========
            sparse_stride = 20
            sparse_score = d[5:-5:sparse_stride, 5:-5:sparse_stride]
            score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
            score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
            if use3d:
                score_grad = np.stack([score_grad_x, score_grad_y, -(score_grad_x**2+score_grad_y**2)], axis=2)
                score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
                score_grad_x, score_grad_y, score_grad_z = [score_grad[:, :, dim] for dim in range(3)]
            else:
                score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
                score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True) / 10
                score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
            if use3d:
                # c_ax.quiver(xx[5:-5:sparse_stride, 5:-5:sparse_stride], yy[5:-5:sparse_stride, 5:-5:sparse_stride], sparse_score+0.05, 
                #     score_grad_x, score_grad_y, score_grad_z,
                #     color='red') #width=1e-2, headwidth=2, headlength=5
                c_ax.set_aspect('auto', adjustable='box')
                nearest_gridpoint = np.argmin(np.square(grid_points-q).sum(axis=1))
                print(xx.reshape(-1)[nearest_gridpoint], yy.reshape(-1)[nearest_gridpoint], )
                q_score = d.reshape(-1, 1)[nearest_gridpoint]
                # c_ax.scatter([q[0].item()], [q[1].item()], q_score+0.5, marker='o', s=40, c='orange', edgecolors='black', zorder=100)
                circle_patch = Circle([q[0].item(), q[1].item()], np.pi/20, ec='k', fc="orange", alpha=1)
                c_ax.add_patch(circle_patch)
                from mpl_toolkits.mplot3d import art3d
                art3d.pathpatch_2d_to_3d(circle_patch, z=q_score, zdir="z")
                c_ax.view_init(60, 225)
                c_ax.set_facecolor('white')
                c_ax.w_xaxis.set_pane_color(ax.get_facecolor())
                c_ax.w_yaxis.set_pane_color(ax.get_facecolor())
                c_ax.w_zaxis.set_pane_color(ax.get_facecolor())
            else:
                c_ax.quiver(xx[5:-5:sparse_stride, 5:-5:sparse_stride], yy[5:-5:sparse_stride, 5:-5:sparse_stride], score_grad_x, score_grad_y, 
                    color='red', width=1e-2, headwidth=2, headlength=5, pivot='mid')
                c_ax.set_aspect('equal', adjustable='box')
                c_ax.scatter([q[0].item()], [q[1].item()], marker='o', s=40, c='orange', edgecolors='black')
            # fig.colorbar(color_mesh, ax=c_ax)

            # c_ax.axis('equal')
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
        checker_type: CollisionChecker = DiffCo,
        pretrained_checker: str = None,
        DOF: int = None,
        env_name: str = None,
        dataset_filepath: str = None,
        lmbda=10,
        keep_all: bool = False,
        use_fk: bool = True,
        fitting_target: str = 'label',
        fitting_epsilon: float = 0.01,
        kernel_type: kernel.KernelFunc = kernel.Polyharmonic,
        fit_full_poly: bool = False,
        scoring_method: str = 'rbf_score',
        safety_margin: int = 0,
        random_seed: int = None):
    """Run experiment.

    checker_type (CollisionChecker): The collision checker class (defaults to
        DiffCo).
    pretrained_checker (str): Path to a pretrained collision checker. If
        provided, the training phase is skipped and the pretrained checker is
        used instead (may provide a speedup). If None (default), a new collision
        checker is trained and saved.
    DOF (int): Robot's degrees of freedom. Used to identify the dataset file if
        the path to the dataset is not provided. (Should deprecate in favor of
        requiring dataset filepath?) Defaults to None, but if a dataset filename
        is not provided, DOF must be provided.
    env_name (str): Dataset environment nickname. Used to identify the dataset
        file if the path to the dataset is not provided. (Should deprecate in
        favor of requiring dataset filepath?) Defaults to None, but if a dataset
        filename is not provided, env_name must be provided.
    dataset_filepath (str): Path to dataset. Defaults to None, in which case
        DOF and env_name must be provided.
    lmbda (int): Argument passed to RQKernel when training a new collision
        checker. Defaults to 10.
    keep_all (bool): Argument for training the collision checker. When False
        (default), support points are filtered. When True, all support points
        are kept.
    use_fk (bool): Flag for using forward kinematics or not. Defaults to True.
    fitting_target (str): The fitting target. Must be one of the following:
        'label', 'dist', or 'hypo'. Defaults to 'label'.
    fitting_epsilon (float): Argument passed to the checker's fit function.
        Defaults to 0.01.
    kernel_type (str): The type of kernel function to use when fit_full_poly is
        False. Currently supported kernel types are Polyharmonic and
        MultiQuadratic.
    fit_full_poly (bool): When True, uses the collision checkers fit_full_poly
        fitting function. When False (default), uses the fit_poly fitting
        function.
    scoring_method (str): Scoring method for the collision checker. Supported
        scoring methods are 'rbf_score', 'poly_score', and 'score'. Defaults to
        'rbf_score'.
    safety_margin (int): Amount to offset test predictions by when running the
        scoring method. Defaults to 0.
    random_seed (int): Random seed used to reproduce the same results, useful
        for debugging. Defaults to None
    """
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if env_name:
        dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    elif dataset_filepath:
        dataset = torch.load(dataset_filepath)
    cfgs = dataset['data']
    labels = dataset['label']
    dists = dataset['dist']
    obstacles = dataset['obs']
    if 'rparam' in dataset:
        robot = dataset['robot'](*dataset['rparam'])
    else:
        robot = dataset['robot']()
    train_num = 6000
    shuffled_indices = torch.LongTensor(np.random.choice(len(cfgs), len(cfgs), replace=False))
    train_indices = shuffled_indices[:train_num]
    test_indices = shuffled_indices[train_num:]
    fkine = robot.fkine if use_fk else None

    if pretrained_checker:
        with open(pretrained_checker, 'rb') as f:
            checker = pickle.load(f)
            print('checker loaded: {}'.format(f.name))
    else:
        kernel_func = kernel.FKKernel(fkine, kernel.RQKernel(lmbda)) if use_fk else kernel.RQKernel(lmbda)
        checker = checker_type(obstacles, kernel_func=kernel_func, beta=1.0) 
        checker.train(cfgs[train_indices], labels[train_indices], max_iteration=len(cfgs[train_indices]), distance=dists[train_indices],
                keep_all=keep_all)
        os.makedirs('results', exist_ok=True)
        with open('results/checker_errvis.p', 'wb') as f:
            pickle.dump(checker, f)
            print('checker saved: {}'.format(f.name))
    if fit_full_poly:
        checker.fit_full_poly(epsilon=fitting_epsilon, k=3, target=fitting_target, fkine=fkine)
    else:
        if kernel_type == kernel.Polyharmonic:
            kernel_func = kernel.Polyharmonic(1, fitting_epsilon)
        elif kernel_type == kernel.MultiQuadratic:
            kernel_func = kernel.MultiQuadratic(fitting_epsilon)
        else:
            raise NotImplementedError(kernel_type)
        checker.fit_poly(kernel_func=kernel_func, target=fitting_target, fkine=fkine)
    if scoring_method == 'rbf_score':
        dist_est = checker.rbf_score
    elif scoring_method == 'poly_score':
        dist_est = checker.poly_score
    elif scoring_method == 'score':
        dist_est = checker.score
    else:
        raise NotImplementedError(scoring_method)

    # Check DiffCo test ACC
    test_preds = (dist_est(cfgs[test_indices])-safety_margin > 0) * 2 - 1
    test_labels = labels[test_indices].reshape(test_preds.shape)
    test_acc = torch.sum(test_preds == test_labels, dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[test_labels ==1] == 1, dtype=torch.float32) / len(test_preds[test_labels ==1])
    test_tnr = torch.sum(test_preds[test_labels ==-1] == -1, dtype=torch.float32) / len(test_preds[test_labels==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    print(len(checker.gains), 'Support Points')
    
    '''# =============== test error ============
    # est = est / est.std() * gt_grid.std()
    # print('{:.4f}, {:.4f}, {:.4f}'.format(
    #     (est-gt_grid).mean(), (est-gt_grid).std(), gt_grid.std()))
    '''
    
    ''' diffco 3-figure compare (work, c space 1, c space 2)==========
    from diffco import DiffCo

    checker = DiffCo(
        obstacles, 
        kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), 
        rbf_kernel=kernel.Polyharmonic(1, epsilon=1)) #kernel.Polyharmonic(1, epsilon=1)) kernel.MultiQuadratic(epsilon=1)
    checker.train(cfgs[train_indices], dists[train_indices], fkine=fkine, max_iteration=int(1e4), n_left_out_points=300, dtol=1e-1)
    checker.gains = checker.gains.reshape(-1, 1)
    dist_est = checker.rbf_score

    size = [400, 400]
    env_name_gt = env_name if 'compare' in env_name else env_name+'_for_compare'
    gt_grid = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['dist']
    grid_points = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['data']
    raw_grid_score = checker.rbf_score(grid_points)
    raw_grid_score = torch.from_numpy(ndimage.gaussian_filter(raw_grid_score, 1))

    use3d = False
    dpi=100
    est, c_axes = create_plots(robot, obstacles, checker.rbf_score, gt_grid, use3d=use3d) # raw_grid_score)#gt_grid)
    c_support_points = checker.support_points
    c_axes[1].scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig('figs/original_DiffCo_score_compared_{}_dpi{}.jpg'.format('3d' if use3d else '2d', dpi), dpi=dpi, bbox_inches='tight')
    # plt.savefig('figs/robot_gallery/2d_{}dof_{}.jpg'.format(DOF, env_name), dpi=dpi, bbox_inches='tight')
    # plt.savefig('figs/vis_{}.png'.format(env_name), dpi=500)
    # plt.savefig('figs/new_diffco_vis.png')
    ''' #===============================

    # ''' =============== correlation ==============
    gt_grid = dists
    # gt_grid = checker.distance

    # size = [400, 400]
    # yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    # grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    est_grid = dist_est(cfgs[train_indices])
    # est_grid = dist_est(checker.support_points)

    # indices = np.random.choice(range(len(est_grid)), size=400, replace=False)
    gt_grid = gt_grid[train_indices]
    # est_grid = est_grid[indices]

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

    # from scipy import stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(est_grid.numpy().reshape(-1), gt_grid.numpy().reshape(-1))
    # print('{}DOF, environment {}, with FK {}, r-squared: {}'.format(DOF, env_name, checker.fkine is not None, r_value**2))
    # ax.text(xlim_max/4, -7*ylim_max/8, '$\\mathrm{R}^2='+('{:.4f}$'.format(r_value**2)), fontsize=15, 
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))
    # ax.set_title('{} original supports, {} random samples'.format(checker.num_origin_supports, checker.n_left_out_points))

    # plt.show()
    # plt.savefig('figs/correlation/training_{}dof_{}_{}_{}ransample_rsquare.png'.format(DOF, env_name, 'hybriddiffco', checker.n_left_out_points))
    # plt.savefig('figs/correlation/{}dof_{}_{}.pdf'.format(DOF, env_name, fitting_target))#, dpi=500)
    os.makedirs('figs/correlation', exist_ok=True)
    plt.savefig('figs/correlation/{}dof_{}_{}_{}.png'.format(DOF, env_name, fitting_target, 'woFK' if checker.fkine is None else 'withFK'), dpi=300)
    
    # ''' 

    ''' timing
    # times = []
    # st = time()
    # # for i, cfg in enumerate(cfgs):
    # #     st1 = time()
    # #     dist_est(cfg)
    # #     end1 = time()
    # #     times.append(end1-st1)
    # dist_est(cfgs)
    # end = time()
    # times = np.array(times)
    # print('std: {}, mean {}, avg {}'.format(times.std(), times.mean(), (end-st)/len(cfgs)))
    '''

    ''' decomposition
    env_name_gt = env_name if 'compare' in env_name else env_name+'_for_compare'
    gt_grid = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['label']
    est, c_axes = create_plots(robot, obstacles, dist_est, gt_grid) # raw_grid_score)#gt_grid)
    DiffCoClustering(cfgs, fkine, c_axes[0])
    plt.show()
    '''
    # return r_value ** 2

    
    




if __name__ == "__main__":
    desc = 'Tool for calculating and plotting correlation between DiffCo and FCL libraries.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c', '--checker', dest='checker_type', help='Collision checker class',
        choices=['DiffCo', 'MultiDiffco'], default='DiffCo')
    parser.add_argument('--pretrained-checker', help='path to pretrained collision checker', type=str, default=None)
    parser.add_argument('-d', '--dataset', dest='dataset_filepath', help='Dataset filepath')
    parser.add_argument('--dof', dest='DOF', help='degrees of freedom', type=int)
    parser.add_argument('--env', dest='env_name', help='environment tag name', type=str)
    parser.add_argument('-l', '--lambdas', dest='lmbda', help='# of lambdas for DiffCo kernel', type=int, default=10)
    parser.add_argument('--keep-all', action='store_true', default=False)
    parser.add_argument('--no-fk', dest='use_fk', action='store_false', default=True)
    parser.add_argument('--fitting-target', choices=['label', 'dist', 'hypo'], default='label')
    parser.add_argument('--fitting-epsilon', type=float, default=0.01)
    parser.add_argument('-k', '--kernel-type', choices=['polyharmonic', 'multiquadratic'], default='polyharmonic')
    parser.add_argument('--fit-full-poly', action='store_true', default=False)
    parser.add_argument('--scoring-method', choices=['rbf_score', 'poly_score', 'score'], default='rbf_score')
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

    if not args.dataset_filepath:
        if not args.DOF or not args.env_name:
            parser.error('without dataset, both --dof and --env are required')
    # print(vars(args))
    main(**vars(args))
