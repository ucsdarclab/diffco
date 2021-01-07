import sys
import json
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
from time import time

def main(DOF, env_name, lmbda=10):
    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data']
    cfgs = cfgs.view(len(cfgs), -1)
    labels = dataset['label']
    obstacles = dataset['obs']
    robot = dataset['robot'](*dataset['rparam'])
    train_num = 35000
    indices = torch.LongTensor(np.random.choice(len(cfgs), train_num, replace=False))
    fkine = robot.fkine
    checker = DiffCo(obstacles, kernel_func=kernel.LineFKKernel(fkine, kernel.RQKernel(lmbda)), beta=1.0) # kernel.LineKernel(kernel.FKKernel(fkine, kernel.RQKernel(lmbda)))
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    keep_all = False
    if 'compare' not in env_name:
        checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]),
            keep_all=keep_all)
    else:
        checker.train(cfgs[indices], labels[indices], max_iteration=len(cfgs[indices]),
            keep_all=keep_all)

    # Check DiffCo test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    print(len(checker.gains), 'Support Points')
    # assert(test_acc > 0.9)

    return

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine)
    # checker.fit_rbf(kernel_func=kernel.MultiQuadratic(Epsilon), target=fitting_target, fkine=fkine)
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine) #, lmbd=10)
    dist_est = checker.rbf_score
    #  = checker.score
    # dist_est = checker.poly_score

    ''' ==================3-figure compare (work, c space 1, c space 2)==========
    size = [400, 400]
    env_name_gt = env_name if 'compare' in env_name else env_name+'_for_compare'
    # gt_grid = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['dist']
    # grid_points = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name_gt))['data']
    raw_grid_score = checker.score(grid_points)

    est, c_axes = create_plots(robot, obstacles, dist_est, gt_grid) # raw_grid_score)#gt_grid)
    plt.show()
    # plt.savefig('figs/original_DiffCo_score_compared.pdf'.format(env_name), dpi=500)
    # plt.savefig('figs/vis_{}.png'.format(env_name), dpi=500)
    '''
    
    '''# =============== test error ============
    # est = est / est.std() * gt_grid.std()
    # print('{:.4f}, {:.4f}, {:.4f}'.format(
    #     (est-gt_grid).mean(), (est-gt_grid).std(), gt_grid.std()))
    '''
    
    # ''' =============== correlation ==============
    gt_grid = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))['dist']

    # yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    # grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    est_grid = dist_est(cfgs[train_num:])

    # indices = np.random.choice(range(len(est_grid)), size=400, replace=False)
    gt_grid = gt_grid[train_num:]
    # est_grid = est_grid[indices]

    # ''' plot
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
    # ax.

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(est_grid.numpy().reshape(-1), gt_grid.numpy().reshape(-1))
    print('{}DOF, environment {}, with FK {}, r-squared: {}'.format(DOF, env_name, checker.fkine is not None, r_value**2))
    ax.text(xlim_max/4, -7*ylim_max/8, '$\\mathrm{R}^2='+('{:.4f}$'.format(r_value**2)), fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))#, fontdict={"family": "Times New Roman",})

    # plt.show()
    # plt.savefig('figs/correlation/{}dof_{}_{}.pdf'.format(DOF, env_name, fitting_target))#, dpi=500)
    plt.savefig('figs/correlation/{}dof_{}_{}_{}_rsquare.png'.format(DOF, env_name, fitting_target, 'woFK' if checker.fkine is None else 'withFK'), dpi=300)
    # '''
    
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
    return r_value ** 2
    
    
if __name__ == "__main__":
    # DOF = 2
    # env_name = '1rect' # '2rect' # '1rect_1circle' '1rect' 'narrow' '2instance' 3circle
    envs = [
        # (2, '1rect'),
        # (2, '3circle'),
        # (7, '1rect_1circle_7d'),
        # (7, '3circle_7d'),
        (2, '1rect_1circle_line'),
    ]
    for DOF, env_name in envs:
        main(DOF, env_name, lmbda=10)
    # lmbdas = np.power(10, np.arange(-1, 3, step=0.1))
    # rs = []
    # for DOF, env_name in envs:
    #     for lmbda in lmbdas:
    #         rs.append(main(DOF, env_name, lmbda))
    # plt.plot(lmbdas, rs)
    # plt.xticks(lmbdas)
    # plt.yticks(rs)
    # plt.show()
    # with open('rvalue_tests.json', 'w') as f:
    #     json.dump(
    #         {'lambda': lmbdas.tolist(),
    #         'rvalues': rs}, f)