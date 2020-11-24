import sys
import json
sys.path.append('/home/yuheng/FastronPlus-pytorch/')
from Fastronpp import Fastron, MultiFastron
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
from time import time

def create_plots(robot, obstacles, dist_est, gt_grid):
    fig = plt.figure(figsize=(3*2+0.5, 3 * 1))
    # fig = plt.figure(figsize=(3*2+0.5, 3 * 1)) # temp
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    gs = fig.add_gridspec(1, 3)
    # gs = fig.add_gridspec(1, 2) # TEMP
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
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
    q = torch.FloatTensor([-np.pi/8, np.pi/4])#-np.pi/4])
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

    c_axes = []
    with sns.axes_style('ticks'):
        for i, d in enumerate([est_grid, gt_grid], 1):
        # for i, d in enumerate([est_grid], 1): #TEMP
        # for i, d in enumerate([gt_grid], 1): # for clustering
            c_ax = fig.add_subplot(gs[0, i])
            c_axes.append(c_ax)
            c_ax.grid(False)

            # score_fastron = checker.score(grid_points).reshape(size)
            # score = (torch.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
            color_mesh = c_ax.pcolormesh(xx, yy, d, cmap='RdBu_r', vmin=-torch.abs(d).max(), vmax=torch.abs(d).max())#, alpha=0.5) # binary shading='gouraud', 
            c_ax.contour(xx, yy, d, levels=[0], linewidths=1, alpha=0.4) #-1.5, -0.75, 0, 0.3
            # sparse_score = d[5:-5:20, 5:-5:20]
            # score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
            # score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
            # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
            # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
            # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
            # c_ax.quiver(xx[5:-5:20, 5:-5:20], yy[5:-5:20, 5:-5:20], score_grad_x, score_grad_y, color='red', width=1e-2, headwidth=2, headlength=5)
            # c_ax.scatter([q[0]], [q[1]], marker='o', s=40, c='orange', edgecolors='black')
            # fig.colorbar(color_mesh, ax=c_ax)

            c_ax.set_aspect('equal', adjustable='box')
            # c_ax.axis('equal')
            c_ax.set_xlim(-np.pi, np.pi)
            c_ax.set_ylim(-np.pi, np.pi)
            c_ax.set_xticks([-np.pi, 0, np.pi])
            c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
            c_ax.set_yticks([-np.pi, 0, np.pi])
            c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
    
    return est_grid.reshape(-1), c_axes

def FastronClustering(cfgs, fkine, c_ax):
    from sklearn.cluster import KMeans
    # N = 1000
    numClusters = 3
    colorarr = ['b','g','r','c','m','y','k','w']
    # data = np.random.rand(N,2)
    # data *= np.pi*2
    # FkData = np.zeros((N,4))
    # for i in range(N):
    # 	FkData[i,:] = FkKernel(data[i,0],data[i,1]).T

    # kmeans = KMeans(n_clusters=4, random_state=0).fit(FkData)
    kmeans = KMeans(n_clusters=numClusters).fit(fkine(cfgs).reshape(len(cfgs), -1))
    # datasets = []
    # for i in range(numClusters):
    # 	index = [kmeans.labels_ == i]
    # 	d = data[tuple(index)]
    # 	datasets.append(d)
    # 	plt.scatter(d[:,0],d[:,1],c=colorarr[i])
    size = [400, 400]
    yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    preds = kmeans.predict(fkine(grid_points).reshape(len(grid_points), -1)).reshape(size)
    preds = torch.LongTensor(preds)
    
    c_ax.pcolormesh(xx, yy, preds, cmap='Set1', shading='gouraud', vmin=-0.5, vmax=numClusters-0.5,  alpha=0.05)
    c_ax.grid(False, visible=False)

def main(DOF, env_name, lmbda=10):
    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data']
    labels = dataset['label']
    dists = dataset['dist']
    obstacles = dataset['obs']
    robot = dataset['robot'](*dataset['rparam'])
    train_num = 6000
    indices = torch.LongTensor(np.random.choice(len(cfgs), train_num, replace=False))
    fkine = robot.fkine
    checker = Fastron(obstacles, kernel_func=kernel.RQKernel(lmbda), beta=1.0) # TODO: kernel.FKKernel(fkine, 
    # checker = MultiFastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    keep_all = False
    if 'compare' not in env_name:
        checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num],
            keep_all=keep_all)
    else:
        checker.train(cfgs[indices], labels[indices], max_iteration=len(cfgs[indices]), distance=dists[indices],
            keep_all=keep_all)

    # Check Fastron test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    print(len(checker.gains), 'Support Points')
    # assert(test_acc > 0.9)

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target) #, fkine=fkine) # epsilon=Epsilon, 
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
    # plt.savefig('figs/original_fastron_score_compared.pdf'.format(env_name), dpi=500)
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
    FastronClustering(cfgs, fkine, c_axes[0])
    plt.show()
    '''
    return r_value ** 2
    
    




if __name__ == "__main__":
    # DOF = 2
    # env_name = '1rect' # '2rect' # '1rect_1circle' '1rect' 'narrow' '2instance' 3circle
    envs = [
        (2, '1rect'),
        (2, '3circle'),
        (7, '1rect_1circle_7d'),
        (7, '3circle_7d')
    ]
    for DOF, env_name in envs:
        main(DOF, env_name)
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