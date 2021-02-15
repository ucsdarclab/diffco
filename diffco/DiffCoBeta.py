import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.svm import SVC
from scipy import ndimage
from scipy.interpolate import Rbf
from tqdm import tqdm
from time import time
from . import kernel
from .Obstacles import Obstacle
from .DiffCo import CollisionChecker, DiffCo


class DiffCoBeta(DiffCo):
    def __init__(self, obstacles, kernel_func='rq', rbf_kernel=None, gamma=1, beta=1, gt_checker=None):
        super().__init__(obstacles, kernel_func, gamma, beta, gt_checker)
        # self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        # self.kernel_func = kernel.RQKernel(gamma) if kernel_func=='rq' else kernel_func
        self.rbf_kernel = kernel.Polyharmonic(k=1, epsilon=1) if rbf_kernel is None else rbf_kernel
        # self.gamma = self.kernel_func.gamma #C0.2 # 1/(2*self.support_points.var())
        self.fkine = None

    def train(self, X, d, fkine=None, max_iteration=1000, n_left_out_points = 100, dtol=1e-4, keep_all=False):
        time_start = time()
        self.n_left_out_points = n_left_out_points
        self.distance=d[:-n_left_out_points]
        self.train_perceptron(X[:-n_left_out_points], (d[:-n_left_out_points]>=0)*2.-1, max_iteration=max_iteration)
        if not keep_all:
            # self.gains[self.gains.abs() < 0.005] = 0
            self.support_points = self.support_points[self.gains != 0]
            # self.support_fkine = self.support_fkine[self.gains != 0]
            self.hypothesis = self.hypothesis[self.gains != 0]
            self.distance = self.distance[self.gains != 0] if self.distance is not None else None
            self.gains = self.gains[self.gains != 0]
            # self.rbf_nodes = self.gains
            print('Number of gains = ', len(self.gains))
        self.num_origin_supports = len(self.gains)

        self.train_distance(
            torch.cat([self.support_points, X[-n_left_out_points:]], dim=0), #self.support_points, #,  torch.cat([self.support_points, X[-n_left_out_points:]], dim=0)
            torch.cat([self.distance, d[-n_left_out_points:]], dim=0), #self.distance,  #, torch.cat([self.distance, d[-n_left_out_points:]], dim=0)
            fkine=fkine, max_iteration=max_iteration, dtol=dtol)
        time_elapsed = time() - time_start
        print('DiffCo training done. {:.4f} secs cost'.format(time_elapsed))
    
    def train_distance(self, X, d, fkine, max_iteration, dtol):
        self.initialize_distance(X, d)
        
        print('DiffCo training...')
        if fkine is not None:
            X = fkine(X).reshape([len(X), -1])
            self.fkine = fkine
            self.support_fkine = X
        self.kernel_matrix = self.rbf_kernel(X, X) # not considering inter-class
        # assert (torch.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-torch.eye(len(X))).abs().max() < 1e-3, \
            # (torch.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-torch.eye(len(X))).abs().max()

        # self.gains = self.kernel_matrix@self.distance/torch.sqrt(self.distance.reshape(1, -1)@self.kernel_matrix.transpose(1, 0)@self.kernel_matrix@self.distance.reshape(-1, 1))
        # self.gains = self.gains.reshape(-1)

        # self.gains = torch.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
        # self.hypothesis = self.kernel_matrix @ self.gains

        # self.gains = self.gains.requires_grad_(True)
        # opt = torch.optim.Adam([self.gains], lr=0.001)
        for it in tqdm(range(max_iteration//100)):
            lda = 0.01 # 0.00001
            gtol = 0.05
            loss = (self.distance-self.kernel_matrix@self.gains).pow(2).mean() + lda*self.gains.pow(2).sum()*2
            if it % 100 == 0:
                print(it, 'Loss = ', loss.data.item())
            # if loss.data.item() < dtol or (self.gains.abs() < gtol).sum() == 0:
            #     break
            
            # for _ in range(200):
            #     loss = (self.distance-self.kernel_matrix@self.gains).pow(2).mean() + 1*self.gains.abs().sum()
            #     opt.zero_grad()
            #     loss.backward()
            #     # opt.param_groupds
            #     opt.step()
            
            # sq = self.kernel_matrix.transpose(1, 0) @self.kernel_matrix \
            #         + lda*torch.eye(len(self.kernel_matrix))
            # sqinv = torch.inverse(sq)
            # assert (sqinv@sq - torch.eye(len(sq))).abs().max() < 1e-4, (sqinv@sq - torch.eye(len(sq))).abs().max()
            # self.gains = sqinv @ self.kernel_matrix.transpose(1, 0) @ self.distance[:, None]
            # self.gains = self.gains.view(-1)
            # assert (self.kernel_matrix@self.gains - self.distance).abs().max() < 1e-3, (self.kernel_matrix@self.gains - self.distance).abs().max()
            '''
            self.gains = torch.solve( #@self.kernel_matrix
                self.kernel_matrix.transpose(1, 0) @ self.distance[:, None], 
                self.kernel_matrix.transpose(1, 0) @self.kernel_matrix\
                    + lda*torch.eye(len(self.kernel_matrix))
                ).solution.reshape(-1) #-0.05*torch.sign(self.gains[:, None])
            
            self.gains = self.gains.data
            print('Small gains: ', (self.gains.abs() < gtol).sum())
            self.gains[self.gains.abs() < gtol] = 0
            self.support_points = self.support_points[self.gains != 0]
            self.support_fkine = self.support_fkine[self.gains != 0]
            self.hypothesis = self.hypothesis[self.gains != 0]
            self.distance = self.distance[self.gains != 0]
            self.kernel_matrix = self.kernel_matrix[self.gains != 0][:, self.gains != 0]
            '''
            
            self.gains = torch.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
            # self.gains = self.gains.requires_grad_(True)
            # opt = torch.optim.Adam([self.gains], lr=0.001)
        self.rbf_nodes = self.gains
        self.hypothesis = self.rbf_score(self.support_points) # self.kernel_matrix@self.gains

        # self.gains = self.gains.data

            

        # for it in tqdm(range(max_iteration)):
        #     abs_err = (self.hypothesis-self.distance).abs()
        #     max_err, max_i = torch.max(abs_err, 0)  #1./
        #     # if self.kernel_matrix[min_i, min_i] == 0:
        #     #     self.kernel_matrix[min_i] = self.kernel_func(self.support_points[min_i], self.support_points)
        #     #     self.kernel_matrix[:, min_i] = self.kernel_matrix[min_i]
        #     print('Max error:', max_err)
        #     if max_err >= dtol:
        #         delta_gain = (self.distance@self.kernel_matrix[max_i] - (self.kernel_matrix*self.kernel_matrix[max_i]).sum(1)@self.gains)\
        #             /self.kernel_matrix[max_i].pow(2).sum()
        #         # assert delta_gain > -1000 and delta_gain < 1000
        #         print('Delta gain', delta_gain)
        #         self.gains[max_i] += delta_gain
        #         # assert delta_gain < 1000 and delta_gain > -1000 and self.kernel_matrix[min_i].max() < 1000
        #         self.hypothesis += delta_gain * self.kernel_matrix[max_i]
        #         # self.hypothesis[min_margin_idx] = self.gains @ self.kernel_matrix[:, min_margin_idx]
        #         continue
            
        #     # modified_margin = self.y*(self.hypothesis - self.gains * np.diag(self.kernel_matrix)) * (self.gains != 0 )  # 
        #     # max_margin, max_i = torch.max(modified_margin, 0)
        #     # if max_margin > 0 and torch.sum(self.gains != 0) > 1:
        #     #     self.hypothesis -= self.gains[max_i]*self.kernel_matrix[max_i]
        #     #     self.gains[max_i] = 0
        #     #     continue

        #     break

        # print('Ended at iteration {}'.format(it))
        print('Max Abs error: {}'.format((self.hypothesis-self.distance).abs().max()))
    
    def initialize_distance(self, X, d):
        self.support_points = X.clone()
        self.distance = d.clone()
        num_init_points = len(X)
        self.gains = torch.zeros(num_init_points, dtype=X.dtype)
        self.kernel_matrix = torch.zeros((num_init_points, num_init_points), dtype=X.dtype)
        self.hypothesis = torch.zeros(num_init_points, dtype=X.dtype)
    
    
    def fit_poly(self, kernel_func=None, target='hypo', fkine=None): #epsilon=None, 
        X = self.support_points
        if fkine is not None:
            X = fkine(X).reshape([len(X), -1])
            self.fkine = fkine
            self.support_fkine = X
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.rbf_kernel = kernel.MultiQuadratic(1) if kernel_func is None else kernel_func
        kmat = self.rbf_kernel(X, X)

        self.rbf_nodes = torch.solve(y[:, None], kmat).solution.reshape(-1)
        # print(kmat@self.rbf_nodes) # DEBUG
    
    def rbf_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.fkine is not None:
            point = self.fkine(point).reshape([len(point), -1])
            supports = self.support_fkine
        else:
            supports = self.support_points
        return torch.matmul(self.rbf_kernel(point, supports), self.rbf_nodes.unsqueeze(1))

def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
    import seaborn as sns
    sns.set()
    import matplotlib.patheffects as path_effects
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
        ax = fig.add_subplot(gs[:, :-1]) #sum([list(range(r*(num_class+1)+1, (r+1)*(num_class+1))) for r in range(num_class)], [])) #, projection='3d'
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[cat, -1])

                # score_diffco = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_diffco)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_diffco)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                contour_plot = c_ax.contour(xx, yy, score, levels=[-18, -10, 0, 3.5 if cat==0 else 2.5], linewidths=1, alpha=0.4, colors='k') #-1.5, -0.75, 0, 0.3
                ax.clabel(contour_plot, inline=1, fmt='%.1f', fontsize=8)
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
    ax.tick_params(labelsize=18)
    for obs in obstacles:
        cat = obs[3] if len(obs) >= 4 else 1
        print('{}, cat {}, {}'.format(obs[0], cat, obs))
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



def main():
    DOF = 2
    env_name = '1rect' # '2rect' # '1rect_1circle' '1rect' 'narrow' '2instance'

    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data']
    labels = dataset['label'] #[:, 0] #.max(1).values
    dists = dataset['dist'] #.reshape(-1, 1) #.max(1).values
    obstacles = dataset['obs']
    obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    print(obstacles)
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine
    Epsilon = 1 #0.01
    checker = DiffCoBeta(obstacles, kernel_func=kernel.Polyharmonic(1, Epsilon))
    checker.train(cfgs[:train_num], dists[:train_num], fkine=fkine, max_iteration=int(1e4), dtol=1e-1)
    checker.gains = checker.gains.reshape(-1, 1)

    # Check DiffCo test ACC
    # test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    # test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    # test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    # test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    # print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    # assert(test_acc > 0.9)

    dist_est = checker.rbf_score
    print('MIN_SCORE = {:.6f}'.format(dist_est(cfgs[train_num:]).min()))

    cfg_path_plots = []
    if robot.dof > 2:
        fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
    elif robot.dof == 2:
        fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
    
    plt.show()