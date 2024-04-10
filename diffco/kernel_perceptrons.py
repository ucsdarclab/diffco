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


class Perceptron():
    def __init__(self):
        self.support_points = None
    
    def predict(self, point):
        return torch.any(torch.stack([obs.is_collision(point) for obs in self.obstacles], dim=1), dim=1)
    
    def score(self, point):
        raise NotImplementedError
    
    def line_predict(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.is_collision(p), points))
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    


class DiffCo(Perceptron):
    def __init__(self, kernel_func='rq', gamma=1, beta=1, transform=None):
        super().__init__()
        # self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        self.train_method = None
        self.kernel_func = kernel.RQKernel(gamma) if kernel_func=='rq' else kernel_func
        # self.gamma = self.kernel_func.gamma #C0.2 # 1/(2*self.support_points.var())
        self.beta = beta
        self.transform = transform
        self._cuda = False

        self.support_points = None
        self.gains = None
        self.hypothesis = None
        self.y = None
        self.distance = None
        self.kernel_matrix = None

    def train(self, X, y, max_iteration=1000, method='original', distance=None, keep_all=False, verbose=False):
        self.train_method = method
        self.distance = distance.reshape(-1) if distance is not None else None
        time_start = time()
        if method == 'original':
            self.train_perceptron(X, y, max_iteration, verbose=verbose)
        elif method == 'sgd':
            self.train_sgd(max_iteration)
        elif method == 'svm':
            self.train_svm()
        
        if not keep_all:
            mask = self.gains != 0
            if mask.sum() < 2:
                mask[torch.where(mask == 0)[0][0]] = True
            self.filter_support_points_(mask)
        time_elapsed = time() - time_start
        if verbose:
            print('{} training done. {:.4f} secs cost'.format(method, time_elapsed))
    
    def filter_support_points_(self, mask):
        self.support_points = self.support_points[mask]
        self.support_transformed = self.support_points if self.transform is None else self.support_transformed[mask]
        self.hypothesis = self.hypothesis[mask]
        self.y = self.y[mask]
        self.distance = self.distance[mask] if self.distance is not None else None
        self.gains = self.gains[mask]
        self.kernel_matrix = self.kernel_matrix[np.ix_(mask, mask)]

    def train_perceptron(self, X, y, max_iteration=1000, verbose=False):
        self.initialize(X, y)
        
        if verbose:
            print('DiffCo training...')
            start_time = time()
        for it in tqdm(range(max_iteration), ncols=0, dynamic_ncols=False, disable=not verbose):
            margin = self.y * self.hypothesis
            min_margin, min_i = torch.min(margin, 0)  #1./
            if self.kernel_matrix[min_i, min_i] == 0:
                self.kernel_matrix[min_i] = self.kernel_func(self.support_transformed[min_i], self.support_transformed)
                self.kernel_matrix[:, min_i] = self.kernel_matrix[min_i]
                # print(self.kernel_matrix[:, min_i], self.kernel_matrix[:, min_i].max(), self.kernel_matrix[:, min_i].min())
            if min_margin <= 0:
                delta_gain = (self.beta**((1+self.y[min_i])/2)*self.y[min_i] - self.hypothesis[min_i])/self.kernel_matrix[min_i, min_i]# 
                # assert delta_gain > -1000 and delta_gain < 1000
                self.gains[min_i] += delta_gain
                # assert delta_gain < 1000 and delta_gain > -1000 and self.kernel_matrix[min_i].max() < 1000
                self.hypothesis += delta_gain * self.kernel_matrix[min_i]
                # self.hypothesis[min_margin_idx] = self.gains @ self.kernel_matrix[:, min_margin_idx]
                continue
            
            modified_margin = self.y*(self.hypothesis - self.gains * np.diag(self.kernel_matrix)) * (self.gains != 0 )  # 
            max_margin, max_i = torch.max(modified_margin, 0)
            if max_margin > 0 and torch.sum(self.gains != 0) > 1:
                self.hypothesis -= self.gains[max_i]*self.kernel_matrix[max_i]
                self.gains[max_i] = 0
                continue

            break

        if verbose:
            print(f'Ended at iteration {it}, cost {time()-start_time:.4f} secs')
            print('ACC: {}'.format(torch.sum((self.hypothesis > 0) == (self.y > 0)) / float(len(self.y))))
    
    def initialize(self, X, y):
        self.support_points = X.clone()
        self.support_transformed = self.support_points if self.transform is None \
            else self.transform(self.support_points)
        self.y = y.reshape(-1).clone()
        assert len(y) == len(X)
        num_init_points = len(X)
        # self.support_points = torch.rand((num_init_points, 2), dtype=torch.float32) * 10
        self.gains = torch.zeros(num_init_points, dtype=X.dtype)
        # K = np.tile(self.support_points[np.newaxis, :], (num_init_points, 1, 1))
        # self.kernel_matrix = (self.support_points@self.support_points.T+1)**2
        # self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        self.kernel_matrix = torch.zeros((num_init_points, num_init_points), dtype=X.dtype)
        self.hypothesis = torch.zeros(num_init_points, dtype=X.dtype)
        self.max_n_support = 200  # TODO
        
    def train_sgd(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        y = torch.FloatTensor(self.y)
        K = torch.FloatTensor(self.kernel_matrix)
        gains = torch.FloatTensor(self.gains, requires_grad=True)

        # self.grad = self.kernel_matrix@self.y
        for it in range(max_iteration):
            margin = torch.matmul(gains, K)*y
            margin[margin > 0] = torch.log(1 + margin[margin > 0])
            sum_margin = margin.sum()
            gains.grad = None
            sum_margin.backward()
            gains.data += 0.001 * gains.grad
            # self.gains /= np.linalg.norm(self.gains)
        
        self.gains = gains.detach().numpy()
        
    def train_svm(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        self.svm = SVC(C=1e8, kernel='rbf', max_iter=max_iteration) 
        self.svm.fit(self.support_points, self.y)
        # self.support_points = self.svm.support_vectors_
        self.gains[self.svm.support_] = self.svm.dual_coef_.reshape(-1)
        self.intercept = self.svm.intercept_
        self.svm_gamma = self.svm._gamma # self.svm.gamma if isinstance(self.svm.gamma, (float, int)) else 1/(self.support_points.shape[1]*self.support_points.var())
        print('SVM Gamma: {}'.format(self.svm.gamma))
        # print('Intercept:', self.intercept)
        # print('Gains: ', self.svm.dual_coef_)
        print('ACC: {}'.format(np.sum((self.svm.predict(self.support_points) > 0) == (self.y > 0)) / len(self.y)))
    
    def fit_poly(self, kernel_func=None, target='hypo'):
        X = self.support_transformed
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.rbf_kernel = kernel_func
        kmat = self.rbf_kernel(X, X)

        self.rbf_nodes = torch.linalg.solve(kmat, y[:, None]).reshape(-1)
        # print(kmat@self.rbf_nodes) # DEBUG
        if self._cuda:
            self.cuda()
    
    def cuda(self):
        self.support_points = self.support_points.cuda()
        if self.transform is not None:
            self.support_transformed = self.support_transformed.cuda()
        self.rbf_nodes = self.rbf_nodes.cuda()
        self._cuda = True
    
    @property
    def device(self):
        return self.support_points.device
    
    def poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        point = point.to(device=self.rbf_nodes.device, dtype=self.rbf_nodes.dtype)
        if self.transform is not None:
            point = self.transform(point)
        supports = self.support_transformed
        return torch.matmul(self.rbf_kernel(point, supports), self.rbf_nodes.unsqueeze(1))
    
    def fit_full_poly(self, epsilon=1, k=2, lmbd=0, target='hypo'):
        X = self.support_transformed
        self.poly_kernel = kernel.Polyharmonic(k=k, epsilon=epsilon)
        phi = self.poly_kernel(X, X)
        phi.fill_diagonal_(lmbd)
        l1 = torch.cat([phi, X, torch.ones((len(X), 1))], dim=1)
        l2 = torch.cat([X.T, torch.zeros((X.shape[1], X.shape[1]+1))], dim=1)
        l3 = torch.cat([torch.ones((1, len(X))), torch.zeros(1, X.shape[1]+1)], dim=1)
        # print([l.shape for l in [l1, l2, l3]])
        L = torch.cat([l1, l2, l3], dim=0)
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.poly_nodes = torch.solve(
            torch.cat([y, torch.zeros(X.shape[1]+1)], dim=0).reshape(-1, 1),
            L).solution.reshape(-1)
    
    def full_poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point).reshape([len(point), -1])
        supports = self.support_transformed
        phi_x = torch.cat(
            [self.poly_kernel(point, supports), point, torch.ones(len(point), 1)], 
            dim=1)
        if phi_x.shape[0] == 1:
            phi_x = phi_x.squeeze_(0)
        return torch.matmul(phi_x, self.poly_nodes)

    def is_collision(self, point):
        return self.score(point) > 0
    
    def score(self, point):
        if self.train_method == 'original':
            return self.score_original(point)
        elif self.train_method == 'sgd':
            self.score_original(point)
        elif self.train_method == 'svm':
            self.score_svm(point)

    def score_original(self, point):
        # kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point)
        kernel_values = self.kernel_func(point, self.support_transformed)
        score = torch.matmul(kernel_values, self.gains)#.unsqueeze_(1))
        return score
    
    def score_nn(self, point):
        dif_abs = np.abs((self.support_points-point))
        dist = np.sqrt(np.sum(dif_abs**2, axis=1))
        dist -= dist.min()
        # print(dist.min())
        kernel_values = self.kernel_func(point, self.support_points)
        # nn_idx = np.argmax(kernel_values)
        # score = self.hypothesis[nn_idx] * (2-kernel_values[nn_idx])
        score = torch.matmul(self.gains, kernel_values)
        return score
    
    def score_svm(self, point):
        # kernel_values = self.kernel_func(point, self.support_points)
        kernel_values = np.exp(-self.svm_gamma*np.sum((self.support_points-point)**2, axis=1))
        return torch.matmul(self.gains, kernel_values) + self.intercept
        # return self.svm.decision_function(point.reshape(1, -1))

    def vis(model, size=100, seed=2019):
        import seaborn as sns
        sns.set()
        if isinstance(size, int):
            size = [size, size]
        yy, xx = torch.meshgrid(torch.linspace(0, 10, size[0]), torch.linspace(0, 10, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        fig, ax = plt.subplots(figsize=(4, 4)) #(figsize=(42, 10)) (14, 10)
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"]})

        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # model.initialize(3000)
        X = torch.rand(8000, 2) * 10
        gt_checker = Perceptron(model.obstacles)
        Y = gt_checker.predict(X).float() * 2 - 1

        model.train(X, Y, max_iteration=len(X), method='original')
        real_support_points = model.support_points
        grid_score = model.score(grid_points).reshape(size[0], size[1])
        # ax1 = plt.subplot(1,2,1)
        with sns.axes_style('ticks'):
            ax1 = plt.subplot(1, 1, 1)
            c = ax1.pcolormesh(xx, yy, grid_score, cmap='RdBu_r', vmin=-np.abs(grid_score).max(), vmax=np.abs(grid_score).max())
            ax1.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', s=3, c='black')
            ax1.contour(xx, yy, grid_score, levels=0, linewidths=1, alpha=0.3)
            ax1.axis('equal')
            ax1.set_aspect('equal', adjustable='box')
            # fig.colorbar(c, ax=ax1)
            sparse_score = grid_score[::20, ::20]
            score_grad_x = -ndimage.sobel(sparse_score, axis=1)
            score_grad_y = -ndimage.sobel(sparse_score, axis=0)
            score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
            score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
            score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
            ax1.quiver(xx[5:-5:20, ::20], yy[5:-5:20, ::20], score_grad_x, score_grad_y, width=1e-2, headwidth=2, headlength=5, color='red')
        # ax1.set_title('Original DiffCo (kernel={}), no. of support points = {}'.format(model.kernel_func.__class__.__name__+str(model.kernel_func.__dict__), len(real_support_points)))

        # grid_nn_score = np.fromiter(map(model.score_nn, grid_points), np.float).reshape((size[0], size[1]))
        # ax2 = plt.subplot(1,3,2)
        # c = ax2.pcolormesh(xx, yy, grid_nn_score, cmap='RdBu_r', vmin=-np.abs(grid_nn_score).max(), vmax=np.abs(grid_nn_score).max())
        # ax2.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
        # ax2.contour(xx, yy, (grid_nn_score).astype(float), levels=0)
        # ax2.axis('equal')
        # fig.colorbar(c, ax=ax2)
        # sparse_score = grid_nn_score[::10, ::10]
        # score_grad_x = -ndimage.sobel(sparse_score, axis=1)
        # score_grad_y = -ndimage.sobel(sparse_score, axis=0)
        # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
        # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
        # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
        # ax2.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=30, color='red')
        # ax2.set_title('NN inference, no. of support points = {}'.format(len(real_support_points)))

        # np.random.seed(seed)
        # model.initialize(3000)
        # model.train(1000, method='svm')
        # real_support_points = model.support_points
        # grid_svm_score = np.fromiter(map(model.score_svm, grid_points), np.float).reshape((size[0], size[1]))
        # ax3 = plt.subplot(122)
        # c = ax3.pcolormesh(xx, yy, grid_svm_score, cmap='RdBu_r', vmin=-np.abs(grid_svm_score).max(), vmax=np.abs(grid_svm_score).max())
        # ax3.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
        # ax3.contour(xx, yy, (grid_svm_score).astype(float), levels=0)
        # ax3.axis('equal')
        # fig.colorbar(c, ax=ax3)
        # sparse_score = grid_svm_score[::10, ::10]
        # score_grad_x = -ndimage.sobel(sparse_score, axis=1)
        # score_grad_y = -ndimage.sobel(sparse_score, axis=0)
        # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
        # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
        # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
        # ax3.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=30, color='red')
        # ax3.set_title('SVM, no. of support points={}'.format(len(model.support_points)))

        for obs in model.obstacles:
            if obs.kind == 'circle':
                circle_artist = plt.Circle(obs.position, radius=obs.size/2, color=[0, 0, 0, 0.1])
                ax1.add_artist(circle_artist)
                # circle_artist = plt.Circle(obs.position, radius=obs.size/2, color=[0, 0, 0, 0.3])
                # ax3.add_artist(circle_artist)
            elif obs.kind == 'rect':
                rect_artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color=[0, 0, 0, 0.1])
                ax1.add_artist(rect_artist)
                # rect_artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color=[0, 0, 0, 0.3])
                # ax3.add_artist(rect_artist)
            else:
                raise NotImplementedError('Unknown obstacle type')

        plt.show()


class DiffCoBeta(DiffCo):
    def __init__(self, obstacles, kernel_func='rq', rbf_kernel=None, gamma=1, beta=1, k=1, epsilon=1, gt_checker=None):
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
        self.kernel_matrix = self.rbf_kernel(X, X)+0.1*torch.eye(len(X)) # not considering inter-class
        assert (torch.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-torch.eye(len(X))).abs().max() < 1e-3, \
            (torch.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-torch.eye(len(X))).abs().max()

        # self.gains = self.kernel_matrix@self.distance/torch.sqrt(self.distance.reshape(1, -1)@self.kernel_matrix.transpose(1, 0)@self.kernel_matrix@self.distance.reshape(-1, 1))
        # self.gains = self.gains.reshape(-1)

        self.gains = torch.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
        self.hypothesis = self.kernel_matrix @ self.gains

        # self.gains = self.gains.requires_grad_(True)
        # opt = torch.optim.Adam([self.gains], lr=0.001)
        # for it in tqdm(range(max_iteration//100)):
        #     lda = 0.01 # 0.00001
        #     gtol = 0.05
        #     loss = (self.distance-self.kernel_matrix@self.gains).pow(2).mean() + lda*self.gains.pow(2).sum()*2
        #     if it % 100 == 0:
        #         print(it, 'Loss = ', loss.data.item())
        #     # if loss.data.item() < dtol or (self.gains.abs() < gtol).sum() == 0:
        #     #     break
            
        #     # for _ in range(200):
        #     #     loss = (self.distance-self.kernel_matrix@self.gains).pow(2).mean() + 1*self.gains.abs().sum()
        #     #     opt.zero_grad()
        #     #     loss.backward()
        #     #     # opt.param_groupds
        #     #     opt.step()
            
        #     # sq = self.kernel_matrix.transpose(1, 0) @self.kernel_matrix \
        #     #         + lda*torch.eye(len(self.kernel_matrix))
        #     # sqinv = torch.inverse(sq)
        #     # assert (sqinv@sq - torch.eye(len(sq))).abs().max() < 1e-4, (sqinv@sq - torch.eye(len(sq))).abs().max()
        #     # self.gains = sqinv @ self.kernel_matrix.transpose(1, 0) @ self.distance[:, None]
        #     # self.gains = self.gains.view(-1)
        #     # assert (self.kernel_matrix@self.gains - self.distance).abs().max() < 1e-3, (self.kernel_matrix@self.gains - self.distance).abs().max()
        #     '''
        #     self.gains = torch.solve( #@self.kernel_matrix
        #         self.kernel_matrix.transpose(1, 0) @ self.distance[:, None], 
        #         self.kernel_matrix.transpose(1, 0) @self.kernel_matrix\
        #             + lda*torch.eye(len(self.kernel_matrix))
        #         ).solution.reshape(-1) #-0.05*torch.sign(self.gains[:, None])
            
        #     self.gains = self.gains.data
        #     print('Small gains: ', (self.gains.abs() < gtol).sum())
        #     self.gains[self.gains.abs() < gtol] = 0
        #     self.support_points = self.support_points[self.gains != 0]
        #     self.support_fkine = self.support_fkine[self.gains != 0]
        #     self.hypothesis = self.hypothesis[self.gains != 0]
        #     self.distance = self.distance[self.gains != 0]
        #     self.kernel_matrix = self.kernel_matrix[self.gains != 0][:, self.gains != 0]
        #     '''
            
        #     self.gains = torch.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
        #     # self.gains = self.gains.requires_grad_(True)
        #     # opt = torch.optim.Adam([self.gains], lr=0.001)
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
        # self.kernel_matrix = torch.zeros((num_init_points, num_init_points), dtype=X.dtype)
        # self.hypothesis = torch.zeros(num_init_points, dtype=X.dtype)
    
    
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


class MultiDiffCo(DiffCo):
    def __init__(self, objects, kernel_func='rq', gamma=1, beta=1, gt_checker=None):
        super().__init__(objects, kernel_func, gamma, beta, gt_checker)
    
    def train(self, X, y, max_iteration=1000, gains=None, hypothesis=None, method='original', distance=None, kernel_matrix=None):
        self.train_method = method
        self.distance = distance
        time_start = time()
        if method == 'original':
            self.train_perceptron(X, y, max_iteration, gains, hypothesis, kernel_matrix)
        elif method == 'sgd':
            self.train_sgd(max_iteration)
        elif method == 'svm':
            self.train_svm()
        
        non_zero_weight_cnt = torch.sum(self.gains != 0, axis=1)
        self.filter_support_points_(non_zero_weight_cnt != 0)
        # self.support_points = self.support_points[non_zero_weight_cnt != 0]
        # self.kernel_matrix = self.kernel_matrix[non_zero_weight_cnt != 0]
        # self.kernel_matrix = self.kernel_matrix[:, non_zero_weight_cnt != 0]
        # self.hypothesis = self.hypothesis[non_zero_weight_cnt != 0]
        # self.y = self.y[non_zero_weight_cnt != 0]
        # if distance is not None:
        #     self.distance = distance[non_zero_weight_cnt != 0]
        # else:
        #     self.distance = None
        # self.gains = self.gains[non_zero_weight_cnt != 0]

        time_elapsed = time() - time_start
        print('{} training done. {:.4f} secs cost'.format(method, time_elapsed))

    def train_perceptron(self, X, y, max_iteration=1000, gains=None, hypothesis=None, kernel_matrix=None):
        self.initialize(X, y, gains=gains, hypothesis=hypothesis, kernel_matrix=kernel_matrix)
        complete = torch.zeros(self.num_class, dtype=torch.bool)

        print('MultiDiffCo training...')
        for it in tqdm(range(max_iteration)):
            margin = self.y * self.hypothesis
            
            for c in range(self.num_class):
                c_margin, c_y, c_gain, c_hypo = margin[:, c], self.y[:, c], self.gains[:, c], self.hypothesis[:, c]
                min_margin, min_i = torch.min(c_margin, 0)
                if self.kernel_matrix[min_i, min_i] == 0:
                    self.kernel_matrix[min_i] = self.kernel_func(self.support_points[min_i], self.support_points)
                    self.kernel_matrix[:, min_i] = self.kernel_matrix[min_i]
                if min_margin <= 0:
                    delta_gain = (self.beta**((1+c_y[min_i])/2) * c_y[min_i] - c_hypo[min_i])/self.kernel_matrix[min_i, min_i]
                    c_gain[min_i] += delta_gain
                    c_hypo += delta_gain * self.kernel_matrix[min_i]
                    continue
            
                modified_margin = c_y*(c_hypo - c_gain * np.diag(self.kernel_matrix)) * (c_gain != 0 )  # 
                max_margin, max_i = torch.max(modified_margin, 0)
                if max_margin > 0 and torch.sum(c_gain != 0) > 1:
                    c_hypo -= c_gain[max_i] * self.kernel_matrix[max_i]
                    c_gain[max_i] = 0
                    continue

                complete[c] = True

            if torch.min(complete):
                break
        
        print('Ended at iteration {}'.format(it))
        print('ACC: {}'.format(torch.sum((self.hypothesis > 0) == (self.y > 0)) / (np.prod(self.y.shape))))

    def train_sgd(self, max_iteration=1000):
        raise NotImplementedError
    
    def train_svm(self):
        raise NotImplementedError

    def initialize(self, X, y, gains=None, hypothesis=None, kernel_matrix=None): 
        self.support_points = X.clone()
        self.y = y.clone()
        num_init_points = len(X)
        self.num_class = y.shape[1]
        if gains is None and hypothesis is None and kernel_matrix is None:
            self.gains = torch.zeros((num_init_points, self.num_class), dtype=X.dtype)
            self.hypothesis = torch.zeros((num_init_points, self.num_class), dtype=X.dtype)
            self.kernel_matrix = torch.zeros((num_init_points, num_init_points), dtype=X.dtype)
        elif gains is None or hypothesis is None or kernel_matrix is None:
            raise ValueError('DiffCo: you passed in some existing parameters but not all three of gains, hypothesis, and kernel_matrix')
        else:
            self.gains = gains
            self.hypothesis = hypothesis
            self.kernel_matrix = kernel_matrix
        # self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        # self.max_n_support = 200 # Not enforced, might be a TODO
    
    def predict(self, point):
        score = self.score(point)
        # max_class_idx = np.argmax(score)
        return (score > 0)*2-1
        # return np.argmax(self.score(point))
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def score(self, points):
        kernel_values = self.kernel_func(points, self.support_points)
        scores = torch.matmul(kernel_values, self.gains)
        # kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        # score = self.gains@kernel_values
        return scores

    def fit_poly(self, kernel_func=None, target='hypo', fkine=None, reg=0):
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

        # min_d, min_i = (kmat+torch.eye(len(kmat)).fill_diagonal_(float('inf'))).min(dim=0)
        # plt.plot(range(len(kmat)), min_d)
        # plt.show()
        for c in range(self.num_class):
            c_nonzero = torch.nonzero(self.gains[:, c])
            c_iszero = torch.nonzero(self.gains[:, c] == 0).reshape(-1)
            assert len(c_nonzero) + len(c_iszero) == len(self.gains)
            ridx = c_nonzero.repeat([1, len(c_iszero)]).reshape(-1)
            cidx = c_iszero.repeat([len(c_nonzero), 1]).reshape(-1)
            kmat[ridx, cidx] = 0
            kmat[cidx, ridx] = 0
        self.rbf_nodes = torch.linalg.solve(kmat+reg*torch.eye(len(kmat), dtype=kmat.dtype), y)
        for c in range(self.num_class):
            self.rbf_nodes[self.gains == 0] = 0
        assert self.rbf_nodes.shape == (len(self.support_points), self.num_class)
    
    def rbf_score(self, point):
        if point.ndim == 1:
            point = point[None, :]
        if self.fkine is not None:
            point = self.fkine(point).reshape((len(point), -1))
            supports = self.support_fkine
        else:
            supports = self.support_points
        # kmat = self.rbf_kernel(point, supports)
        # class_scores = []
        # for c in range(self.num_class):
        #     class_scores.append(torch.matmul(kmat[:, self.gains[:, c] != 0], self.rbf_nodes[self.gains[:, c] != 0, c]))
        # return torch.stack(class_scores, dim=1)
        return torch.matmul(self.rbf_kernel(point, supports), self.rbf_nodes)
    
    def fit_full_poly(self, epsilon=1, k=2, lmbd=0, target='hypo', fkine=None):
        X = self.support_points
        if fkine is not None:
            X = fkine(X).reshape([len(X), -1])
            self.fkine = fkine
            self.support_fkine = X
        self.poly_kernel = kernel.Polyharmonic(k=k, epsilon=epsilon)
        phi = self.poly_kernel(X, X)
        phi.fill_diagonal_(lmbd)
        print(phi.shape)
        l1 = torch.cat([phi, X, torch.ones((len(X), 1))], dim=1)
        l2 = torch.cat([X.T, torch.zeros((X.shape[1], X.shape[1]+1))], dim=1)
        l3 = torch.cat([torch.ones((1, len(X))), torch.zeros(1, X.shape[1]+1)], dim=1)
        print([l.shape for l in [l1, l2, l3]])
        L = torch.cat([l1, l2, l3], dim=0)
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.poly_nodes = torch.solve(
            torch.cat([y, torch.zeros(X.shape[1]+1, self.num_class)], dim=0),
            L).solution
    
    def poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.fkine is not None:
            point = self.fkine(point).reshape([len(point), -1])
            supports = self.support_fkine
        else:
            supports = self.support_points
        phi_x = torch.cat(
            [self.poly_kernel(point, supports), point, torch.ones(len(point), 1)], 
            dim=1)
        return torch.matmul(phi_x, self.poly_nodes)
    
    # def line_collision(self, start, target, res=50):
    #     points = np.linspace(start, target, res).reshape((1, res, -1))
    #     pair_diff = self.support_points[:, np.newaxis] - points
    #     kernel_values = 1/(1+self.gamma/2*np.sum(pair_diff**2, axis=2))**2
    #     scores = torch.matmul(self.gains, kernel_values)
    #     predicts = np.argmax(scores, axis=0)
    #     predicts[scores[predicts, range(len(predicts))] <= 0] = -1
    #     predicts += 1
    #     # return any(map(lambda p: p > 0 and self.objects[p-1].get_cost() == np.inf, predicts))
    #     return any(predicts > 0)

    
    def vis(self, size=100):
        if isinstance(size, int):
            size = [size, size]
        xx, yy = np.meshgrid(np.linspace(0, 10, size[0]), np.linspace(0, 10, size[1]))
        grid_points = np.stack([xx, yy], axis=2).reshape((-1, 2))
        # grid_score = np.fromiter(map(lambda p: self.score(p)[0], grid_points), np.float64).reshape((size[0], size[1]))
        grid_score = np.fromiter(map(self.predict, grid_points), np.float64).reshape((size[0], size[1]))
        # grid_nn_score = np.array(list(map(self.score_nn, grid_points))).reshape((size[0], size[1]))
        real_support_points = self.support_points#[self.gains != 0]

        fig, ax = plt.subplots()
        ax.set_title('number of support points = {}'.format(len(real_support_points)))

        c = ax.pcolormesh(xx, yy, grid_score, cmap='RdBu_r', vmin=-2, vmax=2)#, vmin=self.hypothesis.min(), vmax=self.hypothesis.max()) # vmin=-2, vmax=2)
        ax.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
        ax.contour(xx, yy, (grid_score>0).astype(float), levels=1)
        ax.axis('equal')
        fig.colorbar(c, ax=ax)

        # ax2 = plt.subplot(1,2,2)
        # c = ax2.pcolormesh(xx, yy, grid_nn_score, cmap='RdBu', vmin=-2, vmax=2)# , vmin=self.hypothesis.min(), vmax=self.hypothesis.max()) # vmin=-2, vmax=2)
        # ax2.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
        # ax2.contour(xx, yy, (grid_nn_score>0).astype(float), levels=1)
        # ax2.axis('equal')
        # fig.colorbar(c, ax=ax2)

        plt.show(block=False)


def test_multi_diffco():
    obstacles = [
        ('circle', (2, 1), 1.5),
        ('rect', (5, 7), (5, 3))]
    obstacles = [Obstacle(*param) for param in obstacles]

    np.random.seed(1314)
    classifier = MultiDiffCo(obstacles, len(obstacles), gamma=1, beta=1)
    # classifier.train(1000)
    print(classifier.gains, classifier.gains.size)
    classifier.vis(200)
    plt.show()
    print(classifier.score([5, 7]))


def test_diffco():
    import kernel
    obstacles = [
        ('circle', (6, 2), 2),
        # ('circle', (2, 7), 1),
        ('rect', (3.5, 6), (2, 1)),
        ('rect', (4, 7), 1),
        ('rect', (5, 8), (10, 1)),
        ('rect', (7.5, 6), (2, 1)),
        ('rect', (8, 7), 1),]
    obstacles = [Obstacle(*param) for param in obstacles]
    # kernel = kernel.CauchyKernel(100)
    # k = kernel.TangentKernel(0.8, 0)
    k = kernel.RQKernel(5)
    # k = kernel.MultiQuadratic(0.7)
    # lambda x, x_prime: -k(x, x_prime)+k(np.array([0, 0]), np.array([[10, 10]]))
    checker = DiffCo(obstacles, kernel_func=k, beta=20)
    vis(checker, 200, seed=1917)


def test_diffco_beta():
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
 