import numpy as np
from time import time
from tqdm import tqdm
import torch
from .DiffCo import CollisionChecker, DiffCo, Obstacle, plt
from . import kernel

class MultiChecker:
    def __init__(self, objects):
        self.objects = objects
    
    def predict(self, point):
        return np.fromiter(map(lambda obj: 1 if obj.is_collision(point) else -1, self.objects), int)
    
    def line_collision(self, start, target, res=50):
        predicts = list(map(lambda i: self.predict(start + (target - start)/res*i), range(res)))
        return any(map(lambda p: p > 0 and self.objects[p-1].get_cost() == np.inf, predicts))

class MultiDiffCo(DiffCo):
    def __init__(self, objects, kernel_func='rq', gamma=1, beta=1, gt_checker=None):
        super().__init__(objects, kernel_func, gamma, beta, gt_checker)
    
    def train(self, X, y, max_iteration=1000, gains=None, hypothesis=None, method='original', distance=None): #kernel_matrix=None
        self.train_method = method
        time_start = time()
        if method == 'original':
            self.train_perceptron(X, y, max_iteration, gains, hypothesis) #, kernel_matrix
        elif method == 'sgd':
            self.train_sgd(max_iteration)
        elif method == 'svm':
            self.train_svm()
        
        non_zero_weight_cnt = torch.sum(self.gains != 0, axis=1)
        self.support_points = self.support_points[non_zero_weight_cnt != 0]
        self.kernel_matrix = self.kernel_matrix[non_zero_weight_cnt != 0]
        self.kernel_matrix = self.kernel_matrix[:, non_zero_weight_cnt != 0]
        self.hypothesis = self.hypothesis[non_zero_weight_cnt != 0]
        self.y = self.y[non_zero_weight_cnt != 0]
        if distance is not None:
            self.distance = distance[non_zero_weight_cnt != 0]
        else:
            self.distance = None
        self.gains = self.gains[non_zero_weight_cnt != 0]

        time_elapsed = time() - time_start
        print('{} training done. {:.4f} secs cost'.format(method, time_elapsed))

    def train_perceptron(self, X, y, max_iteration=1000, gains=None, hypothesis=None): #, kernel_matrix=None):
        self.initialize(X, y, gains=gains, hypothesis=hypothesis)# , kernel_matrix=kernel_matrix)
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

    def train_sgd(self, max_iteration=1000):
        raise NotImplementedError
    
    def train_svm(self):
        raise NotImplementedError

    def initialize(self, X, y, gains=None, hypothesis=None): #, kernel_matrix=None
        self.support_points = X.clone()
        self.y = y.clone()
        num_init_points = len(X)
        self.num_class = y.shape[1]
        if gains is None and hypothesis is None:# and kernel_matrix is None:
            self.gains = torch.zeros((num_init_points, self.num_class), dtype=X.dtype)
            self.hypothesis = torch.zeros((num_init_points, self.num_class), dtype=X.dtype)
        elif gains is None or hypothesis is None: # or kernel_matrix is None:
            raise ValueError('DiffCo: you passed in some existing parameters but not both of gains and hypothesis')
        else:
            self.gains = gains
            self.hypothesis = hypothesis
            # self.kernel_matrix = kernel_matrix
        self.kernel_matrix = torch.zeros((num_init_points, num_init_points), dtype=X.dtype)
        # self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        self.max_n_support = 200 # Not enforced, might be a TODO
    
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
        print(X.shape)
        print(kmat.shape)

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
        print(y.dtype, kmat.dtype)
        self.rbf_nodes = torch.solve(y, kmat+reg*torch.eye(len(kmat), dtype=kmat.dtype)).solution
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


if __name__ == '__main__':
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