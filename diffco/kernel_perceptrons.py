import numpy as np
from matplotlib import pyplot as plt
import torch as th
from sklearn.svm import SVC
from scipy import ndimage
from scipy.interpolate import Rbf
from tqdm import tqdm
from time import time
from . import kernel


class Perceptron():
    def __init__(self):
        self.support_points = None
    
    def predict(self, point):
        return th.any(th.stack([obs.is_collision(point) for obs in self.obstacles], dim=1), dim=1)
    
    def score(self, point):
        raise NotImplementedError
    
    def line_predict(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.is_collision(p), points))
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    


class DiffCo(Perceptron):
    def __init__(self, kernel_func='rq', gamma=1, beta=1, transform=None, max_batch_size=None, max_num_supports=None):
        super().__init__()
        # self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        self.train_method = None
        self.kernel_func = kernel.RQKernel(gamma) if kernel_func=='rq' else kernel_func
        self.beta = beta
        self.transform = transform
        self._cuda = False

        self.support_points = None
        self.gains = None
        self.hypothesis = None
        self.y = None
        self.distance = None
        self.kernel_matrix = None
        self.rbf_nodes = None
        self.max_batch_size = max_batch_size  # TODO: maximum batch size in inference
        
        # This means the gain, support point, kernel matrix, rbf_nodes, etc. will be truncated to this number
        # or padded with zeros if the number of supports is less than this number
        self.max_num_supports = max_num_supports
        self._valid_supports = 0


    def train(
            self, 
            X, 
            y, 
            update=False,
            exist_mask=None,
            max_iteration=1000,
            method='original', 
            distance=None, 
            verbose=False):
        device = X.device
        if device.type == 'cuda':
            self._cuda = True

        self.train_method = method
        self.distance = distance.reshape(-1) if distance is not None else None
        time_start = time()
        self.train_perceptron(
            X, y, update=update, exist_mask=exist_mask, 
            max_iteration=max_iteration, verbose=verbose
        )
        
        time_elapsed = time() - time_start
        if verbose:
            print(f'DiffCo training done. {time_elapsed:.4f} secs cost')
    
    def filter_support_points_(self, mask):
        self.support_points = self.support_points[mask]
        self.support_transformed = self.support_points if self.transform is None else self.support_transformed[mask]
        self.hypothesis = self.hypothesis[mask]
        self.y = self.y[mask]
        self.distance = self.distance[mask] if self.distance is not None else None
        self.gains = self.gains[mask]
        chosen_idx = th.where(mask)[0]
        if len(chosen_idx) > 10000:
            kernel_matrix_cpu = self.kernel_matrix.cpu()
            chosen_idx = chosen_idx.cpu()
            del self.kernel_matrix
            self.kernel_matrix = kernel_matrix_cpu[chosen_idx[:, None], chosen_idx[None, :]].to(self.support_points.device)
        else:
            self.kernel_matrix = self.kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]

    def train_perceptron(
            self, 
            X, y, 
            update=False,
            exist_mask=None,
            max_iteration=1000,
            verbose=False
        ):
        if update:
            gains, X, X_transformed, kernel_matrix, hypothesis, y = self.jump_start_initialize(X, y, exist_mask)
        else:
            gains, X, X_transformed, kernel_matrix, hypothesis, y = self.initialize(X, y)
        
        if verbose:
            print('DiffCo training...')
            start_time = time()
        for it in tqdm(range(max_iteration), ncols=0, dynamic_ncols=False, disable=not verbose):
            margin = y * hypothesis
            min_margin, min_i = th.min(margin, 0)
            if kernel_matrix[min_i, min_i] == 0:
                kernel_matrix[min_i] = self.kernel_func(X_transformed[min_i], X_transformed)
                kernel_matrix[:, min_i] = kernel_matrix[min_i]
            if min_margin <= 0:
                delta_gain = (self.beta**((1+y[min_i])/2)*y[min_i] - hypothesis[min_i])/kernel_matrix[min_i, min_i]
                gains[min_i] += delta_gain
                hypothesis += delta_gain * kernel_matrix[min_i]
                continue
            
            modified_margin = y * (hypothesis - gains * th.diag(kernel_matrix)) * (gains != 0 )  # 
            max_margin, max_i = th.max(modified_margin, 0)
            if max_margin > 0 and th.sum(gains != 0) > 1:
                hypothesis -= gains[max_i] * kernel_matrix[max_i]
                gains[max_i] = 0
                continue

            break

        if verbose:
            print(f'Ended at iteration {it}, cost {time()-start_time:.4f} secs')
            print('ACC: {}'.format(th.sum((hypothesis > 0) == (y > 0)) / float(len(y))))
        
        mask = gains != 0
        if mask.sum() < 2:
            mask[th.where(mask == 0)[0][0]] = True
        chosen_idx = th.where(mask)[0]
        if self.max_num_supports is None:
            self.support_points = X[mask]
            self.support_transformed = X_transformed[mask]
            self.hypothesis = hypothesis[mask]
            self.y = y[mask]
            self.distance = self.distance[mask] if self.distance is not None else None
            self.gains = gains[mask]
            self.rbf_nodes = self.gains.new_zeros(len(self.gains))
            if len(chosen_idx) > 10000: # to lower the VRAM usage
                kernel_matrix_cpu = kernel_matrix.cpu()
                chosen_idx = chosen_idx.cpu()
                del kernel_matrix
                self.kernel_matrix = kernel_matrix_cpu[chosen_idx[:, None], chosen_idx[None, :]].to(self.support_points.device)
            else:
                self.kernel_matrix = kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]
            self._valid_supports = len(self.support_points)
        else:
            if self.support_points is None:
                self.support_points = th.zeros((self.max_num_supports, X.shape[1]), dtype=X.dtype, device=X.device)
                self.support_transformed = th.zeros(
                    (self.max_num_supports,) + X_transformed.shape[1:], 
                    dtype=X_transformed.dtype, device=X_transformed.device)
                self.hypothesis = th.zeros(self.max_num_supports, dtype=hypothesis.dtype, device=hypothesis.device)
                self.y = th.zeros(self.max_num_supports, dtype=y.dtype, device=y.device)
                distance = self.distance
                self.distance = th.zeros(self.max_num_supports, dtype=self.distance.dtype, device=self.distance.device) if self.distance is not None else None
                self.gains = th.zeros(self.max_num_supports, dtype=gains.dtype, device=gains.device)
                self.kernel_matrix = th.zeros((self.max_num_supports, self.max_num_supports), dtype=kernel_matrix.dtype, device=kernel_matrix.device)
            else:
                distance = self.distance.clone()

            if len(chosen_idx) > self.max_num_supports:
                top_k_idx = th.topk(gains.abs(), self.max_num_supports, largest=False).indices
                mask.zero_()
                mask[top_k_idx] = True
                chosen_idx = th.where(mask)[0]
            self.support_points.zero_()
            self.support_points[:len(chosen_idx)] = X[chosen_idx]
            self.support_transformed.zero_()
            self.support_transformed[:len(chosen_idx)] = X_transformed[chosen_idx]
            self.hypothesis.zero_()
            self.hypothesis[:len(chosen_idx)] = hypothesis[chosen_idx]
            self.y.zero_()
            self.y[:len(chosen_idx)] = y[chosen_idx]
            if self.distance is not None:
                self.distance.zero_()
                self.distance[:len(chosen_idx)] = distance[chosen_idx] if self.distance is not None else None
            self.gains.zero_()
            self.gains[:len(chosen_idx)] = gains[chosen_idx]
            self.rbf_nodes = self.gains.new_zeros(len(self.gains))
            self.kernel_matrix.zero_()
            self.kernel_matrix[:len(chosen_idx), :len(chosen_idx)] = kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]
            self._valid_supports = len(chosen_idx)
            assert th.allclose(self.hypothesis, self.kernel_matrix @ self.gains, atol=1e-4), f"diff: {th.abs(self.hypothesis - self.kernel_matrix @ self.gains).max()}"

    
    @property
    def valid_supports(self):
        return self._valid_supports

    
    def initialize(self, X, y):
        X_transformed = X if self.transform is None else self.transform(X)
        
        if len(X) <= 10000:
            # move to cpu for faster training
            X = X.cpu()
            X_transformed = X_transformed.cpu()
            y = y.cpu()

        y = y.reshape(-1)
        assert len(y) == len(X)
        num_init_points = len(X)
        gains = th.zeros(num_init_points, dtype=X.dtype, device=X.device)
        kernel_matrix = th.zeros((num_init_points, num_init_points), dtype=X.dtype, device=X.device)
        hypothesis = th.zeros(num_init_points, dtype=X.dtype, device=X.device)

        return gains, X, X_transformed, kernel_matrix, hypothesis, y

    def jump_start_initialize(self, X, y, exist_mask):
        num_init_points = len(X)
        if num_init_points <= 10000:
            # move to cpu for faster training
            device = th.device('cpu')
        else:
            device = X.device

        novel_points = X[~exist_mask]
        assert num_init_points - len(novel_points) == self.valid_supports
        # assert th.allclose(X[exist_mask], self.support_points)

        # This may need to happen in the original device as some transformation may not be supported in CPU or GPU
        novel_hypothesis = self.score_original(novel_points).to(device)
        hypothesis = th.zeros(num_init_points, dtype=X.dtype, device=device)
        hypothesis[exist_mask] = self.hypothesis[:self.valid_supports].to(device)
        hypothesis[~exist_mask] = novel_hypothesis
        transformed_novel_points = novel_points if self.transform is None else self.transform(novel_points).to(device)
        
        kernel_matrix = th.zeros((num_init_points, num_init_points), dtype=X.dtype, device=device)
        exist_idx = th.where(exist_mask)[0].to(device)
        novel_idx = th.where(~exist_mask)[0].to(device)
        kernel_matrix[exist_idx[:, None], exist_idx[None, :]] = self.kernel_matrix[:self.valid_supports, :self.valid_supports].to(device)
        # kernel_matrix[novel_idx[:, None], novel_idx[None, :]] = self.kernel_func(transformed_novel_points, transformed_novel_points)
        kernel_matrix[exist_idx[:, None], novel_idx[None, :]] = self.kernel_func(
            self.support_transformed[:self.valid_supports].to(device), transformed_novel_points)
        kernel_matrix[novel_idx[:, None], exist_idx[None, :]] = kernel_matrix[exist_idx[:, None], novel_idx[None, :]].T
        kernel_matrix = kernel_matrix.to(device)

        X_transformed = self.support_transformed.new_zeros((num_init_points,) + transformed_novel_points.shape[1:]).to(device)
        X_transformed[exist_mask] = self.support_transformed[:self.valid_supports].to(device)
        X_transformed[~exist_mask] = transformed_novel_points
        
        y = y.to(device)
        X = X.to(device)

        y = y.reshape(-1)
        
        gains = th.zeros(num_init_points, dtype=X.dtype, device=device)
        gains[exist_mask] = self.gains[:self.valid_supports].to(device)
        # self.gains = gains

        th.cuda.empty_cache()

        # check_score = self.kernel_func(X_transformed, X_transformed) @ gains
        check_score = kernel_matrix @ gains
        assert th.allclose(check_score, hypothesis, atol=1e-4), f"diff: {th.abs(check_score - hypothesis).max()}"
        return gains, X, X_transformed, kernel_matrix, hypothesis, y
    
    def fit_poly(self, kernel_func, target='hypo'):
        X = self.support_transformed
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.rbf_kernel = kernel_func
        kmat = self.rbf_kernel(X[:self.valid_supports], X[:self.valid_supports])

        self.rbf_nodes.zero_()
        self.rbf_nodes[:self.valid_supports] = th.linalg.solve(kmat, y[:self.valid_supports, None]).reshape(-1)
        
        # print(kmat@self.rbf_nodes) # DEBUG
        if self._cuda:
            self.cuda()
    
    def cuda(self):
        self.support_points = self.support_points.cuda()
        if self.transform is not None:
            self.support_transformed = self.support_transformed.cuda()
        self.rbf_nodes = self.rbf_nodes.cuda()
        self.gains = self.gains.cuda()
        self._cuda = True
    
    def to(self, device: th.device):
        self.support_points = self.support_points.to(device)
        if self.transform is not None:
            self.support_transformed = self.support_transformed.to(device)
        if self.rbf_nodes is not None:
            self.rbf_nodes = self.rbf_nodes.to(device)
        self._cuda = device.type == 'cuda'
    
    @property
    def device(self):
        return self.support_points.device
    
    def poly_score(self, point=None, transformed_point=None):
        if transformed_point is None:
            if point.ndim == 1:
                point = point.unsqueeze(0)
            point = point.to(device=self.rbf_nodes.device, dtype=self.rbf_nodes.dtype)
            if self.transform is not None:
                point = self.transform(point)
        else:
            point = transformed_point
        supports = self.support_transformed
        return th.matmul(self.rbf_kernel(point, supports), self.rbf_nodes.unsqueeze(1))
    
    def fit_full_poly(self, epsilon=1, k=2, lmbd=0, target='hypo'):
        X = self.support_transformed
        self.poly_kernel = kernel.Polyharmonic(k=k, epsilon=epsilon)
        phi = self.poly_kernel(X, X)
        phi.fill_diagonal_(lmbd)
        l1 = th.cat([phi, X, th.ones((len(X), 1))], dim=1)
        l2 = th.cat([X.T, th.zeros((X.shape[1], X.shape[1]+1))], dim=1)
        l3 = th.cat([th.ones((1, len(X))), th.zeros(1, X.shape[1]+1)], dim=1)
        # print([l.shape for l in [l1, l2, l3]])
        L = th.cat([l1, l2, l3], dim=0)
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.poly_nodes = th.solve(
            th.cat([y, th.zeros(X.shape[1]+1)], dim=0).reshape(-1, 1),
            L).solution.reshape(-1)
        if self._cuda:
            self.cuda()
    
    def full_poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point).reshape([len(point), -1])
        supports = self.support_transformed
        phi_x = th.cat(
            [self.poly_kernel(point, supports), point, th.ones(len(point), 1)], 
            dim=1)
        if phi_x.shape[0] == 1:
            phi_x = phi_x.squeeze_(0)
        return th.matmul(phi_x, self.poly_nodes)

    def is_collision(self, point):
        return self.score(point) > 0
    
    def score(self, point):
        return self.score_original(point)

    def score_original(self, point):
        # kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point)
        kernel_values = self.kernel_func(point, self.support_transformed)
        score = th.matmul(kernel_values, self.gains)#.unsqueeze_(1))
        return score


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
            th.cat([self.support_points, X[-n_left_out_points:]], dim=0), #self.support_points, #,  th.cat([self.support_points, X[-n_left_out_points:]], dim=0)
            th.cat([self.distance, d[-n_left_out_points:]], dim=0), #self.distance,  #, th.cat([self.distance, d[-n_left_out_points:]], dim=0)
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
        self.kernel_matrix = self.rbf_kernel(X, X)+0.1*th.eye(len(X)) # not considering inter-class
        assert (th.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-th.eye(len(X))).abs().max() < 1e-3, \
            (th.inverse(self.kernel_matrix.transpose(1, 0))@self.kernel_matrix.transpose(1, 0)-th.eye(len(X))).abs().max()

        # self.gains = self.kernel_matrix@self.distance/th.sqrt(self.distance.reshape(1, -1)@self.kernel_matrix.transpose(1, 0)@self.kernel_matrix@self.distance.reshape(-1, 1))
        # self.gains = self.gains.reshape(-1)

        self.gains = th.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
        self.hypothesis = self.kernel_matrix @ self.gains

        # self.gains = self.gains.requires_grad_(True)
        # opt = th.optim.Adam([self.gains], lr=0.001)
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
        #     #         + lda*th.eye(len(self.kernel_matrix))
        #     # sqinv = th.inverse(sq)
        #     # assert (sqinv@sq - th.eye(len(sq))).abs().max() < 1e-4, (sqinv@sq - th.eye(len(sq))).abs().max()
        #     # self.gains = sqinv @ self.kernel_matrix.transpose(1, 0) @ self.distance[:, None]
        #     # self.gains = self.gains.view(-1)
        #     # assert (self.kernel_matrix@self.gains - self.distance).abs().max() < 1e-3, (self.kernel_matrix@self.gains - self.distance).abs().max()
        #     '''
        #     self.gains = th.solve( #@self.kernel_matrix
        #         self.kernel_matrix.transpose(1, 0) @ self.distance[:, None], 
        #         self.kernel_matrix.transpose(1, 0) @self.kernel_matrix\
        #             + lda*th.eye(len(self.kernel_matrix))
        #         ).solution.reshape(-1) #-0.05*th.sign(self.gains[:, None])
            
        #     self.gains = self.gains.data
        #     print('Small gains: ', (self.gains.abs() < gtol).sum())
        #     self.gains[self.gains.abs() < gtol] = 0
        #     self.support_points = self.support_points[self.gains != 0]
        #     self.support_fkine = self.support_fkine[self.gains != 0]
        #     self.hypothesis = self.hypothesis[self.gains != 0]
        #     self.distance = self.distance[self.gains != 0]
        #     self.kernel_matrix = self.kernel_matrix[self.gains != 0][:, self.gains != 0]
        #     '''
            
        #     self.gains = th.solve(self.distance[:, None], self.kernel_matrix).solution.reshape(-1)
        #     # self.gains = self.gains.requires_grad_(True)
        #     # opt = th.optim.Adam([self.gains], lr=0.001)
        self.rbf_nodes = self.gains
        self.hypothesis = self.rbf_score(self.support_points) # self.kernel_matrix@self.gains

        # self.gains = self.gains.data

            

        # for it in tqdm(range(max_iteration)):
        #     abs_err = (self.hypothesis-self.distance).abs()
        #     max_err, max_i = th.max(abs_err, 0)  #1./
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
        #     # max_margin, max_i = th.max(modified_margin, 0)
        #     # if max_margin > 0 and th.sum(self.gains != 0) > 1:
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
        self.gains = th.zeros(num_init_points, dtype=X.dtype)
        # self.kernel_matrix = th.zeros((num_init_points, num_init_points), dtype=X.dtype)
        # self.hypothesis = th.zeros(num_init_points, dtype=X.dtype)
    
    
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

        self.rbf_nodes = th.solve(y[:, None], kmat).solution.reshape(-1)
        # print(kmat@self.rbf_nodes) # DEBUG
    
    def rbf_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.fkine is not None:
            point = self.fkine(point).reshape([len(point), -1])
            supports = self.support_fkine
        else:
            supports = self.support_points
        return th.matmul(self.rbf_kernel(point, supports), self.rbf_nodes.unsqueeze(1))


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
        
        non_zero_weight_cnt = th.sum(self.gains != 0, axis=1)
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
        complete = th.zeros(self.num_class, dtype=th.bool)

        print('MultiDiffCo training...')
        for it in tqdm(range(max_iteration)):
            margin = self.y * self.hypothesis
            
            for c in range(self.num_class):
                c_margin, c_y, c_gain, c_hypo = margin[:, c], self.y[:, c], self.gains[:, c], self.hypothesis[:, c]
                min_margin, min_i = th.min(c_margin, 0)
                if self.kernel_matrix[min_i, min_i] == 0:
                    self.kernel_matrix[min_i] = self.kernel_func(self.support_points[min_i], self.support_points)
                    self.kernel_matrix[:, min_i] = self.kernel_matrix[min_i]
                if min_margin <= 0:
                    delta_gain = (self.beta**((1+c_y[min_i])/2) * c_y[min_i] - c_hypo[min_i])/self.kernel_matrix[min_i, min_i]
                    c_gain[min_i] += delta_gain
                    c_hypo += delta_gain * self.kernel_matrix[min_i]
                    continue
            
                modified_margin = c_y*(c_hypo - c_gain * np.diag(self.kernel_matrix)) * (c_gain != 0 )  # 
                max_margin, max_i = th.max(modified_margin, 0)
                if max_margin > 0 and th.sum(c_gain != 0) > 1:
                    c_hypo -= c_gain[max_i] * self.kernel_matrix[max_i]
                    c_gain[max_i] = 0
                    continue

                complete[c] = True

            if th.min(complete):
                break
        
        print('Ended at iteration {}'.format(it))
        print('ACC: {}'.format(th.sum((self.hypothesis > 0) == (self.y > 0)) / (np.prod(self.y.shape))))

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
            self.gains = th.zeros((num_init_points, self.num_class), dtype=X.dtype)
            self.hypothesis = th.zeros((num_init_points, self.num_class), dtype=X.dtype)
            self.kernel_matrix = th.zeros((num_init_points, num_init_points), dtype=X.dtype)
        elif gains is None or hypothesis is None or kernel_matrix is None:
            raise ValueError('DiffCo: you passed in some existing parameters but not all three of gains, hypothesis, and kernel_matrix')
        else:
            self.gains = gains
            self.hypothesis = hypothesis
            self.kernel_matrix = kernel_matrix
        
    
    def predict(self, point):
        score = self.score(point)
        # max_class_idx = np.argmax(score)
        return (score > 0)*2-1
        # return np.argmax(self.score(point))
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def score(self, points):
        kernel_values = self.kernel_func(points, self.support_points)
        scores = th.matmul(kernel_values, self.gains)
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

        # min_d, min_i = (kmat+th.eye(len(kmat)).fill_diagonal_(float('inf'))).min(dim=0)
        # plt.plot(range(len(kmat)), min_d)
        # plt.show()
        for c in range(self.num_class):
            c_nonzero = th.nonzero(self.gains[:, c])
            c_iszero = th.nonzero(self.gains[:, c] == 0).reshape(-1)
            assert len(c_nonzero) + len(c_iszero) == len(self.gains)
            ridx = c_nonzero.repeat([1, len(c_iszero)]).reshape(-1)
            cidx = c_iszero.repeat([len(c_nonzero), 1]).reshape(-1)
            kmat[ridx, cidx] = 0
            kmat[cidx, ridx] = 0
        self.rbf_nodes = th.linalg.solve(kmat+reg*th.eye(len(kmat), dtype=kmat.dtype), y)
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
        #     class_scores.append(th.matmul(kmat[:, self.gains[:, c] != 0], self.rbf_nodes[self.gains[:, c] != 0, c]))
        # return th.stack(class_scores, dim=1)
        return th.matmul(self.rbf_kernel(point, supports), self.rbf_nodes)
    
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
        l1 = th.cat([phi, X, th.ones((len(X), 1))], dim=1)
        l2 = th.cat([X.T, th.zeros((X.shape[1], X.shape[1]+1))], dim=1)
        l3 = th.cat([th.ones((1, len(X))), th.zeros(1, X.shape[1]+1)], dim=1)
        print([l.shape for l in [l1, l2, l3]])
        L = th.cat([l1, l2, l3], dim=0)
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.poly_nodes = th.solve(
            th.cat([y, th.zeros(X.shape[1]+1, self.num_class)], dim=0),
            L).solution
    
    def poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.fkine is not None:
            point = self.fkine(point).reshape([len(point), -1])
            supports = self.support_fkine
        else:
            supports = self.support_points
        phi_x = th.cat(
            [self.poly_kernel(point, supports), point, th.ones(len(point), 1)], 
            dim=1)
        return th.matmul(phi_x, self.poly_nodes)
    
    # def line_collision(self, start, target, res=50):
    #     points = np.linspace(start, target, res).reshape((1, res, -1))
    #     pair_diff = self.support_points[:, np.newaxis] - points
    #     kernel_values = 1/(1+self.gamma/2*np.sum(pair_diff**2, axis=2))**2
    #     scores = th.matmul(self.gains, kernel_values)
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
            yy, xx = th.meshgrid(th.linspace(-np.pi, np.pi, size[0]), th.linspace(-np.pi, np.pi, size[1]))
            grid_points = th.stack([xx, yy], axis=2).reshape((-1, 2))
            score_spline = dist_est(grid_points).reshape(size+[num_class])
            c_axes = []
            with sns.axes_style('ticks'):
                for cat in range(num_class):
                    c_ax = fig.add_subplot(gs[cat, -1])

                    # score_diffco = checker.score(grid_points).reshape(size)
                    # score = (th.sign(score_diffco)+1)/2*(score_spline-score_spline.min()) + (-th.sign(score_diffco)+1)/2*(score_spline-score_spline.max())
                    score = score_spline[:, :, cat]
                    color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-th.abs(score).max(), vmax=th.abs(score).max())
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

    dataset = th.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
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
    # test_acc = th.sum(test_preds == labels[train_num:], dtype=th.float32)/len(test_preds.view(-1))
    # test_tpr = th.sum(test_preds[labels[train_num:]==1] == 1, dtype=th.float32) / len(test_preds[labels[train_num:]==1])
    # test_tnr = th.sum(test_preds[labels[train_num:]==-1] == -1, dtype=th.float32) / len(test_preds[labels[train_num:]==-1])
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
 

class MultiDimDiffCo(Perceptron):
    def __init__(self, kernel_func='multi_dim_rq', gamma=1, beta=1, transform=None, max_batch_size=None, max_num_supports=None):
        super().__init__()
        # self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        self.train_method = None
        self.kernel_func = kernel.MultiDimRQKernel(gamma) if kernel_func=='multi_dim_rq' else kernel_func
        self.beta = beta
        self.transform = transform
        self._cuda = False

        self.support_points = None
        self.gains = None
        self.hypothesis = None
        self.y = None
        self.distance = None
        self.kernel_matrix = None
        self.rbf_nodes = None
        self.max_batch_size = max_batch_size  # TODO: maximum batch size in inference
        
        # This means the gain, support point, kernel matrix, rbf_nodes, etc. will be truncated to this number
        # or padded with zeros if the number of supports is less than this number
        self.max_num_supports = max_num_supports
        self._valid_supports = 0


    def train(
            self, 
            X, 
            y, 
            update=False,
            exist_mask=None,
            max_iteration=1000,
            method='original', 
            distance=None, 
            verbose=False):
        device = X.device
        if device.type == 'cuda':
            self._cuda = True

        self.train_method = method
        self.distance = distance.reshape(-1) if distance is not None else None
        time_start = time()
        self.train_perceptron(
            X, y, update=update, exist_mask=exist_mask, 
            max_iteration=max_iteration, verbose=verbose
        )
        
        time_elapsed = time() - time_start
        if verbose:
            print(f'DiffCo training done. {time_elapsed:.4f} secs cost')
    
    def filter_support_points_(self, mask):
        self.support_points = self.support_points[mask]
        self.support_transformed = self.support_points if self.transform is None else self.support_transformed[mask]
        self.hypothesis = self.hypothesis[mask]
        self.y = self.y[mask]
        self.distance = self.distance[mask] if self.distance is not None else None
        self.gains = self.gains[mask]
        chosen_idx = th.where(mask)[0]
        if len(chosen_idx) > 10000:
            kernel_matrix_cpu = self.kernel_matrix.cpu()
            chosen_idx = chosen_idx.cpu()
            del self.kernel_matrix
            self.kernel_matrix = kernel_matrix_cpu[chosen_idx[:, None], chosen_idx[None, :]].to(self.support_points.device)
        else:
            self.kernel_matrix = self.kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]

    def train_perceptron(
            self, 
            X, y, 
            update=False,
            exist_mask=None,
            max_iteration=1000,
            verbose=False
        ):
        if update:
            gains, X, X_transformed, kernel_matrix, kernel_valid, hypothesis, y = self.jump_start_initialize(X, y, exist_mask)
        else:
            gains, X, X_transformed, kernel_matrix, kernel_valid, hypothesis, y = self.initialize(X, y)
        
        if verbose:
            print('DiffCo training...')
            start_time = time()
        for it in tqdm(range(max_iteration), ncols=0, dynamic_ncols=False, disable=not verbose):
            margin = y * hypothesis
            min_margin, min_i = th.min(margin, 0)
            # if kernel_matrix[min_i, min_i] == 0:
            if kernel_valid[min_i] == 0:
                # kvalues = self.kernel_func(X_transformed[min_i], X_transformed)
                # print(f'kvalues: {kvalues.shape}, self.kernel_func: {self.kernel_func}')
                kernel_matrix[min_i] = self.kernel_func(X_transformed[min_i], X_transformed)
                kernel_matrix[:, min_i] = kernel_matrix[min_i]
                kernel_valid[min_i] = 1
            if min_margin <= 0:
                # delta_gain = (self.beta**((1+y[min_i])/2)*y[min_i] - hypothesis[min_i])/kernel_matrix[min_i, min_i]
                k_ii = kernel_matrix[min_i, min_i]
                inv_k_ii =  k_ii / k_ii.square().sum()  # th.linalg.pinv(kernel_matrix[min_i, min_i])
                delta_gain = (self.beta**((1+y[min_i])/2)*y[min_i] - hypothesis[min_i]) * inv_k_ii # (C1, C2, ..., Cn)
                gains[min_i] += delta_gain
                delta_gain = delta_gain.flatten()
                k_values = kernel_matrix[min_i].flatten(start_dim=1)
                hypothesis += k_values @ delta_gain
                # check_hypothesis = kernel_matrix.flatten(start_dim=1) @ gains.flatten()
                # assert th.allclose(hypothesis, check_hypothesis, atol=1e-4), f'diff: {th.abs(hypothesis-check_hypothesis).max()}'
                # print(f'check diff: {th.abs(hypothesis-check_hypothesis).max()}')
                # print(f'h[min_i]: {hypothesis[min_i]}, y[min_i]: {y[min_i]}')
                # acc = th.sum((hypothesis > 0) == (y > 0)) / float(len(y))
                # print(f'ACC: {acc:.4f}')
                # assert th.allclose(hypothesis[min_i], y[min_i]), f"h[min_i]: {hypothesis[min_i]}, y[min_i]: {y[min_i]}"
                continue
            
            diag_kernel = kernel_matrix[th.arange(len(kernel_matrix)), th.arange(len(kernel_matrix))]  # (N, C1, C2, ..., Cn)
            diag_kernel = diag_kernel.flatten(start_dim=1)
            flattened_gains = gains.flatten(start_dim=1)
            delta_hypothesis = (diag_kernel * flattened_gains).sum(dim=-1)  # (N)
            non_zero_gain_mask = (flattened_gains != 0).any(dim=-1)
            modified_margin = y * (hypothesis - delta_hypothesis) * non_zero_gain_mask  # (N)
            max_margin, max_i = th.max(modified_margin, 0)
            if max_margin > 0 and th.sum(non_zero_gain_mask) > 1:
                k_values = kernel_matrix[max_i].flatten(start_dim=1)
                hypothesis -= k_values @ flattened_gains[max_i]
                gains[max_i] = 0
                continue

            break

        if verbose:
            print(f'Ended at iteration {it}, cost {time()-start_time:.4f} secs')
            print('ACC: {}'.format(th.sum((hypothesis > 0) == (y > 0)) / float(len(y))))
        
        non_zero_gain_mask = (gains != 0).flatten(start_dim=1).any(dim=-1)
        if non_zero_gain_mask.sum() < 2:
            non_zero_gain_mask[th.where(non_zero_gain_mask == 0)[0][0]] = True
        chosen_idx = th.where(non_zero_gain_mask)[0]
        if self.max_num_supports is None:
            self.support_points = X[non_zero_gain_mask]
            self.support_transformed = X_transformed[non_zero_gain_mask]
            self.hypothesis = hypothesis[non_zero_gain_mask]
            self.y = y[non_zero_gain_mask]
            self.distance = self.distance[non_zero_gain_mask] if self.distance is not None else None
            self.gains = gains[non_zero_gain_mask]
            self.rbf_nodes = self.gains.new_zeros(self.gains.shape)
            if len(chosen_idx) > 10000: # to lower the VRAM usage
                kernel_matrix_cpu = kernel_matrix.cpu()
                chosen_idx = chosen_idx.cpu()
                del kernel_matrix
                self.kernel_matrix = kernel_matrix_cpu[chosen_idx[:, None], chosen_idx[None, :]].to(self.support_points.device)
            else:
                self.kernel_matrix = kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]
            self._valid_supports = len(self.support_points)
        else:
            if self.support_points is None:
                self.support_points = th.zeros((self.max_num_supports, X.shape[1]), dtype=X.dtype, device=X.device)
                self.support_transformed = th.zeros(
                    (self.max_num_supports,) + X_transformed.shape[1:], 
                    dtype=X_transformed.dtype, device=X_transformed.device)
                self.hypothesis = th.zeros(self.max_num_supports, dtype=hypothesis.dtype, device=hypothesis.device)
                self.y = th.zeros(self.max_num_supports, dtype=y.dtype, device=y.device)
                distance = self.distance
                self.distance = th.zeros(self.max_num_supports, dtype=self.distance.dtype, device=self.distance.device) if self.distance is not None else None
                self.gains = th.zeros(self.max_num_supports, dtype=gains.dtype, device=gains.device)
                self.kernel_matrix = th.zeros((self.max_num_supports, self.max_num_supports), dtype=kernel_matrix.dtype, device=kernel_matrix.device)
            else:
                distance = self.distance.clone()

            if len(chosen_idx) > self.max_num_supports:
                top_k_idx = th.topk(gains.abs(), self.max_num_supports, largest=False).indices
                mask.zero_()
                mask[top_k_idx] = True
                chosen_idx = th.where(mask)[0]
            self.support_points.zero_()
            self.support_points[:len(chosen_idx)] = X[chosen_idx]
            self.support_transformed.zero_()
            self.support_transformed[:len(chosen_idx)] = X_transformed[chosen_idx]
            self.hypothesis.zero_()
            self.hypothesis[:len(chosen_idx)] = hypothesis[chosen_idx]
            self.y.zero_()
            self.y[:len(chosen_idx)] = y[chosen_idx]
            if self.distance is not None:
                self.distance.zero_()
                self.distance[:len(chosen_idx)] = distance[chosen_idx] if self.distance is not None else None
            self.gains.zero_()
            self.gains[:len(chosen_idx)] = gains[chosen_idx]
            self.rbf_nodes = self.gains.new_zeros(len(self.gains))
            self.kernel_matrix.zero_()
            self.kernel_matrix[:len(chosen_idx), :len(chosen_idx)] = kernel_matrix[chosen_idx[:, None], chosen_idx[None, :]]
            self._valid_supports = len(chosen_idx)
            assert th.allclose(self.hypothesis, self.kernel_matrix @ self.gains, atol=1e-4), f"diff: {th.abs(self.hypothesis - self.kernel_matrix @ self.gains).max()}"

    
    @property
    def valid_supports(self):
        return self._valid_supports

    
    def initialize(self, X, y):
        X_transformed = X if self.transform is None else self.transform(X)
        
        if len(X) <= 10000:
            # move to cpu for faster training
            X = X.cpu()
            X_transformed = X_transformed.cpu()
            y = y.cpu()

        y = y.reshape(-1)
        assert len(y) == len(X)
        num_init_points = len(X)
        gains = th.zeros_like(X_transformed)
        kernel_matrix = th.zeros((num_init_points, num_init_points) + X_transformed.shape[1:], dtype=X.dtype, device=X.device)
        kernel_valid = th.zeros(num_init_points, dtype=th.bool, device=X.device)
        hypothesis = th.zeros(num_init_points, dtype=X.dtype, device=X.device)

        return gains, X, X_transformed, kernel_matrix, kernel_valid, hypothesis, y

    def jump_start_initialize(self, X, y, exist_mask):
        num_init_points = len(X)
        if num_init_points <= 10000:
            # move to cpu for faster training
            device = th.device('cpu')
        else:
            device = X.device

        novel_points = X[~exist_mask]
        assert num_init_points - len(novel_points) == self.valid_supports
        # assert th.allclose(X[exist_mask], self.support_points)

        # This may need to happen in the original device as some transformation may not be supported in CPU or GPU
        novel_hypothesis = self.score_original(novel_points).to(device)
        hypothesis = th.zeros(num_init_points, dtype=X.dtype, device=device)
        hypothesis[exist_mask] = self.hypothesis[:self.valid_supports].to(device)
        hypothesis[~exist_mask] = novel_hypothesis
        transformed_novel_points = novel_points if self.transform is None else self.transform(novel_points).to(device)
        
        kernel_matrix = th.zeros((num_init_points, num_init_points), dtype=X.dtype, device=device)
        exist_idx = th.where(exist_mask)[0].to(device)
        novel_idx = th.where(~exist_mask)[0].to(device)
        kernel_matrix[exist_idx[:, None], exist_idx[None, :]] = self.kernel_matrix[:self.valid_supports, :self.valid_supports].to(device)
        # kernel_matrix[novel_idx[:, None], novel_idx[None, :]] = self.kernel_func(transformed_novel_points, transformed_novel_points)
        kernel_matrix[exist_idx[:, None], novel_idx[None, :]] = self.kernel_func(
            self.support_transformed[:self.valid_supports].to(device), transformed_novel_points)
        kernel_matrix[novel_idx[:, None], exist_idx[None, :]] = kernel_matrix[exist_idx[:, None], novel_idx[None, :]].T
        kernel_matrix = kernel_matrix.to(device)

        X_transformed = self.support_transformed.new_zeros((num_init_points,) + transformed_novel_points.shape[1:]).to(device)
        X_transformed[exist_mask] = self.support_transformed[:self.valid_supports].to(device)
        X_transformed[~exist_mask] = transformed_novel_points
        
        y = y.to(device)
        X = X.to(device)

        y = y.reshape(-1)
        
        gains = th.zeros(num_init_points, dtype=X.dtype, device=device)
        gains[exist_mask] = self.gains[:self.valid_supports].to(device)
        # self.gains = gains

        th.cuda.empty_cache()

        # check_score = self.kernel_func(X_transformed, X_transformed) @ gains
        check_score = kernel_matrix @ gains
        assert th.allclose(check_score, hypothesis, atol=1e-4), f"diff: {th.abs(check_score - hypothesis).max()}"
        return gains, X, X_transformed, kernel_matrix, hypothesis, y
    
    def fit_poly(self, kernel_func, target='hypo'):
        X = self.support_transformed
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.rbf_kernel = kernel_func
        kmat = self.rbf_kernel(X[:self.valid_supports], X[:self.valid_supports])
        # kmat = kmat.flatten(start_dim=2)[:, :, 0].flatten(start_dim=1)#.sum(dim=-1)
        kmat = kmat.flatten(start_dim=1)
        print(f'kmat: {kmat}')

        self.rbf_nodes.zero_()
        # self.rbf_nodes = self.rbf_nodes.new_zeros(self.rbf_nodes.shape[0])
        # th.linalg.solve(kmat, y[:self.valid_supports, None], out=self.rbf_nodes[:self.valid_supports].view(-1, 1)).reshape(-1, *self.rbf_nodes.shape[1:])
        # sol = self.rbf_nodes[:self.valid_supports].flatten().unsqueeze(1)
        # assert kmat.shape[0] == kmat.shape[1]
        ret = th.linalg.lstsq(kmat, y[:self.valid_supports, None], driver='gelsd')
        sol = ret.solution
        # # sol = kmat.pinverse() @ y[:self.valid_supports, None]
        self.rbf_nodes[:self.valid_supports] = sol.reshape(-1, *self.rbf_nodes.shape[1:])
        print(f'sol = {sol}')
        print(f'residuals: {ret.residuals}')
        print(f'singular_values: {ret.singular_values}')
        print(f'rank: {ret.rank}')
        # check_score = th.matmul(kmat, sol).flatten()
        check_score = self.poly_score(transformed_point=X[:self.valid_supports]).flatten()
        assert check_score.shape == y[:self.valid_supports].shape, f"check_score: {check_score.shape}, y: {y[:self.valid_supports].shape}"
        print(f'check_score: {check_score[:50]}, y: {y[:self.valid_supports][:50]}')
        diff = th.abs(check_score - y[:self.valid_supports])
        print(f'diff: {diff.max()}, {diff.mean()}')
        # assert th.allclose(check_score, y[:self.valid_supports], atol=1e-4), f"diff: max {diff.max()}, mean {diff.mean()}"
        
        print(f'sol.shape: {sol.shape}, rbf_nodes.shape: {self.rbf_nodes[:self.valid_supports].shape}')
        
        # check_score = th.matmul(kmat, self.rbf_nodes[:self.valid_supports].flatten())
        
        
        
        # print(kmat@self.rbf_nodes) # DEBUG
        if self._cuda:
            self.cuda()
    
    def cuda(self):
        self.support_points = self.support_points.cuda()
        if self.transform is not None:
            self.support_transformed = self.support_transformed.cuda()
        self.rbf_nodes = self.rbf_nodes.cuda()
        self.gains = self.gains.cuda()
        self._cuda = True
    
    def to(self, device: th.device):
        self.support_points = self.support_points.to(device)
        if self.transform is not None:
            self.support_transformed = self.support_transformed.to(device)
        if self.rbf_nodes is not None:
            self.rbf_nodes = self.rbf_nodes.to(device)
        self._cuda = device.type == 'cuda'
    
    @property
    def device(self):
        return self.support_points.device
    
    def poly_score(self, point=None, transformed_point=None):
        if transformed_point is None:
            if point.ndim == 1:
                point = point.unsqueeze(0)
            point = point.to(device=self.rbf_nodes.device, dtype=self.rbf_nodes.dtype)
            if self.transform is not None:
                point = self.transform(point)
        else:
            point = transformed_point
        supports = self.support_transformed
        k_values = self.rbf_kernel(point, supports).flatten(start_dim=1)
        rbf_nodes = self.rbf_nodes.flatten().unsqueeze(1)
        return th.matmul(k_values, rbf_nodes)
        # return th.matmul(self.rbf_kernel(point, supports), self.rbf_nodes.unsqueeze(1))
    
    def fit_full_poly(self, epsilon=1, k=2, lmbd=0, target='hypo'):
        X = self.support_transformed
        self.poly_kernel = kernel.Polyharmonic(k=k, epsilon=epsilon)
        phi = self.poly_kernel(X, X)
        phi.fill_diagonal_(lmbd)
        l1 = th.cat([phi, X, th.ones((len(X), 1))], dim=1)
        l2 = th.cat([X.T, th.zeros((X.shape[1], X.shape[1]+1))], dim=1)
        l3 = th.cat([th.ones((1, len(X))), th.zeros(1, X.shape[1]+1)], dim=1)
        # print([l.shape for l in [l1, l2, l3]])
        L = th.cat([l1, l2, l3], dim=0)
        if target == 'hypo':
            y = self.hypothesis
        elif 'dist' in target:
            y = self.distance
        elif 'label' in target:
            y = self.y
        self.poly_nodes = th.solve(
            th.cat([y, th.zeros(X.shape[1]+1)], dim=0).reshape(-1, 1),
            L).solution.reshape(-1)
        if self._cuda:
            self.cuda()
    
    def full_poly_score(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point).reshape([len(point), -1])
        supports = self.support_transformed
        phi_x = th.cat(
            [self.poly_kernel(point, supports), point, th.ones(len(point), 1)], 
            dim=1)
        if phi_x.shape[0] == 1:
            phi_x = phi_x.squeeze_(0)
        return th.matmul(phi_x, self.poly_nodes)

    def is_collision(self, point):
        return self.score(point) > 0
    
    def score(self, point):
        return self.score_original(point)

    def score_original(self, point):
        # kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.transform is not None:
            point = self.transform(point)
        kernel_values = self.kernel_func(point, self.support_transformed)
        score = th.matmul(kernel_values, self.gains)#.unsqueeze_(1))
        return score