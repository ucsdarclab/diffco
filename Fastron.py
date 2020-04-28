import numpy as np
from matplotlib import pyplot as plt

class Obstacle:
    def __init__(self, kind, position, size):
        self.kind = kind
        if self.kind not in ['circle', 'rect']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = np.array(position)
        self.size = np.array(size)
    
    def is_collision(self, point):
        if self.kind == 'circle':
            return np.linalg.norm(self.position-point) < self.size/2
        elif self.kind == 'rect':
            return np.all(np.abs(self.position-point) < self.size/2)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))

class CollisionChecker():
    def __init__(self, obstacles):
        self.obstacles = obstacles
    
    def is_collision(self, point):
        return any(map(lambda obs: obs.is_collision(point), self.obstacles))
    
    def line_collision(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.is_collision(p), points))
    
    def predict(self, X):
        return np.array(list(map(lambda x: self.is_collision(x), X)))

class Fastron(CollisionChecker):
    def __init__(self, obstacles, gt_checker=None):
        super().__init__(obstacles)
        self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        self.initialize()
    
    def initialize(self):
        self.support_points = np.random.random((1000, 2)) * 10
        self.gains = np.zeros(1000)
        self.gamma = 30
        K = np.tile(self.support_points[np.newaxis, :], (1000, 1, 1))
        self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        self.hypothesis = self.gains@self.kernel_matrix
        self.max_n_support = 200
        self.conditional_bias = 1

    def train(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        
        print('Fastron training...')
        for it in range(max_iteration):
            margin = self.y * self.hypothesis
            min_margin_idx = np.argmin(margin)
            if margin[min_margin_idx] <= 0:
                delta_gain = self.conditional_bias**((1+self.y[min_margin_idx])/2) * self.y[min_margin_idx] - self.hypothesis[min_margin_idx]
                self.gains[min_margin_idx] += delta_gain
                self.hypothesis += delta_gain * self.kernel_matrix[min_margin_idx]
        print('Training done')


    def is_collision(self, point):
        return self.score(point) > 0
    
    def score(self, point):
        kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        score = self.gains@kernel_values
        return score
    
    def vis(self, size=100):
        if isinstance(size, int):
            size = [size, size]
        xx, yy = np.meshgrid(np.linspace(0, 10, size[0]), np.linspace(0, 10, size[1]))
        grid_points = np.stack([xx, yy], axis=2).reshape((-1, 2))
        grid_score = np.array(list(map(self.score, grid_points))).reshape((size[0], size[1]))
        non_zero_gains = list(filter(lambda a: a != 0, self.gains))
        real_support_points = self.support_points[self.gains != 0]

        fig, ax = plt.subplots()
        ax.set_title('number of support points = {}'.format(len(real_support_points)))
        c = ax.pcolormesh(xx, yy, grid_score, cmap='RdBu', vmin=-2, vmax=2)
        plt.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
        plt.contour(xx, yy, grid_score, levels=1)
        plt.axis('equal')
        fig.colorbar(c, ax=ax)
        plt.show()

if __name__ == '__main__':
    obstacles = [
        ('circle', (2, 1), 1.5),
        ('rect', (5, 7), (5, 3))]
    obstacles = [Obstacle(*param) for param in obstacles]
    checker = Fastron(obstacles)
    checker.train(1000)
    # plt.axis('equal')
    # plt.xlim((0, 10))
    # plt.ylim((0, 10))
    # plot_decision_regions(X=checker.support_points, y=checker.y.astype(np.integer), clf=checker, markers=[None], legend=None, filler_feature_ranges=[(0, 10), (0, 10)])
    # plt.show()
    # print(checker.support_points, checker.gains)
    checker.vis(200)
