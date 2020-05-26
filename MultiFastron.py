import numpy as np
from Fastron import CollisionChecker, Obstacle, plt

class MultiChecker:
    def __init__(self, objects: [Obstacle,]):
        self.objects = objects
    
    def predict(self, point):
        return np.fromiter(map(lambda obj: 1 if obj.is_collision(point) else -1, self.objects), int)
    
    def line_collision(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.predict(p), points))

class MultiFastron(MultiChecker):
    def __init__(self, objects, num_class=None, gamma=1, beta=1, gt_checker=None):
        super().__init__(objects)
        self.gt_checker = gt_checker if gt_checker is not None else MultiChecker(self.objects)
        self.num_class = len(objects) if num_class is None else num_class
        self.gamma = gamma
        self.beta = beta
        self.beta = beta
        self.initialize(1000)

    def initialize(self, num_init_points=100):
        self.support_points = np.random.random((num_init_points, 2)) * 10
        self.gains = np.zeros((self.num_class, num_init_points))
        K = np.tile(self.support_points[np.newaxis, :], (num_init_points, 1, 1))
        self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        self.hypothesis = self.gains@self.kernel_matrix
        self.max_n_support = 200
    
    def train_sgd(self, max_iteration=1000):
        pass
    
    def train(self, max_iteration=1000, method='original'):
        if method == 'original':
            self.train_original(max_iteration)
        elif method == 'sgd':
            self.train_sgd(max_iteration)
    
    def train_original(self, max_iteration=1000):
        self.y = np.array([self.gt_checker.predict(p) for p in self.support_points]).transpose()

        print('Fastron training...')
        for it in range(max_iteration):
            margin = self.y * self.hypothesis
            for class_margin, class_y, class_gain, class_hypo in zip(margin, self.y, self.gains, self.hypothesis):
                min_margin_idx = np.argmin(class_margin)
                if class_margin[min_margin_idx] <= 0:
                    delta_gain = self.beta**((1+class_y[min_margin_idx])/2) * class_y[min_margin_idx] - class_hypo[min_margin_idx]
                    class_gain[min_margin_idx] += delta_gain
                    class_hypo += delta_gain * self.kernel_matrix[min_margin_idx]
        valid = np.sum(self.gains != 0, axis=0)
        self.support_points = self.support_points[valid != 0]
        self.hypothesis = self.hypothesis[:, valid != 0]
        self.gains = self.gains[:, valid != 0]
        print('Training done')
    
    def predict(self, point):
        score = self.score(point)
        max_class_idx = np.argmax(score)
        return max_class_idx+1 if score[max_class_idx] > 0 else 0
        # return np.argmax(self.score(point))

    def score(self, point):
        kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        score = self.gains@kernel_values
        return score
    
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

        plt.show()


if __name__ == '__main__':
    obstacles = [
        ('circle', (2, 1), 1.5),
        ('rect', (5, 7), (5, 3))]
    obstacles = [Obstacle(*param) for param in obstacles]

    np.random.seed(1314)
    classifier = MultiFastron(obstacles, len(obstacles), gamma=1, beta=1)
    classifier.train(1000)
    print(classifier.gains, classifier.gains.size)
    classifier.vis(200)