from Fastronpp.Fastron import Fastron, Obstacle
from Fastronpp import kernel
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import Rbf

def plot_score(checker, score, n, i):
    ax = plt.subplot(1, n, i)
    c = ax.pcolormesh(xx, yy, score, cmap='RdBu_r', vmin=-np.abs(score).max(), vmax=np.abs(score).max())
    ax.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
    ax.contour(xx, yy, (score).astype(float), levels=0)
    ax.axis('equal')
    fig.colorbar(c, ax=ax)
    sparse_score = score[::10, ::10]
    score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    ax.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=20, color='red')

    for obs in checker.obstacles:
        if obs.kind == 'circle':
            artist = plt.Circle(obs.position, radius=obs.size/2, color=[0, 0, 0, 0.3])            
        elif obs.kind == 'rect':
            artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color=[0, 0, 0, 0.3])
        else:
            raise NotImplementedError('Unknown obstacle type')
        ax.add_artist(artist)

if __name__ == "__main__":
    # obstacles = [
    #     ('circle', (6, 2), 1.5),
    #     ('rect', (2, 6), (1.5, 1.5))]
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
    checker = Fastron(obstacles, kernel_func=k, beta=20)

    np.random.seed(1917)
    checker.initialize(3000)
    checker.train(200000, method='original')
    
    rbfi = Rbf(checker.support_points[:, 0], checker.support_points[:, 1], checker.hypothesis)#, epsilon=10) #checker.y) #, checker.hypothesis)
    # print(checker.support_points[:, 0], checker.support_points[:, 1], checker.hypothesis)
    # print(rbfi.__dict__)
    fig, ax = plt.subplots(figsize=(21, 7))
    size = [150, 150]
    xx, yy = np.meshgrid(np.linspace(0, 10, size[0]), np.linspace(0, 10, size[1]))
    grid_points = np.stack([xx, yy], axis=2).reshape((-1, 2))

    score_spline = rbfi(grid_points[:, 0], grid_points[:, 1]).reshape(size)
    score_fastron = checker.score(grid_points).reshape(size)
    sigma = 0.001
    # comb_score = score_spline
    # comb_score = (1-np.exp(-(score_fastron/sigma)**2))*(score_spline) #-score_spline.min()
    # comb_score = np.sign(score_fastron) * (score_spline-score_spline.min())
    comb_score = (np.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-np.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
    
    # ax1 = plt.subplot(121)
    # c = ax1.pcolormesh(xx, yy, comb_score, cmap='RdBu_r', vmin=-np.abs(score_spline).max(), vmax=np.abs(score_spline).max())
    # ax1.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
    # ax1.contour(xx, yy, (score_spline).astype(float), levels=0)
    # ax1.axis('equal')
    # fig.colorbar(c, ax=ax1)
    # sparse_score = score_spline[::10, ::10]
    # score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    # score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    # ax1.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=40, color='red')
    # ax1.set_title('{}'.format(rbfi.epsilon))
    plot_score(checker, comb_score, 3, 1)

    # ax2 = plt.subplot(122)
    # c = ax2.pcolormesh(xx, yy, score_fastron, cmap='RdBu_r', vmin=-np.abs(score_fastron).max(), vmax=np.abs(score_fastron).max())
    # ax2.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
    # ax2.contour(xx, yy, (score_fastron).astype(float), levels=0)
    # ax2.axis('equal')
    # fig.colorbar(c, ax=ax2)
    # sparse_score = score_fastron[::10, ::10]
    # score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    # score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    # ax2.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=40, color='red')
    # ax2.set_title('{}'.format(rbfi.epsilon))
    plot_score(checker, score_fastron, 3, 2)
    plot_score(checker, score_spline, 3, 3)
    plt.show()
