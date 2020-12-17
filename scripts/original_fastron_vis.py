import sys
sys.path.append('/home/yuheng/DiffCo/')
from diffco.DiffCo import *

# from import kernel
obstacles = [
    ('circle', (6, 2), 2),
    # ('circle', (2, 7), 1),
    ('rect', (3.5, 6), (2, 1)),
    ('rect', (4, 7), (1, 1)),
    ('rect', (5, 8), (10, 1)),
    ('rect', (7.5, 6), (2, 1)),
    ('rect', (8, 7), (1, 1)),]
obstacles = [Obstacle(*param) for param in obstacles]
# kernel = kernel.CauchyKernel(100)
# k = kernel.TangentKernel(0.8, 0)
k = kernel.RQKernel(20)
# k = kernel.MultiQuadratic(0.7)
# lambda x, x_prime: -k(x, x_prime)+k(np.array([0, 0]), np.array([[10, 10]]))
checker = DiffCo(obstacles, kernel_func=k, beta=1)
vis(checker, 400, seed=1917)