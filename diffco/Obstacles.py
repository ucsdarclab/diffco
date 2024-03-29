import torch
import fcl
import numpy as np

class Obstacle:
    def __init__(self, kind, position, size, cost=np.inf):
        self.kind = kind
        if self.kind not in ['circle', 'rect']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = torch.FloatTensor(position)
        self.size = torch.FloatTensor([size]) if kind == 'circle' else torch.FloatTensor(size)
        self.cost = cost
    
    def is_collision(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.kind == 'circle':
            return torch.norm(self.position-point, dim=1) < self.size/2
        elif self.kind == 'rect':
            return torch.all(torch.abs(self.position-point) < self.size/2, dim=1)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    
    def get_cost(self):
        return self.cost

class FCLObstacle:
    def __init__(self, shape, position, size=None, category=None, **kwargs):
        self.size = size
        self.position = position
        if shape == 'circle':
            pos_3d = torch.DoubleTensor([position[0], position[1], 0])
            self.geom = fcl.Cylinder(size, 1000)
        elif shape == 'rect':
            pos_3d = torch.DoubleTensor([position[0], position[1], 0])
            self.geom = fcl.Box(size[0], size[1], 1000)
        elif shape == 'mesh':
            pos_3d = torch.DoubleTensor(position)
            self.geom = kwargs['geom']

        self.cobj = fcl.CollisionObject(self.geom, fcl.Transform(pos_3d))
        self.category = category

class Simple1DDynamicObstacle:
    def __init__(self, size, position_func):
        self.size = size
        self.position_func = position_func

    def is_collision(self, st_point, distance=True):
        ''' st_point is the query point (joints, time), p is the center of the obstacle
            distance indicates whether to return collision distance
        '''

        p = self.position_func(st_point[:, -1:])
        d = self.size/2 - torch.abs(st_point[:, :-1]-p)
        in_collision = d > 0 # point >= p-self.size/2 and point <= p+self.size/2
        # if in_collision:
        #     d = torch.minimum(point - p + self.size/2, p + self.size/2 - point)
        # else:
        #     d = -torch.minimum(torch.abs(point-p+self.size/2), torch.abs(point - p - self.size/2))
        if distance:
            return in_collision, d
        else:
            return in_collision

class ObstacleMotion:
    def predict(self, t):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class LinearMotion(ObstacleMotion):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def predict(self, t):
        return self.A * t + self.B

class SineMotion(ObstacleMotion):
    def __init__(self, A, alpha, beta, bias):
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
    
    def predict(self, t):
        return self.A*torch.sin(self.alpha*t+self.beta) + self.bias

# class ComposeMotion # TODO