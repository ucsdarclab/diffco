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
    def __init__(self, shape, position, size, category=None):
        self.size = size
        self.position = position
        if shape == 'circle':
            pos_3d = torch.FloatTensor([position[0], position[1], 0])
            self.geom = fcl.Cylinder(size, 1000)
        elif shape == 'rect':
            pos_3d = torch.FloatTensor([position[0], position[1], 0])
            self.geom = fcl.Box(size[0], size[1], 1000)

        self.cobj = fcl.CollisionObject(self.geom, fcl.Transform(pos_3d))
        self.category = category