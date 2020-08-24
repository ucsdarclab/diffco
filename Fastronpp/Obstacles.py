import torch
import fcl
import numpy as np

class Obstacle:
    def __init__(self, kind, position, size, cost=np.inf):
        self.kind = kind
        if self.kind not in ['circle', 'rect']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = torch.tensor(position, dtype=torch.float32)
        self.size = torch.tensor(size, dtype=torch.float32) if kind != 'rect' or (isinstance(size, (list, tuple, np.ndarray)) and len(size) == len(position)) else torch.tensor([size, size], dtype=torch.float32)
        self.cost = cost
    
    def is_collision(self, point):
        if self.kind == 'circle':
            return torch.norm(self.position-point) < self.size/2
        elif self.kind == 'rect':
            return torch.all(torch.abs(self.position-point) < self.size/2)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    
    def get_cost(self):
        return self.cost

def FCLObstacle(kind, position, size):
    if kind == 'circle':
        position = torch.FloatTensor([position[0], position[1], 0])
        return fcl.CollisionObject(fcl.Cylinder(size, 1000), fcl.Transform(position))
    elif kind == 'rect':
        position = torch.FloatTensor([position[0], position[1], 0])
        return fcl.CollisionObject(fcl.Box(size[0], size[1], 1000), fcl.Transform(position))