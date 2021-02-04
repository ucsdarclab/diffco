import numpy as np
import fcl
import torch

# R = np.array([[0.0, -1.0, 0.0],
#               [1.0,  0.0, 0.0],
#               [0.0,  0.0, 1.0]])
R = np.array([[1.0, 0.0, 0.0],
              [0.0,  1.0, 0.0],
              [0.0,  0.0, 1.0]])
T = np.array([1.0, 1.865, 0])

g1 = fcl.Box(1,2,3)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

# g2 = fcl.Cone(1,3)
g2 = fcl.Cylinder(0.01, 1000)
t2 = fcl.Transform()
o2 = fcl.CollisionObject(g2, t2)

# request = fcl.DistanceRequest(gjk_solver_type=fcl.GJKSolverType.GST_INDEP)
# result = fcl.DistanceResult()
request = fcl.CollisionRequest(enable_contact=True)
result = fcl.CollisionResult()

# ret = fcl.distance(o1, o2, request, result)
# ret = fcl.collide(o1, o2, request, result)

size = 50, 50
yy, xx = torch.meshgrid(torch.linspace(-5, 5, size[0]), torch.linspace(-5, 5, size[1]))
grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
grid_labels = torch.zeros_like(grid_points)[:, 0]
for i, (x, y) in enumerate(grid_points):
    print(x, y)
    o2.setTranslation([x, y, 0])
    fcl.update()
    ret = fcl.collide(o1, o2, request, result)
    grid_labels[i] = result.is_collision
    print(result.is_collision)

import matplotlib.pyplot as plt
plt.scatter(grid_points[grid_labels==True, 0], grid_points[grid_labels==True, 1])
plt.show()

# print(ret, result.contacts[0].penetration_depth)