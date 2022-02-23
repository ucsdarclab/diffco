import numpy as np
import fcl
import torch
from scipy.spatial.transform import Rotation

# R = np.array([[0.0, -1.0, 0.0],
#               [1.0,  0.0, 0.0],
#               [0.0,  0.0, 1.0]])
# R = np.array([[1.0, 0.0, 0.0],
#               [0.0,  1.0, 0.0],
#               [0.0,  0.0, 1.0]])
# T = np.array([1.0, 1.865, 0])

g1 = fcl.Box(10,20,1000)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

# g2 = fcl.Cone(1,3)
# g2 = fcl.Cylinder(0.01, 1000)
g2 = fcl.Box(10,20,1000)
t2 = fcl.Transform( Rotation.from_rotvec([0., 0., np.pi/4]).as_quat()[[3,0,1,2]], [10., 0., 0.])
o2 = fcl.CollisionObject(g2, t2)

request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)#, gjk_solver_type=fcl.GJKSolverType.GST_LIBCCD)
result = fcl.DistanceResult()
# request.enable_signed_distance = True
# request.distance_tolerance = 0.01
# request = fcl.CollisionRequest(enable_contact=True)
# request.enable_nearest_points = True
# result = fcl.CollisionResult()

ret = fcl.distance(o1, o2, request, result)
# ret = fcl.collide(o1, o2, request, result)

# size = 50, 50
# yy, xx = torch.meshgrid(torch.linspace(-5, 5, size[0]), torch.linspace(-5, 5, size[1]))
# grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
# grid_labels = torch.zeros_like(grid_points)[:, 0]
# for i, (x, y) in enumerate(grid_points):
#     print(x, y)
#     o2.setTranslation([x, y, 0])
#     # fcl.update()
#     # result.clear()
#     result = fcl.CollisionResult()
#     ret = fcl.collide(o1, o2, request, result)
#     grid_labels[i] = result.is_collision
#     print(result.is_collision)
# print(grid_labels)
# import matplotlib.pyplot as plt
# plt.scatter(grid_points[grid_labels==False, 0], grid_points[grid_labels==False, 1], c='g')
# plt.scatter(grid_points[grid_labels==True, 0], grid_points[grid_labels==True, 1], c='r')
# plt.show()

print(result.nearest_points) #, result.contacts[0].penetration_depth)
print(result.min_distance) #, result.contacts[0].penetration_depth)