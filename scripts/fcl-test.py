import numpy as np
import fcl

R = np.array([[0.0, -1.0, 0.0],
              [1.0,  0.0, 0.0],
              [0.0,  0.0, 1.0]])
T = np.array([1.0, 1.865, 0])

g1 = fcl.Box(1,2,3)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

g2 = fcl.Cone(1,3)
t2 = fcl.Transform(R, T)
o2 = fcl.CollisionObject(g2, t2)

# request = fcl.DistanceRequest(gjk_solver_type=fcl.GJKSolverType.GST_INDEP)
# result = fcl.DistanceResult()
request = fcl.CollisionRequest(enable_contact=True)
result = fcl.CollisionResult()

# ret = fcl.distance(o1, o2, request, result)
ret = fcl.collide(o1, o2, request, result)

print(ret, result.contacts[0].penetration_depth)