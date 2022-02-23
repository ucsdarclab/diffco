import numpy as np
import fcl
from scipy.spatial.transform import Rotation

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 1.0, 3.0])
v3 = np.array([3.0, 2.0, 1.0])
x, y, z = 1.0, 2.0, 3.0
rad, lz = 1.0, 3.0
n = np.array([1.0, 0.0, 0.0])
d = 5.0

t = fcl.TriangleP(v1, v2, v3) # Triangle defined by three points
b = fcl.Box(x, y, z)          # Axis-aligned box with given side lengths
s = fcl.Sphere(rad)           # Sphere with given radius
e = fcl.Ellipsoid(x, y, z)    # Axis-aligned ellipsoid with given radii
# c = fcl.Capsule(rad, lz)      # Capsule with given radius and height along z-axis
# c = fcl.Cone(rad, lz)         # Cone with given radius and cylinder height along z-axis
c = fcl.Cylinder(rad, lz)     # Cylinder with given radius and height along z-axis
h = fcl.Halfspace(n, d)       # Half-space defined by {x : <n, x> < d}
p = fcl.Plane(n, d)           # Plane defined by {x : <n, x> = d}

verts = np.array([[1.0, 1.0, 1.0],
                  [2.0, 1.0, 1.0],
                  [1.0, 2.0, 1.0],
                  [1.0, 1.0, 2.0]])
tris  = np.array([[0,2,1],
                  [0,3,2],
                  [0,1,3],
                  [1,2,3]])

m = fcl.BVHModel()
m.beginModel(len(verts), len(tris))
m.addSubModel(verts, tris)
m.endModel()

objs1 = [fcl.CollisionObject(b, fcl.Transform(Rotation.from_euler('XYZ', [0, 0, np.arccos(2.0/3)-np.arctan(0.5)]).as_quat()[[3,0,1,2]], [1.5, 0, 0]))] #, fcl.CollisionObject(s)] np.arccos(2.0/3) Rotation.from_euler('XYZ', [0, 0, np.pi/2]).as_quat()[[3,0,1,2]], [1.5, 0, 0]
print(Rotation.from_rotvec([0, 0, np.arccos(2.0/3)]).as_quat()[[3, 0, 1, 2]])
objs2 = [fcl.CollisionObject(c)] #, fcl.CollisionObject(m)]

manager1 = fcl.DynamicAABBTreeCollisionManager()
manager2 = fcl.DynamicAABBTreeCollisionManager()

manager1.registerObjects(objs1)
manager2.registerObjects(objs2)

manager1.setup()
manager2.setup()

# #=====================================================================
# # Managed internal (sub-n^2) collision checking
# #=====================================================================
# cdata = fcl.CollisionData()
# manager1.collide(cdata, fcl.defaultCollisionCallback)
# print('Collision within manager 1?: {}'.format(cdata.result.is_collision))

# #=====================================================================
# # Managed internal (sub-n^2) distance checking
# #=====================================================================
# ddata = fcl.DistanceData()
# manager1.distance(ddata, fcl.defaultDistanceCallback)
# print('Closest distance within manager 1?: {}'.format(ddata.result.min_distance))

#=====================================================================
# Managed one to many collision checking
#=====================================================================
# req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
# rdata = fcl.CollisionData(request = req)

# manager1.collide(fcl.CollisionObject(m), rdata, fcl.defaultCollisionCallback)
# print('Collision between manager 1 and Mesh?: {}'.format(rdata.result.is_collision))
# print('Contacts:')
# for c in rdata.result.contacts:
#     print('\tO1: {}, O2: {}, depth: {}'.format(c.o1, c.o2, c.penetration_depth))

#=====================================================================
# Managed many to many collision checking
#=====================================================================
# req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
# rdata = fcl.CollisionData(request = req)
req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
# ddata = fcl.DistanceData(request = req)
res = fcl.DistanceResult()
# manager1.distance(manager2, rdata, fcl.defaultDistanceCallback)
# manager1.distance(manager2, ddata, fcl.defaultDistanceCallback)
fcl.distance(objs1[0], objs2[0], request=req, result=res)
# print('Collision between manager 1 and manager 2?: {}'.format(ddata.result.is_collision))
print('Collision between manager 1 and manager 2?: {}'.format(res.min_distance))
# print('Contacts:')
# for c in ddata.result.contacts:
#     print('\tO1: {}, O2: {}, depth: {}, pos: {}, normal: {}'.format(c.o1, c.o2, c.penetration_depth, c.pos, c.normal))
# print(1-2/np.sqrt(5))