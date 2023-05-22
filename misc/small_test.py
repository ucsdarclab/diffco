import numpy as np
import fcl
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects
from matplotlib.cm import get_cmap
import seaborn as sns
sns.set()

box1, box1_t = (20., 5.), (-0.0,0.0)
box2, box2_t = (5., 5.), (-5., 10.)
box3, box3_t = (5., 5.), (-15., 15.)
theta1, theta2 = np.pi/4, 0
theta3 = 0


h = 1000.
box1_bottom_left, box2_bottom_left, box3_bottom_left = (-box1[0]/2, -box1[1]/2), (-box2[0]/2, -box2[1]/2), (-box3[0]/2, -box3[0]/2)
g1 = fcl.Box(*box1, h)
t1 = fcl.Transform(Rotation.from_rotvec([0., 0., theta1]).as_quat()[[3,0,1,2]], [*box1_t,0])
o1 = fcl.CollisionObject(g1, t1)
g2 = fcl.Box(*box2, h)
t2 = fcl.Transform(Rotation.from_rotvec([0., 0., theta2]).as_quat()[[3,0,1,2]], [*box2_t,0])
o2 = fcl.CollisionObject(g2, t2)
g3 = fcl.Box(*box3, h)
t3 = fcl.Transform(Rotation.from_rotvec([0., 0., theta3]).as_quat()[[3,0,1,2]], [*box3_t,0])
o3 = fcl.CollisionObject(g3, t3)

robot_manager = fcl.DynamicAABBTreeCollisionManager()
robot_manager.registerObjects([o1])
robot_manager.setup()
obs_manager = fcl.DynamicAABBTreeCollisionManager()
obs_manager.registerObjects([o2, o3])
# obs_manager.registerObjects([o3, o2])

obs_manager.setup()
ddata = fcl.DistanceData(request=fcl.DistanceRequest(enable_nearest_points=True, gjk_solver_type=fcl.GJKSolverType.GST_LIBCCD))
robot_manager.distance(obs_manager, ddata, fcl.defaultDistanceCallback)
p_o1 = ddata.result.nearest_points[0][:2]
p_o2 = ddata.result.nearest_points[1][:2]
assert np.isclose(ddata.result.min_distance, np.linalg.norm(p_o1-p_o2)), \
    (ddata.result.min_distance, np.linalg.norm(p_o1-p_o2))

## below are just visualization
box1_bottom_left = (Rotation.from_rotvec([0., 0., theta1]).as_matrix() @ np.array([*box1_bottom_left, 0]))[:2]+box1_t
box2_bottom_left = (Rotation.from_rotvec([0., 0., theta2]).as_matrix() @ np.array([*box2_bottom_left, 0]))[:2]+box2_t
box3_bottom_left = (Rotation.from_rotvec([0., 0., theta3]).as_matrix() @ np.array([*box3_bottom_left, 0]))[:2]+box3_t

fig, ax = plt.subplots(1,1)
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal', adjustable='box')

cmaps = [get_cmap('Reds'), get_cmap('Blues')]
with sns.axes_style('ticks'):
    ax.add_patch(Rectangle(tuple(box1_bottom_left), *box1, angle=np.degrees(theta1), path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[1](0.5), alpha=0.5))
    ax.add_patch(Rectangle(tuple(box2_bottom_left), *box2, angle=np.degrees(theta2), path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[0](0.5), alpha=0.5))
    ax.add_patch(Rectangle(tuple(box3_bottom_left), *box3, angle=np.degrees(theta3), path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[0](0.5), alpha=0.5))
ax.plot(*p_o1, 'o', color=cmaps[1](0.5), markersize=4)
ax.plot(*p_o2, 'o', color=cmaps[0](0.5), markersize=4)
plt.show(block=True)