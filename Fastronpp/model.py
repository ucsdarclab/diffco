import fcl
import torch
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation

class Model():
    def fkine(self, q):
        raise NotImplementedError
    
    def polygons(self, q):
        raise NotImplementedError

class RevolutePlanarRobot(Model):
    def __init__(self, link_length, link_width, dof=None, limits=None):
        if limits is None:
            limits = [-np.pi, np.pi]
        if dof is None:
            dof = len(link_length)
        if isinstance(link_length, (int, float)):
            link_length = [link_length] * dof
        if len(limits) == 2 and isinstance(limits[0], (int, float)):
            limits = [limits] * dof
        assert len(limits) == dof and len(link_length) == dof
        self.dof = dof
        self.link_width = link_width
        self.link_length = torch.FloatTensor(link_length)
        self.limits = torch.FloatTensor(limits)
        self.collision_objs = None

    def fkine(self, q):
        q = torch.reshape(q, (-1, self.dof))
        q = torch.cumsum(q, dim=1)
        c = torch.cos(q)
        s = torch.sin(q)
        x = torch.cumsum(self.link_length * c, dim=1)
        y = torch.cumsum(self.link_length * s, dim=1)
        controls = torch.stack([x, y], dim=2)
        return controls
    
    @torch.no_grad()
    def update_polygons(self, q):
        joints = self.fkine(q)[0]
        joints = torch.cat([torch.zeros(1, 2, dtype=joints.dtype), joints], dim=0)
        centers = (joints[:-1] + joints[1:])/2
        q = torch.reshape(q, (-1, self.dof))
        angles = torch.cumsum(q, dim=1)[0]
        if self.collision_objs is None:
            self.collision_objs = []
            for trans, angle, l in zip(centers, angles, self.link_length):
                obj = fcl.Box(l, self.link_width, 1000)
                self.collision_objs.append(fcl.CollisionObject(obj, fcl.Transform(
                    Rotation.from_rotvec([0, 0, angle]).as_quat()[[3,0,1,2]], 
                    [trans[0], trans[1], 0])))
        else:
            for obj, trans, angle, l in zip(self.collision_objs, centers, angles, self.link_length):
                obj.setTransform(fcl.Transform(
                    Rotation.from_rotvec([0, 0, angle]).as_quat()[[3,0,1,2]], 
                    [trans[0], trans[1], 0]))
                # self.collsion_objs.append(fcl.CollisionObject(obj, fcl.Transform(
                #     tgm.angle_axis_to_quaternion(torch.FloatTensor([0, 0, angle])), 
                #     [point[0], point[1], 0])))

        return self.collision_objs

class DHParameters():
    def __init__(self, a=0, alpha=0, d=0, theta=0):
        self.a = torch.FloatTensor(a)
        self.alpha = torch.FloatTensor(alpha)
        self.d = torch.FloatTensor(d)
        self.theta = torch.FloatTensor(theta)

class BaxterFK(Model):
    def __init__(self):
        # measurement source: 
        # https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf
        self.limits = torch.FloatTensor([[-1.70167993878, 1.70167993878],
                        [-2.147, 1.047],
                        [-3.05417993878, 3.05417993878],
                        [-0.05, 2.618],
                        [-3.059, 3.059],
                        [-1.57079632679, 2.094],
                        [-3.059, 3.059]])
        L = torch.FloatTensor([
            270.35, # L0
            69,     # L1
            364.35, # L2
            69,     # L3
            374.29, # L4
            10,     # L5
            387.35  # L6, from the center of wrist-pitch to the end effector tip
            ]) / 1000
        self.L = L
        
        # modeling source: 
        # https://www.researchgate.net/publication/299640286_Baxter_Kinematic_Modeling_Validation_and_Reconfigurable_Representation
        self.dhparams = DHParameters(
            a = [L[1], 0, L[3], 0, L[5], 0, 0],
            alpha=[-pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, 0],
            d=[L[0], 0, L[2], 0, L[4], 0, L[6]],
            theta=[0, pi/2, 0, 0, 0, 0, 0]
        )
        self.c_alpha = self.dhparams.alpha.cos()
        self.s_alpha = self.dhparams.alpha.sin()
        self.dof = 7
        self.fk_mask = [True, False, True, False, True, False, True]
        self.fkine_backup = None
    
    def fkine(self, q, reuse=False):
        if reuse:
            return self.fkine_backup
        q = torch.reshape(q, (-1, self.dof))
        angles = q + self.dhparams.theta
        c_theta = angles.cos() # n * self.dof
        s_theta = angles.sin()
        tfs = torch.stack([
            torch.stack([c_theta, -s_theta*self.c_alpha, s_theta*self.s_alpha, self.dhparams.a*c_theta], dim=2),
            torch.stack([s_theta, c_theta*self.c_alpha, -c_theta * self.s_alpha, self.dhparams.a * s_theta], dim=2),
            torch.stack([torch.zeros_like(q), self.s_alpha.repeat(len(q), 1), self.c_alpha.repeat(len(q), 1), self.dhparams.d.repeat(len(q), 1)], dim=2),
            torch.stack([torch.zeros_like(q)]*3 + [torch.ones_like(q)], dim=2)
        ], dim=2)
        assert tfs.shape == (len(q), self.dof, 4, 4)
        cum_tfs = []
        tmp_tf = tfs[:, 0]
        if self.fk_mask[0]:
            cum_tfs.append(tmp_tf)
        for i in range(1, self.dof):
            tmp_tf = torch.bmm(tmp_tf, tfs[:, i])
            if self.fk_mask[i]:
                cum_tfs.append(tmp_tf)
        self.fkine_backup = torch.stack([t[:, :3, 3] for t in cum_tfs], dim=1)
        return self.fkine_backup





if __name__ == "__main__":
    lw_data = 0.3
    robot = RevolutePlanarRobot(1, dof=7, link_width=lw_data)
    num_frames = 100
    q = 2*(torch.rand(num_frames, 7)-0.5) * np.pi/1.5
    points = robot.fkine(q)
    points = torch.cat([torch.zeros(num_frames, 1, 2), points], axis=1)
    print(points)

    robot_links = robot.update_polygons(q[0])
    cir_pos = torch.FloatTensor([2, 2, 0])
    obs = [fcl.CollisionObject(fcl.Cylinder(1, 1), fcl.Transform(cir_pos))]
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    obs_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    obs_manager.registerObjects(obs)
    robot_manager.setup()
    obs_manager.setup()
    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    rdata = fcl.CollisionData(request = req)
    robot_manager.collide(obs_manager, rdata, fcl.defaultCollisionCallback)
    in_collision = rdata.result.is_collision

    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    import matplotlib.patheffects as path_effects

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111) #, projection='3d'
    ax.axis('equal')
    ax.set_xlim(-8, 7)
    ax.set_ylim(-8, 7)
    ax.set_aspect('equal', adjustable='box')

    trans = ax.transData.transform
    lw = ((trans((1, lw_data))-trans((0,0)))*72/ax.figure.dpi)[1]
    print('Line width = ', lw)
    link_plot, = ax.plot([], [], color='silver', lw=lw, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()], solid_capstyle='round')
    joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)
    # link_plot, = ax.plot(points[0, :, 0], points[0, :, 1], color='silver', lw=lw, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()], solid_capstyle='round')
    # joint_plot, = ax.plot(points[0, :-1, 0], points[0, :-1, 1], 'o', color='tab:red', markersize=lw)
    # eff_plot, = ax.plot(points[0, -1:, 0], points[0, -1:, 1], 'o', color='black', markersize=lw)
    # ax.plot(points[1, :, 0], points[1, :, 1], color='mediumvioletred')
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
    from matplotlib.collections import PatchCollection
    # rects = []
    # angles = torch.cumsum(q, dim=1)
    # for lbpoint, angle in zip(points[0], angles[0]):
    #     print(angle)
    #     rects.append(Rectangle(lbpoint, 1, 0.3, angle=angle * 180/np.pi, lw=8, joinstyle='round'))
    # for lbpoint, angle in zip(points[1], angles[1]):
    #     print(angle)
    #     rects.append(Rectangle(lbpoint, 1, 0.3, angle=angle * 180/np.pi))
    # ax.add_collection(PatchCollection(rects))
    ax.add_patch(Circle(cir_pos[:2], 1, path_effects=[path_effects.withSimplePatchShadow()]))
    ax.set_title('In Collision?: {}'.format(in_collision))

    

    def init():
        return link_plot, joint_plot, eff_plot

    def update(i):
        link_plot.set_data(points[i, :, 0], points[i, :, 1])
        joint_plot.set_data(points[i, :-1, 0], points[i, :-1, 1])
        eff_plot.set_data(points[i, -1:, 0], points[i, -1:, 1])
        
        robot.update_polygons(q[i])
        robot_manager.update()
        assert len(robot_manager.getObjects()) == 7
        rdata = fcl.CollisionData(request = req)
        robot_manager.collide(obs_manager, rdata, fcl.defaultCollisionCallback)
        in_collision = rdata.result.is_collision
        if in_collision:
            print('Collision!!')
        ax.set_title('In Collision?: {} {}'.format(in_collision, robot_manager.getObjects()[3].getTranslation()))
        return link_plot, joint_plot, eff_plot
    
    from matplotlib import animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000, blit=False, init_func=init)
    plt.show()

    # plt.show()



