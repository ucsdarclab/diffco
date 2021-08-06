import fcl
import torch
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation
from .utils import rot_2d, euler2mat, DH2mat, rotz
import trimesh

class Model():
    def __init__(self):
        self.dof = None
        self.limits = None
        
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

class RigidPlanarBody(Model):
    def __init__(self, parts, limits=None):
        self.parts = parts # [({type}, {horizontal_dimension/x}, {vertical dimension/y})]
        self.dof = 3
        self.limits = torch.FloatTensor(limits) if limits != None else torch.FloatTensor([[-10, 10], [-10, 10], [-pi, pi]])
        keypoints = []
        for p in parts:
            keypoints.append(p[1])
        self.keypoints = torch.FloatTensor(keypoints).T # 2*M
        self.collision_objs = None
    
    # Assume first two configurations are offsets of (x, y), the third configuration is \theta
    def fkine(self, q):
        q = q.reshape((-1, 3))
        points = rot_2d(q[:, 2]) @ self.keypoints + q[:, :2, None] # N*2*M + N*2*1
        return points.permute((0, 2, 1))
    
    @torch.no_grad()
    def update_polygons(self, q):
        centers = self.fkine(q)[0]
        angle = torch.reshape(q, (3, ))[2]
        if self.collision_objs is None:
            self.collision_objs = []
            for trans, p in zip(centers, self.parts):
                obj = fcl.Box(p[2][0], p[2][1], 1000)
                self.collision_objs.append(fcl.CollisionObject(obj, fcl.Transform(
                    Rotation.from_rotvec([0, 0, angle]).as_quat()[[3,0,1,2]], 
                    [trans[0], trans[1], 0])))
        else:
            for obj, trans, in zip(self.collision_objs, centers):
                obj.setTransform(fcl.Transform(
                    Rotation.from_rotvec([0, 0, angle]).as_quat()[[3,0,1,2]], 
                    [trans[0], trans[1], 0]))

        return self.collision_objs

class RigidBody(Model):
    def __init__(self, body_path, keypoints=None, limits=None, transform=None, center=True):
        # Setting keypoints to none means using the corner points as keypoints
        # Update: if providing
        self.body_path = body_path
        self.collision_objs = []
        self.transform = transform
        if '.dae' in body_path:
            self.mesh = trimesh.load(body_path, force='mesh')
            if transform is not None:
                self.mesh.apply_transform(transform)
            if center: # OMPL center a robot object before planning
                centering_tf = np.eye(4)#
                centering_tf[:3, 3] = -self.mesh.vertices.mean(0)
                self.mesh.apply_transform(centering_tf)
            self.collision_objs.append(fcl.CollisionObject(trimesh.collision.mesh_to_BVH(self.mesh)))
            # self.mesh.show()
            # open3d_mesh = self.mesh.as_open3d
            # import open3d as o3d
            # o3d.visualization.draw_geometries([open3d_mesh])
            # from utils import open3d_save_image
            # open3d_save_image([open3d_mesh], '../debug.png')
        else:
            self.mesh = trimesh.load(body_path)
            self.collision_objs.append(fcl.CollisionObject(trimesh.collision.mesh_to_BVH(self.mesh)))
        self.dof = 6
        self.limits = torch.FloatTensor(limits) if limits != None else torch.FloatTensor(
            [[-10, 10], [-10, 10], [-10, 10], [-pi, pi], [-pi, pi], [-pi, pi]])
        if keypoints is None:
            corners = torch.from_numpy(trimesh.bounds.corners(self.mesh.bounds).astype(np.float32)).T # 3*M
            self.keypoints = corners / corners.norm(dim=0).max()
            assert self.keypoints.shape == corners.shape
        else:
            self.keypoints = torch.FloatTensor(keypoints)
    
    # Assume first two configurations are offsets of (x, y), the third configuration is \theta
    def fkine(self, q):
        q = q.reshape((-1, self.dof))
        points = euler2mat(q[:, 3:]) @ self.keypoints + q[:, :3, None] # N*2*M + N*2*1
        return points.permute((0, 2, 1))
    
    @torch.no_grad()
    def update_polygons(self, q):
        self.collision_objs[0].setTransform(fcl.Transform(
            Rotation.from_matrix(euler2mat(q[3:])[0].numpy()).as_quat()[[3,0,1,2]], q[:3]))

        return self.collision_objs

class DHParameters():
    def __init__(self, a=0, alpha=0, d=0, theta=0):
        self.a = torch.FloatTensor(a)
        self.alpha = torch.FloatTensor(alpha)
        self.d = torch.FloatTensor(d)
        self.theta = torch.FloatTensor(theta)

class BaxterLeftArmFK(Model):
    # Left arm of Baxter robot
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
            d=  [L[0], 0, L[2], 0, L[4], 0, L[6]],
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
        tfs = DH2mat(angles, self.dhparams.a, self.dhparams.d, self.s_alpha, self.c_alpha)
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

class BaxterDualArmFK(Model):
    def __init__(self):
        # measurement source: 
        # https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf
        self.limits = torch.FloatTensor([[-1.70167993878, 1.70167993878],
                        [-2.147, 1.047],
                        [-3.05417993878, 3.05417993878],
                        [-0.05, 2.618],
                        [-3.059, 3.059],
                        [-1.57079632679, 2.094],
                        [-3.059, 3.059]]).repeat(2, 1)
        assert self.limits.shape == (14, 2)
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
        offsets = torch.FloatTensor([278, 64, 1104]) / 1000 # (L, h, H) in the document
        
        # modeling source: 
        # https://www.researchgate.net/publication/299640286_Baxter_Kinematic_Modeling_Validation_and_Reconfigurable_Representation
        self.left_dhparams = DHParameters(
            a = [L[1], 0, L[3], 0, L[5], 0, 0],
            alpha=[-pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, 0],
            d=  [L[0], 0, L[2], 0, L[4], 0, L[6]],
            theta=[0, pi/2, 0, 0, 0, 0, 0]
        )
        self.right_dhparams = DHParameters(
            a = [L[1], 0, L[3], 0, L[5], 0, 0],
            alpha=[-pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, 0], # different from the document, all alphas of right arms have been reverted to align with urdf
            d=  [L[0], 0, L[2], 0, L[4], 0, L[6]],
            theta=[0, pi/2, 0, 0, 0, 0, 0]
        )
        self.c_left_alpha = self.left_dhparams.alpha.cos()
        self.s_left_alpha = self.left_dhparams.alpha.sin()
        self.c_right_alpha = self.right_dhparams.alpha.cos()
        self.s_right_alpha = self.right_dhparams.alpha.sin()
        
        left_base = torch.zeros(1, 4, 4)
        left_base[:, :3, :3] = rotz(torch.tensor([-pi/4]))
        left_base[:, :, 3] = torch.tensor([offsets[0], -offsets[1], offsets[2], 1])
        right_base = torch.zeros(1, 4, 4)
        right_base[:, :3, :3] = rotz(torch.tensor([-3*pi/4]))
        right_base[:, :, 3] = torch.tensor([-offsets[0], -offsets[1], offsets[2], 1])
        self.arm_bases = torch.cat([left_base, right_base], dim=0)
        self.arm_bases = self.arm_bases[None, :] # shape=(1, 2, 4, 4)

        self.dof = 14
        self.fk_mask = [True, False, True, False, True, False, True]
        self.fkine_backup = None
    
    def fkine(self, q, reuse=False):
        if reuse:
            return self.fkine_backup
        q = torch.reshape(q, (-1, self.dof))
        l_angles = q[:, :self.dof//2] + self.left_dhparams.theta
        r_angles = q[:, self.dof//2:] + self.right_dhparams.theta
        l_tfs = DH2mat(l_angles, self.left_dhparams.a, self.left_dhparams.d, self.s_left_alpha, self.c_left_alpha)
        r_tfs = DH2mat(r_angles, self.right_dhparams.a, self.right_dhparams.d, self.s_right_alpha, self.c_right_alpha)
        assert l_tfs.shape == (len(q), self.dof // 2, 4, 4) and r_tfs.shape == (len(q), self.dof // 2, 4, 4)
        tfs = torch.stack([l_tfs, r_tfs], dim=2) # (len(q), self.dof//2, 2, 4, 4)
        cum_tfs = []
        tmp_tf = self.arm_bases
        for i in range(0, self.dof // 2): # transformations for both arms are done in the same iteration
            tmp_tf = torch.matmul(tmp_tf, tfs[:, i]) # (len(q), 2, 4, 4)
            if self.fk_mask[i]:
                cum_tfs.append(tmp_tf)
        self.fkine_backup = torch.cat([t[:, :, :3, 3] for t in cum_tfs], dim=1) # (len(q), 2 * sum(self.fk_mask), 3)
        return self.fkine_backup


class PandaFK(Model):
    def __init__(self):
        # measurement source: 
        # https://frankaemika.github.io/docs/control_parameters.html
        self.limits = torch.FloatTensor([[-2.8973, 2.8973],
                        [-1.7628, 1.7628],
                        [-2.8973, 2.8973],
                        [-3.0718, -0.0698],
                        [-2.8973, 2.8973],
                        [-0.0175, 3.7525],
                        [-2.8973, 2.8973]])
        L = torch.FloatTensor([
            0.3330, # L0
            0.3160, # L1
            0.0825, # L2, Between joint 3 and joint 4 
            0.3840, # L3
            0.0880, # L4
            0.1070*2, # L5
            ])
        self.L = L
        
        # modeling source: 
        # https://www.researchgate.net/publication/299640286_Baxter_Kinematic_Modeling_Validation_and_Reconfigurable_Representation
        self.dhparams = DHParameters(
            # a =   [   0,     0,    0, L[2], -L[2],    0, L[4]],
            # alpha=[   0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2],
            # d=    [L[0],     0, L[1],    0,  L[3],    0, L[5]],
            # theta=[   0,     0,    0,    0,     0,    0,    0]
            a =   [    0,    0, L[2], -L[2],    0, L[4],     0],
            alpha=[-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2,     0],
            d=    [ L[0],     0, L[1],    0,  L[3],    0, L[5]],
            theta=[    0,     0,    0,    0,     0,    0,    0]
        )
        print(self.dhparams.a, self.dhparams.alpha, self.dhparams.d)
        self.c_alpha = self.dhparams.alpha.cos()
        self.s_alpha = self.dhparams.alpha.sin()
        self.dof = 7
        self.fk_mask = [True, False, True, True, True, False, True]
        self.fkine_backup = None
    
    def fkine(self, q, reuse=False):
        if reuse:
            return self.fkine_backup
        q = torch.reshape(q, (-1, self.dof))
        angles = q + self.dhparams.theta
        tfs = DH2mat(angles, self.dhparams.a, self.dhparams.d, self.s_alpha, self.c_alpha)
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

class PointRobot1D(Model):
    def __init__(self, limits):
        # limits: (dof+1) x 2. Last dimension for time.
        self.limits = torch.FloatTensor(limits)
        self.dof = 1
        pass

    def fkine(self, q):
        '''
        Assume q is from [0, 1], d dimensions
        '''
        q = torch.reshape(q, (-1, self.dof))
        return q * (self.limits[:-1, 1] - self.limits[:-1, 0]) + self.limits[:-1, 0]
    
    def normalize(self, q):
        return (q-self.limits[:, 0]) / (self.limits[:, 1]-self.limits[:, 0])
    
    def unnormalize(self, q):
        return q * (self.limits[:, 1]-self.limits[:, 0]) + self.limits[:, 0]

def main():
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

def test_rigid_body():
    transformation_for_home_environment = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    robot = RigidBody('data/Home_robot.dae', [[0, 0, 0]], transform=transformation_for_home_environment)

def test_robot(RobotClass: Model, cfg):
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    ax = plt.subplot(projection='3d')

    robot = RobotClass()
    control_points = robot.fkine(cfg)[0]
    if 'dual' in RobotClass.__name__.lower():
        l_control_points = torch.cat([torch.zeros(1, 3), control_points[0::2]], dim=0)
        r_control_points = torch.cat([torch.zeros(1, 3), control_points[1::2]], dim=0)
        print(l_control_points.norm(dim=1))
        print(r_control_points.norm(dim=1))
        ax.plot(l_control_points[:, 0], l_control_points[:, 1], l_control_points[:, 2], c='green')
        ax.scatter(l_control_points[0, 0], l_control_points[0, 1], l_control_points[0, 2], c='green')
        ax.plot(r_control_points[:, 0], r_control_points[:, 1], r_control_points[:, 2], c='red')
        ax.scatter(r_control_points[0, 0], r_control_points[0, 1], r_control_points[0, 2], c='green')
    else:
        print(control_points.norm(dim=1))
        ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2])
        ax.scatter(control_points[0, 0], control_points[0, 1], control_points[0, 2], c='green')
    ax.axis('auto')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 4)
    plt.show()

if __name__ == "__main__":
    # main() # the main test
    # test_rigid_body() # test if the RigidBody model works
    # cfg = torch.FloatTensor([-67, 21, 65, -41, -18, 138, 39]) / 180*pi
    # cfg = torch.FloatTensor([-111, -83, 95, -84, -16, 149, 137])/180*pi
    # test_robot(PandaFK, cfg) # plot the kinematic chain of the robot
    cfg = torch.FloatTensor([
        45, -39, 0, 39, -75, 30, 0,
        # -39, -39, 0, 39, 0, 0, 0,
        # 0, 0, 0, 0, 0, 0, 0,
        45, -39, 0, 39, -75, 30, 0,
    ]) / 180*pi
    # cfg = torch.zeros(14)
    test_robot(BaxterDualArmFK, cfg)




