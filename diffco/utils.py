import torch
import numpy as np

def rotz(phi):
    res = torch.zeros((len(phi), 3, 3))
    s = torch.sin(phi)
    c = torch.cos(phi)
    res[:, 0, 0] = c
    res[:, 0, 1] = -s
    res[:, 1, 0] = s
    res[:, 1, 1] = c
    res[:, 2, 2] = 1
    return res

def euler2mat(phi):
    # assumes roll pitch yaw (x, y, z). yaw is applied first and roll is last
    phi = phi.reshape((-1, 3))
    s = torch.sin(phi)
    c = torch.cos(phi)
    ones = torch.ones_like(s[:, 0])
    zeros = torch.zeros_like(s[:, 0])

    rx = torch.stack([
        ones, zeros, zeros,
        zeros, c[:, 0], -s[:, 0],
        zeros, s[:, 0], c[:, 0]
    ], dim=1).reshape((len(phi), 3, 3))
    ry = torch.stack([
        c[:, 1], zeros, s[:, 1],
        zeros, ones, zeros,
        -s[:, 1], zeros, c[:, 1], 
    ], dim=1).reshape((len(phi), 3, 3))
    rz = torch.stack([
        c[:, 2], -s[:, 2], zeros, 
        s[:, 2], c[:, 2], zeros,
        zeros, zeros, ones, 
    ], dim=1).reshape((len(phi), 3, 3))
    return rx@ry@rz

def rot_2d(phi):
    res = torch.zeros((len(phi), 2, 2))
    s = torch.sin(phi)
    c = torch.cos(phi)
    res[:, 0, 0] = c
    res[:, 0, 1] = -s
    res[:, 1, 0] = s
    res[:, 1, 1] = c
    return res

# Convert an angle to be between [-pi, pi)
def wrap2pi(theta):
    return (np.pi + theta) % (np.pi*2)-np.pi

# Generate a sequence of angles between q1 and q2,
# acts like np.linspace but considered the wrapping-around problem
# q1 and q2 can both be vectors of the same dimension
def anglin(q1, q2, num=50, endpoint=True):
    q1 = torch.FloatTensor(q1)
    q2 = torch.FloatTensor(q2)
    dq = torch.from_numpy(np.linspace(np.zeros_like(q1), wrap2pi(q2-q1), num, endpoint))
    return wrap2pi(q1 + dq)

# Convert a sequence of adjacent joint angles to be numerically adjacent
# eg. [5pi/6, -pi] will be converted to [5pi/6, pi]
# This is for the convenience of plotting angular configuration space
def make_continue(q, max_gap=np.pi):
    q = torch.FloatTensor(q)
    sudden_change = torch.zeros_like(q)
    sudden_change[1:] = (torch.abs(q[1:]-q[:-1]) > max_gap) * torch.sign(q[1:]-q[:-1])
    offset = -torch.cumsum(sudden_change, dim=0) * np.pi*2
    return q + offset

def open3d_save_image(geoms, path):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for geom in geoms:
        vis.add_geometry(geom)
        vis.update_geometry(geom)
        vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()

def save_ompl_path(filename, path):
    # input x, y, z, roll, pitch, yaw ('zyx' convention euler angles)
    # output x,y,z,q1,q2,q3,w (scalar-last quaternions)
    from scipy.spatial.transform import Rotation
    path = path.data.numpy()
    p_numpy = np.zeros((len(path), 7))
    p_numpy[:, :3] = path[:, :3]
    p_numpy[:, 3:] = Rotation.from_euler('zyx', path[:, 3:]).as_quat()
    with open(filename, 'w') as f:
        f.writelines([' '.join(map(str, cfg))+'\n' for cfg in p_numpy.tolist()])


if __name__ == "__main__":
    # x = np.linspace(-8*np.pi, 8*np.pi, 1000)
    # y = wrap2pi(x)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    print(wrap2pi(0.8*np.pi - (-0.9)*np.pi)/np.pi*180)
    print(anglin([-0.8*np.pi], [0.9*np.pi], 1, False)/np.pi*180)