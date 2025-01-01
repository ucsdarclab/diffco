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
    # assumes roll pitch yaw (x, y, z).
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
    return rz@ry@rx# rx@ry@rz

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

def se2_wrap2pi(x):
    return torch.cat([x[..., :2], wrap2pi(x[..., 2:3])], dim=-1)

# Generate a sequence of angles between q1 and q2,
# acts like np.linspace but considered the wrapping-around problem
# q1 and q2 can both be vectors of the same dimension
def anglin(q1, q2, num=50, endpoint=True):
    q1 = torch.FloatTensor(q1)
    q2 = torch.FloatTensor(q2)
    dq = torch.from_numpy(np.linspace(np.zeros_like(q1), wrap2pi(q2-q1), num, endpoint))
    return wrap2pi(q1 + dq)

def DH2mat(q, a, d, s_alpha, c_alpha):
    c_theta = q.cos() # n * self.dof
    s_theta = q.sin()
    tfs = torch.stack([
        torch.stack([c_theta, -s_theta*c_alpha, s_theta*s_alpha, a*c_theta], dim=2),
        torch.stack([s_theta, c_theta*c_alpha, -c_theta * s_alpha, a * s_theta], dim=2),
        torch.stack([torch.zeros_like(q), s_alpha.repeat(len(q), 1), c_alpha.repeat(len(q), 1), d.repeat(len(q), 1)], dim=2),
        torch.stack([torch.zeros_like(q)]*3 + [torch.ones_like(q)], dim=2)
    ], dim=2)
    return tfs

# Convert a sequence of adjacent joint angles to be numerically adjacent
# eg. [5pi/6, -pi] will be converted to [5pi/6, pi]
# This is for the convenience of plotting angular configuration space
def make_continue(q, max_gap=np.pi):
    q = torch.FloatTensor(q)
    sudden_change = torch.zeros_like(q)
    sudden_change[1:] = (torch.abs(q[1:]-q[:-1]) > max_gap) * torch.sign(q[1:]-q[:-1])
    offset = -torch.cumsum(sudden_change, dim=0) * np.pi*2
    return q + offset

def dense_path(q, max_step=2.0, max_step_num=None):
    if max_step_num is not None:
        tmp_step_size = torch.norm(q[1:] - q[:-1], dim=-1).sum().item() / max_step_num
        max_step = max_step if max_step > tmp_step_size else tmp_step_size
    denseq = []
    for i in range(len(q)-1):
        delta = q[i+1] - q[i]
        dist = delta.norm()
        num_steps = torch.ceil(dist/max_step).item()
        irange = torch.arange(num_steps).reshape(-1, 1).to(q.device)
        denseq.append(q[i] + irange * delta * max_step/dist)
    denseq.append(q[-1:])
    denseq = torch.cat(denseq)
    assert torch.all(denseq[0] == q[0]) and torch.all(denseq[-1] == q[-1])
    return denseq
        


if __name__ == "__main__":
    # x = np.linspace(-8*np.pi, 8*np.pi, 1000)
    # y = wrap2pi(x)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    print(wrap2pi(0.8*np.pi - (-0.9)*np.pi)/np.pi*180)
    print(anglin([-0.8*np.pi], [0.9*np.pi], 1, False)/np.pi*180)