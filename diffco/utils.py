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
    


if __name__ == "__main__":
    # x = np.linspace(-8*np.pi, 8*np.pi, 1000)
    # y = wrap2pi(x)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    print(wrap2pi(0.8*np.pi - (-0.9)*np.pi)/np.pi*180)
    print(anglin([-0.8*np.pi], [0.9*np.pi], 1, False)/np.pi*180)