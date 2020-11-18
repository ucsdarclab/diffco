import torch
import numpy as np

def rotz(phi):
    return torch.FloatTensor(
        [[torch.cos(phi), -torch.sin(phi), 0],
         [torch.sin(phi), torch.cos(phi), 0],
         [0, 0, 1]])

def wrap2pi(theta):
    return (np.pi + theta) % (np.pi*2)-np.pi

def anglin(q1, q2, num=50, endpoint=True):
    q1 = torch.FloatTensor(q1)
    q2 = torch.FloatTensor(q2)
    dq = torch.from_numpy(np.linspace(np.zeros_like(q1), wrap2pi(q2-q1), num, endpoint))
    return wrap2pi(q1 + dq)


if __name__ == "__main__":
    # x = np.linspace(-8*np.pi, 8*np.pi, 1000)
    # y = wrap2pi(x)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    print(wrap2pi(0.8*np.pi - (-0.9)*np.pi)/np.pi*180)
    print(anglin([-0.8*np.pi], [0.9*np.pi], 4, False)/np.pi*180)