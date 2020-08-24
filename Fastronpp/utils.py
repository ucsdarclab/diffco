import torch
import numpy as np

def rotz(phi):
    return torch.FloatTensor(
        [[torch.cos(phi), -torch.sin(phi), 0],
         [torch.sin(phi), torch.cos(phi), 0],
         [0, 0, 1]])

def wrap2pi(theta):
    return (np.pi + theta) % (np.pi*2)-np.pi

if __name__ == "__main__":
    x = np.linspace(-8*np.pi, 8*np.pi, 1000)
    y = wrap2pi(x)
    from matplotlib import pyplot as plt
    plt.plot(x, y)
    plt.show()