import numpy as np

class KernelFunc:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError('You need to define your own __call__ function.')


class RQKernel(KernelFunc):
    def __init__(self, gamma, p=2):
        self.gamma = gamma
        self.p = p

    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[np.newaxis, :] # change to [1, len(x), channel]
        pair_diff = x_primes[:, np.newaxis] - xs
        kvalues = (1/(1+self.gamma/self.p*np.sum(pair_diff**2, axis=2))**self.p)
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues

class CauchyKernel(KernelFunc):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[np.newaxis, :] # change to [1, len(x), channel]
        pair_diff = x_primes[:, np.newaxis] - xs
        kvalues = self.c / (np.sum(pair_diff**2, axis=2) + self.c)
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues

class MultiQuadratic(KernelFunc):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[np.newaxis, :] # change to [1, len(x), channel]
        pair_diff = x_primes[:, np.newaxis] - xs  # [len(x_primes), len(xs), channel]
        kvalues = np.sqrt(np.sum(pair_diff**2, axis=2)/self.epsilon**2 + 1)
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues

class WeightedKernel(KernelFunc):
    def __init__(self, gamma, w, p=2):
        self.gamma = gamma
        self.p = p
        self.w = np.array(w).reshape((1, 1, -1))

    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[np.newaxis, :] # change to [1, len(x), channel]
        pair_diff = x_primes[:, np.newaxis] - xs # [len(x_primes), len(xs), channel]
        kvalues = 1/(1+self.gamma/self.p*np.sum((pair_diff*self.w)**2, axis=2))**self.p
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues

class TangentKernel(KernelFunc):
    def __init__(self, a, c):
        self.a = a
        self.c = c
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[np.newaxis, :] # change to [1, len(x), channel]
        pair_prod = x_primes[:, np.newaxis] * xs # [len(x_primes), len(xs), channel]
        kvalues = np.tanh(self.a * np.sum(pair_prod, 2) + self.c)
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues