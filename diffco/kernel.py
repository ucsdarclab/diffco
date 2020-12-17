import numpy as np
import torch

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
        xs = xs[:, np.newaxis] # change to [1, len(x), channel]
        pair_diff = x_primes[np.newaxis, :] - xs
        kvalues = (1/(1+self.gamma/self.p*torch.sum(pair_diff**2, dim=2))**self.p)
        if kvalues.shape[0] == 1:
            kvalues = kvalues.squeeze_(0)

        return kvalues

class CauchyKernel(KernelFunc):
    def __init__(self, c):
        self.c = c
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[:, np.newaxis] # change to [1, len(x), channel]
        pair_diff = x_primes[np.newaxis, :] - xs
        kvalues = self.c / (np.sum(pair_diff**2, axis=2) + self.c)
        if kvalues.shape[0] == 1:
            kvalues = kvalues.squeeze_(0)
        return kvalues

class MultiQuadratic(KernelFunc):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[:, np.newaxis] # change shape to [1, len(x), channel]
        pair_diff = x_primes[np.newaxis, :] - xs  # shape [len(x_primes), len(xs), channel]
        kvalues = torch.sqrt(torch.sum(pair_diff**2, axis=2)/self.epsilon**2 + 1)
        if kvalues.shape[0] == 1:
            kvalues = kvalues.squeeze(0)
        return kvalues

class Polyharmonic(KernelFunc):
    def __init__(self, k, epsilon):
        self.epsilon = epsilon
        if k % 2 == 0:
            def _even_func(r):
                tmp = (r**k * torch.log(r))
                tmp[torch.isnan(tmp)] = 0
                return tmp
            self._func = _even_func
        else:
            def _odd_func(r):
                return r**k
            self._func = _odd_func
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        r = torch.cdist(xs, x_primes)
        kvalues = self._func(r) / self.epsilon
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues
        

# def mq_r(self, r):
#     kvalues = torch.sqrt(r**2/self.epsilon**2 + 1)
#     return kvalues

# class mq(KernelFunc):
#     def __init__(self, epsilon):
#         self.epsilon = epsilon
    
#     def __call__(self, xs, x_primes):
#         if xs.ndim == 1:
#             xs = xs[np.newaxis, :]
#         xs = xs[np.newaxis, :] # change to [1, len(x), channel]
#         pair_diff = x_primes[:, np.newaxis] - xs  # [len(x_primes), len(xs), channel]
#         kvalues = torch.sqrt(torch.sum(pair_diff**2, axis=2)
#         if kvalues.shape[1] == 1:
#             kvalues = kvalues.squeeze(1)
#         return kvalues

class WeightedKernel(KernelFunc):
    def __init__(self, gamma, w, p=2):
        self.gamma = gamma
        self.p = p
        self.w = np.array(w).reshape((1, 1, -1))

    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[:, np.newaxis] # change shape to [1, len(x), channel]
        pair_diff = x_primes[np.newaxis, :] - xs  # shape [len(x_primes), len(xs), channel]
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
        xs = xs[:, np.newaxis] # change shape to [1, len(x), channel]
        pair_prod = x_primes[np.newaxis, :] * xs # [len(x_primes), len(xs), channel]
        kvalues = np.tanh(self.a * np.sum(pair_prod, 2) + self.c)
        if kvalues.shape[1] == 1:
            kvalues = kvalues.squeeze(1)
        return kvalues

class FKKernel(KernelFunc):
    def __init__(self, fkine, rq_kernel):
        self.fkine = fkine
        self.rq_kernel = rq_kernel
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs_controls = self.fkine(xs).reshape(len(xs), -1)
        x_primes_controls = self.fkine(x_primes).reshape(len(x_primes), -1)
        return self.rq_kernel(xs_controls, x_primes_controls)

class LineKernel(KernelFunc):
    def __init__(self, point_kernel):
        self.point_kernel = point_kernel
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        if x_primes.ndim == 1:
            x_primes = x_primes[np.newaxis, :]
        twice_DOF = xs.shape[1]
        assert twice_DOF == x_primes.shape[1]
        assert twice_DOF%2 == 0
        dof = twice_DOF // 2
        return (self.point_kernel(xs[:, :dof], x_primes[:, :dof])\
            + self.point_kernel(xs[:, dof:], x_primes[:, dof:]))/2

class LineFKKernel(KernelFunc):
    def __init__(self, fkine, rq_kernel):
        self.fkine = fkine
        self.rq_kernel = rq_kernel
    
    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        if x_primes.ndim == 1:
            x_primes = x_primes[np.newaxis, :]
        twice_DOF = xs.shape[1]
        assert twice_DOF == x_primes.shape[1]
        assert twice_DOF % 2 == 0
        dof = twice_DOF // 2
        xs_controls = self.fkine(xs.reshape(-1, dof)).reshape(len(xs), -1)
        x_primes_controls = self.fkine(x_primes.reshape(-1, dof)).reshape(len(x_primes), -1)
        return self.rq_kernel(xs_controls, x_primes_controls)