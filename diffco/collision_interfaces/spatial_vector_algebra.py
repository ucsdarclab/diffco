"""
Spatial vector algebra
====================================
TODO
"""
from __future__ import annotations
from typing import Optional
import torch
import math
import numpy as np
import operator
from functools import reduce


def x_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.ones(batch_size)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def y_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def z_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = angle.new_zeros((batch_size, 3, 3))  # compatible with functorch
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size)
    return R


prod = lambda l: reduce(operator.mul, l, 1)


def cross_product(vec3a, vec3b):
    vec3a = convert_into_at_least_2d_pytorch_tensor(vec3a)
    vec3b = convert_into_at_least_2d_pytorch_tensor(vec3b)
    skew_symm_mat_a = vector3_to_skew_symm_matrix(vec3a)
    return (skew_symm_mat_a @ vec3b.unsqueeze(2)).squeeze(2)


def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def vector3_to_skew_symm_matrix(vec3):
    vec3 = convert_into_at_least_2d_pytorch_tensor(vec3)
    batch_size = vec3.shape[0]
    skew_symm_mat = vec3.new_zeros((batch_size, 3, 3))
    skew_symm_mat[:, 0, 1] = -vec3[:, 2]
    skew_symm_mat[:, 0, 2] = vec3[:, 1]
    skew_symm_mat[:, 1, 0] = vec3[:, 2]
    skew_symm_mat[:, 1, 2] = -vec3[:, 0]
    skew_symm_mat[:, 2, 0] = -vec3[:, 1]
    skew_symm_mat[:, 2, 1] = vec3[:, 0]
    return skew_symm_mat


def torch_square(x):
    return x * x


def exp_map_so3(omega, epsilon=1.0e-14):
    omegahat = vector3_to_skew_symm_matrix(omega).squeeze()

    norm_omega = torch.norm(omega, p=2)
    exp_omegahat = (
        torch.eye(3)
        + ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat)
        + (
            ((1.0 - torch.cos(norm_omega)) / (torch_square(norm_omega + epsilon)))
            * (omegahat @ omegahat)
        )
    )
    return exp_omegahat


def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable)
    else:
        return torch.Tensor(variable)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var


class CoordinateTransform(object):
    def __init__(self, rot=None, trans=None, device="cpu"):
        self._device = torch.device(device)

        if rot is None:
            self._rot = torch.eye(3, device=self._device)
        else:
            self._rot = rot
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(3, device=self._device)
        else:
            self._trans = trans
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)

    def set_translation(self, t):
        self._trans = t
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)
        return

    def set_rotation(self, rot):
        self._rot = rot
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)
        return

    def rotation(self):
        return self._rot

    def translation(self):
        return self._trans

    def inverse(self):
        rot_transpose = self._rot.transpose(-2, -1)
        return CoordinateTransform(
            rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2)
        )

    def multiply_transform(self, coordinate_transform):
        new_rot = self._rot @ coordinate_transform.rotation()
        new_trans = (
            self._rot @ coordinate_transform.translation().unsqueeze(2)
        ).squeeze(2) + self._trans
        return CoordinateTransform(new_rot, new_trans)

    def trans_cross_rot(self):
        return vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_quaternion(self):
        batch_size = self._rot.shape[0]
        M = torch.zeros((batch_size, 4, 4)).to(self._rot.device)
        M[:, :3, :3] = self._rot
        M[:, :3, 3] = self._trans
        M[:, 3, 3] = 1
        q = torch.empty((batch_size, 4)).to(self._rot.device)
        t = torch.einsum("bii->b", M)  # torch.trace(M)
        for n in range(batch_size):
            tn = t[n]
            if tn > M[n, 3, 3]:
                q[n, 3] = tn
                q[n, 2] = M[n, 1, 0] - M[n, 0, 1]
                q[n, 1] = M[n, 0, 2] - M[n, 2, 0]
                q[n, 0] = M[n, 2, 1] - M[n, 1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[n, 1, 1] > M[n, 0, 0]:
                    i, j, k = 1, 2, 0
                if M[n, 2, 2] > M[n, i, i]:
                    i, j, k = 2, 0, 1
                tn = M[n, i, i] - (M[n, j, j] + M[n, k, k]) + M[n, 3, 3]
                q[n, i] = tn
                q[n, j] = M[n, i, j] + M[n, j, i]
                q[n, k] = M[n, k, i] + M[n, i, k]
                q[n, 3] = M[n, k, j] - M[n, j, k]
                # q = q[[3, 0, 1, 2]]
            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

    def to_matrix(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.transpose(-2, -1).matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-2, -1)
        mat[:, 3:, 0:3] = -_Erx
        mat[:, 3:, 3:] = self._rot.transpose(-2, -1)
        return mat

    def to_matrix_transpose(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-1, -2)
        mat[:, 3:, 0:3] = -_Erx.transpose(-1, -2)
        mat[:, 3:, 3:] = self._rot.transpose(-1, -2)
        return mat
