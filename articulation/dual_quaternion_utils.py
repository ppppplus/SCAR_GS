# Copyright (c) 2023 Chaoyang Wang, Carnegie Mellon University.

import os
import sys

# from lietorch import SE3, SO3, Sim3
from typing import Tuple

import torch
import torch.nn.functional as F

sys.path.insert(
    0,
    "%s/../third_party" % os.path.join(os.path.dirname(__file__)),
)

# from quaternion import quaternion_conjugate as _quaternion_conjugate_cuda
# from quaternion import quaternion_mul as _quaternion_mul_cuda

DualQuaternions = Tuple[torch.Tensor, torch.Tensor]
QuaternionTranslation = Tuple[torch.Tensor, torch.Tensor]

"""
    quaternion library from pytorch3d
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
"""


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


@torch.jit.script
def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    https://mathworld.wolfram.com/QuaternionConjugate.html
    when q is unit quaternion, inv(q) = conjugate(q)
    """
    return torch.cat((q[..., 0:1], -q[..., 1:]), -1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return _quaternion_conjugate(q)
    # if q.is_cuda:
    #     out_shape = q.shape
    #     return _quaternion_conjugate_cuda(q.contiguous().view(-1, 4)).view(out_shape)
    # else:
    #     return _quaternion_conjugate(q)


@torch.jit.script
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        out: Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def standardize_dualquaternion(qr: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    sign = torch.where(qr[..., 0:1] < 0, -1, 1)
    return qr * sign, qd * sign

def normalize_dualquaternion(qr: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    qr_mag_inv = qr.norm(p=2, dim=-1, keepdim=True).reciprocal()
    qr, qd = qr * qr_mag_inv, qd * qr_mag_inv
    return (qr, qd)
    # return standardize_dualquaternion(qr, qd)

@torch.jit.script
def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        out: The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _quaternion_mul(a, b)
    # return standardize_quaternion(_quaternion_mul(a, b))
    # if a.is_cuda:
    #     ouput_shape = list(a.shape[:-1]) + [4]
    #     return _quaternion_mul_cuda(
    #         a.view(-1, a.shape[-1]), b.view(-1, b.shape[-1])
    #     ).view(ouput_shape)
    # else:
    #     return _quaternion_mul(a, b)


@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        o: Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    q2 = quaternions**2
    rr, ii, jj, kk = torch.unbind(q2, -1)
    two_s = 2.0 / q2.sum(-1)
    ij = i * j
    ik = i * k
    ir = i * r
    jk = j * k
    jr = j * r
    kr = k * r

    o1 = 1 - two_s * (jj + kk)
    o2 = two_s * (ij - kr)
    o3 = two_s * (ik + jr)
    o4 = two_s * (ij + kr)

    o5 = 1 - two_s * (ii + kk)
    o6 = two_s * (jk - ir)
    o7 = two_s * (ik - jr)
    o8 = two_s * (jk + ir)
    o9 = 1 - two_s * (ii + jj)

    o = torch.stack((o1, o2, o3, o4, o5, o6, o7, o8, o9), -1)

    return o.view(quaternions.shape[:-1] + (3, 3))


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        out: Tensor of rotated points of shape (..., 3).
    """
    point = torch.cat((torch.zeros_like(point[..., :1]), point), -1)
    out = quaternion_mul(
        quaternion_mul(quaternion, point),
        quaternion_conjugate(quaternion)
        # quaternion
    )
    return out[..., 1:]


def quaternion_translation_apply(
    q: torch.Tensor, t: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    p = quaternion_apply(q, point)
    return p + t


def quaternion_translation_inverse(
    q: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_inv = quaternion_conjugate(q)
    t_inv = quaternion_apply(q_inv, -t)
    return q_inv, t_inv


def quaternion_translation_to_dual_quaternion(
    q: torch.Tensor, t: torch.Tensor
) -> DualQuaternions:
    """
    https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    """
    q_d = 0.5 * quaternion_mul(t, q)
    return (q, q_d)


def dual_quaternion_to_se3(dq):
    q_r, t = dual_quaternion_to_quaternion_translation(dq)
    return quaternion_translation_to_se3(q_r, t)


def quaternion_translation_to_se3(q: torch.Tensor, t: torch.Tensor):
    rmat = quaternion_to_matrix(q)
    rt4x4 = torch.cat((rmat, t[..., None]), -1)  # (..., 3, 4)
    rt4x4 = torch.cat((rt4x4, torch.zeros_like(rt4x4[..., :1, :])), -2)  # (..., 4, 4)
    rt4x4[..., 3, 3] = 1
    return rt4x4


def se3_to_quaternion_translation(se3, tuple=True):
    q = matrix_to_quaternion(se3[..., :3, :3])
    t = se3[..., :3, 3]
    if tuple:
        return q, t
    else:
        return torch.cat((q, t), -1)


def dual_quaternion_to_quaternion_translation(
    dq: DualQuaternions,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_r = dq[0]
    q_d = dq[1]
    t = 2 * quaternion_mul(q_d, quaternion_conjugate(q_r))[..., 1:]

    return q_r, t


def dual_quaternion_apply(dq: DualQuaternions, point: torch.Tensor) -> torch.Tensor:
    q, t = dual_quaternion_to_quaternion_translation(dq)
    return quaternion_translation_apply(q, t, point)

def dual_quaternion_inverse_apply(dq: DualQuaternions, point: torch.Tensor) -> torch.Tensor:
    q, t = dual_quaternion_to_quaternion_translation(dq)
    q, t = quaternion_translation_inverse(q, t)
    return quaternion_translation_apply(q, t, point)

def quaternion_translation_mul(
    qt1: Tuple[torch.Tensor, torch.Tensor], qt2: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    q1, t1 = qt1
    q2, t2 = qt2

    # Multiply the rotations
    q = quaternion_mul(q1, q2)

    # Compute the new translation
    t = quaternion_apply(q1, t2) + t1

    return (q, t)


def dual_quaternion_mul(dq1: DualQuaternions, dq2: DualQuaternions) -> DualQuaternions:
    q_r1 = dq1[0]
    q_d1 = dq1[1]
    q_r2 = dq2[0]
    q_d2 = dq2[1]
    r_r = quaternion_mul(q_r1, q_r2)
    r_d = quaternion_mul(q_r1, q_d2) + quaternion_mul(q_d1, q_r2)
    return (r_r, r_d)


def dual_quaternion_q_conjugate(dq: DualQuaternions) -> DualQuaternions:
    r = quaternion_conjugate(dq[0])
    d = quaternion_conjugate(dq[1])
    return (r, d)


@torch.jit.script
def dual_quaternion_d_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return (dq[0], -dq[1])


def dual_quaternion_3rd_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_d_conjugate(dual_quaternion_q_conjugate(dq))


def dual_quaternion_norm(dq: DualQuaternions) -> DualQuaternions:
    dq_qd = dual_quaternion_q_conjugate(dq)
    return dual_quaternion_mul(dq, dq_qd)


def dual_quaternion_inverse(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_q_conjugate(dq)



def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    axis = quaternions[..., 1:] / sin_half_angles_over_angles
    axis = F.normalize(axis, p=2., dim=0)
    return axis, angles

def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """
        Spherical Linear Interpolation (SLERP) between two quaternions.
        
        Args:
            q1: Start quaternion [4], real part first
            q2: End quaternion [4], real part first
            t: Interpolation parameter in [0, 1]
        
        Returns:
            Interpolated quaternion [4]
        """
        # Ensure q1 and q2 are normalized
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # Compute dot product
        dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)
        
        # If dot product is negative, negate one quaternion to take shorter path
        q2 = torch.where(dot_product < 0, -q2, q2)
        dot_product = torch.abs(dot_product)
        
        # Clamp to avoid numerical issues with acos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Compute angle between quaternions
        theta = torch.acos(dot_product)
        sin_theta = torch.sin(theta)
        
        # Compute interpolation coefficients
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        # Handle case where sin_theta is very small (quaternions are very close)
        w1 = torch.where(sin_theta < 1e-6, 1 - t, w1)
        w2 = torch.where(sin_theta < 1e-6, t, w2)
        
        # Interpolate
        result = w1 * q1 + w2 * q2
        return F.normalize(result, p=2, dim=-1)

def dual_quaternion_slerp(dq1: Tuple[torch.Tensor, torch.Tensor], dq2: Tuple[torch.Tensor, torch.Tensor], t: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spherical Linear Interpolation (SLERP) between two dual quaternions.
    
    Args:
        dq1: Start dual quaternion (qr1, qd1)
        dq2: End dual quaternion (qr2, qd2)
        t: Interpolation parameter in [0, 1]
    
    Returns:
        Interpolated dual quaternion (qr_interp, qd_interp)
    """
    qr1, qd1 = dq1
    qr2, qd2 = dq2
    
    # 1. 插值旋转部分 (SLERP)
    qr_interp = quaternion_slerp(qr1, qr2, t)
    
    # 2. 插值虚部 (线性插值 + 投影以保持单位双四元数约束)
    # 对于双四元数，虚部 qd 满足 qr * conjugate(qd) + qd * conjugate(qr) = 0
    # 直接对虚部进行线性插值通常是足够的，只要最后进行归一化
    qd_interp = (1 - t) * qd1 + t * qd2
    
    # 3. 归一化以确保单位双四元数约束
    return normalize_dualquaternion(qr_interp, qd_interp)


def quat_mul(a, b):
    # a,b: (...,4) in (w,x,y,z)
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def quat_conj(q):
    return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

def quat_apply(q, v):
    # q: (...,4), v:(...,3)
    vq = torch.zeros((*v.shape[:-1], 4), device=v.device, dtype=v.dtype)
    vq[...,1:] = v
    return quat_mul(quat_mul(q, vq), quat_conj(q))[...,1:]

def quat_to_axis_angle(q):
    # q: [4] unit, returns axis [3], angle scalar
    q = q / (q.norm() + 1e-12)
    w = q[0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sin(angle / 2.0)
    if s.abs() < 1e-8:
        axis = torch.tensor([1.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
        angle = torch.zeros((), device=q.device, dtype=q.dtype)
        return axis, angle
    axis = q[1:] / s
    return axis, angle

def axis_angle_to_quat(axis, angle):
    axis = axis / (axis.norm() + 1e-12)
    half = angle / 2.0
    q = torch.zeros(4, device=axis.device, dtype=axis.dtype)
    q[0] = torch.cos(half)
    q[1:] = axis * torch.sin(half)
    return q / (q.norm() + 1e-12)

def dq_translation_from_qr_qd(qr, qd):
    # t = 2 * vec( qd ⊗ qr* )
    qr = qr / (qr.norm() + 1e-12)
    t_quat = 2.0 * quat_mul(qd, quat_conj(qr))
    return t_quat[1:]  # [3]

def solve_pivot_from_revolute(qr, t):
    """
    For revolute about some pivot p:
        t = p - R p  =>  (I - R) p = t
    Solve p (least-squares) from qr (rotation) and translation t.
    """
    # Build R implicitly via applying to basis vectors
    device, dtype = t.device, t.dtype
    e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    R = torch.stack([quat_apply(qr, e1), quat_apply(qr, e2), quat_apply(qr, e3)], dim=1)  # 3x3

    A = torch.eye(3, device=device, dtype=dtype) - R  # 3x3
    # Use least squares / pseudo-inverse because A is rank-deficient (axis direction nullspace)
    p = torch.linalg.pinv(A) @ t
    return p  # one valid pivot (not unique along axis)

def scale_dual_quaternion(qr, qd, alpha):
    """
    Scale a revolute dual quaternion using ONLY (qr,qd,alpha).
    Returns (qr_s, qd_s).

    Assumes this DQ represents pure rotation about some pivot (no extra prismatic component).
    """
    qr = qr / (qr.norm() + 1e-12)

    # 1) axis-angle from qr, scale angle
    axis, angle = quat_to_axis_angle(qr)
    angle_s = alpha * angle
    qr_s = axis_angle_to_quat(axis, angle_s)

    # 2) recover original translation t and infer a pivot p (one solution)
    t = dq_translation_from_qr_qd(qr, qd)           # t(θ)
    p = solve_pivot_from_revolute(qr, t)            # a pivot consistent with (qr,t)

    # 3) recompute translation for scaled angle: t_s = p - R_s p
    Rp_s = quat_apply(qr_s, p)
    t_s = p - Rp_s

    # 4) rebuild qd_s = 0.5 * (0,t_s) ⊗ qr_s
    t_quat_s = torch.zeros(4, device=qr.device, dtype=qr.dtype)
    t_quat_s[1:] = t_s
    qd_s = 0.5 * quat_mul(t_quat_s, qr_s)

    return qr_s, qd_s

