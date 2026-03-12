import open3d as o3d
import numpy as np
import math
import os

def hat(w):
    wx, wy, wz = w
    return np.array([
        [0,   -wz, wy],
        [wz,   0, -wx],
        [-wy,  wx, 0]
    ], dtype=np.float64)

def rotation_log(R):
    """SO(3) → (axis, angle)"""
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1)/2, -1, 1))
    if theta < 1e-6:
        return np.array([0,0,0]), 0
    w = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2*np.sin(theta))
    return w, theta

def inverse_left_jacobian(w, theta):
    """Inverse of J for v computation"""
    wx = hat(w)
    I = np.eye(3)

    A = np.sin(theta)/theta
    B = (1 - np.cos(theta))/(theta**2)
    C = (theta - np.sin(theta))/(theta**3)

    J = A*I + B*wx + C*(wx @ wx)
    return np.linalg.inv(J)

def se3_to_twist(T):
    """
    From 4x4 SE(3) T → screw twist (ω, v, θ)
    """
    R = T[:3,:3]
    t = T[:3,3]

    # 1. rotation part
    w, theta = rotation_log(R)
    if theta < 1e-6:
        # prismatic joint (no rotation)
        w = np.array([0,0,0])
        v = t / (np.linalg.norm(t) + 1e-8)
        return w, v, np.linalg.norm(t)

    # 2. translation part
    J_inv = inverse_left_jacobian(w, theta)
    v = J_inv @ t

    return w, v, theta


def estimate_screw_from_icp(points0, points1):
    """
    points0: Nx3 from first view
    points1: Nx3 from second view
    """

    if len(points0) < 30:
        return None   # 点太少不可信

    pc0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points0))
    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))

    result = o3d.pipelines.registration.registration_icp(
        pc0, pc1, 
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = result.transformation  # 4x4

    w, v, theta = se3_to_twist(T)
    return w, v, theta, T


# ============= 刚体对齐 & screw axis 求解 ============= #
from scipy.spatial import cKDTree

def build_correspondence(P0, Pk, max_dist=0.02):
    """
    为 Pk 中每个点找到其在 P0 中的最近邻点，返回配对后的 P0c, Pkc

    max_dist：超过此距离的匹配认为无效（可调 1～5cm）
    """
    tree = cKDTree(P0)
    dist, idx = tree.query(Pk)

    valid = dist < max_dist
    if valid.sum() < 10:
        return None, None  # 无有效匹配

    return P0[idx[valid]], Pk[valid]

def icp_rigid_align(P0, P1, voxel_size=0.01):
    """
    输入:
        P0: N1 x 3  (start cloud)
        P1: N2 x 3  (end cloud)
    输出:
        R, t  使得 P1 ≈ R P0 + t
    """
    if len(P0) < 20 or len(P1) < 20:
        raise ValueError("Too few points for ICP")

    # 转为 Open3D
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(P0)
    tgt.points = o3d.utility.Vector3dVector(P1)

    # 下采样
    src_down = src.voxel_down_sample(voxel_size)
    tgt_down = tgt.voxel_down_sample(voxel_size)

    # 法线
    src_down.estimate_normals()
    tgt_down.estimate_normals()

    # 初始化单位矩阵
    init = np.eye(4)

    # ICP
    result = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down,
        max_correspondence_distance=0.05,     # 半径
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    T = result.transformation  # 4x4
    R = T[:3, :3]
    t = T[:3, 3]

    return R, t

def rotation_axis_from_R(R):
    """ 从旋转矩阵提取旋转轴方向 """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if abs(theta) < 1e-5:
        return None   # too small motion

    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ])
    axis = axis / (2 * math.sin(theta))
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return axis


def screw_axis_from_R_t(R, t):
    """
    对纯旋转物体：求旋转轴（axis point & axis direction）
    """

    # --- rotation axis (eigenvector of eigenvalue=1) ---
    w, v = np.linalg.eig(R)
    idx = np.argmin(np.abs(w - 1))
    axis_dir = np.real(v[:, idx])
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)

    # --- point on axis ---
    # solve (I - R)p = t
    A = np.eye(3) - R
    p = np.linalg.pinv(A) @ t  # 最小二乘求解

    return p, axis_dir

def load_pcd_with_objid(folder):
    ply_path = os.path.join(folder, "points3d.ply")
    obj_path = os.path.join(folder, "obj_ids.npy")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"points3d.ply not found in {folder}")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"obj_ids.npy not found in {folder}")

    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points)
    obj_ids = np.load(obj_path)
    # print(xyz.shape, obj_ids.shape)
    if len(xyz) != len(obj_ids):
        raise ValueError(f"Mismatch: {len(xyz)} points vs {len(obj_ids)} obj_ids")

    return xyz, obj_ids

import torch
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from articulation.point_utils.pointnet2_utils import farthest_point_sample, index_points

def estimate_internal_by_density(xyz, k=20, thresh=0.6):
    """
    返回 internal_mask（True = 内部点）
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz)
    dists, _ = nbrs.kneighbors(xyz)
    # 去掉自身距离
    mean_d = dists[:, 1:].mean(axis=1)
    # 距离小 → 点密 → 更可能是内部
    score = mean_d / mean_d.max()
    internal_mask = score < thresh
    return internal_mask

def mask_by_reference_bbox(
    xyz_ref,        # (N0,3) 参考点云（第一个点云）
    xyz_query,      # (N1,3) 待过滤点云（第二个点云）
    margin=0.02,    # 给 bbox 留的“空隙”
):
    """
    返回：
        internal_mask_query : (N1,) bool
            True  = 在参考点云 bbox 内（需要 mask 掉）
            False = 在外部（保留）
    """
    xyz_ref = np.asarray(xyz_ref)
    xyz_query = np.asarray(xyz_query)

    min0 = xyz_ref.min(axis=0)
    max0 = xyz_ref.max(axis=0)

    lower = min0 + margin
    upper = max0 - margin

    inside = np.all(
        (xyz_query >= lower) & (xyz_query <= upper),
        axis=1
    )

    return inside

def match_pcd(xyz0, xyz1, 
              mask_internal=False,
              N=5000,
              use_cluster=False,
              eps0=None, eps1=None,
              min_samples0=10, min_samples1=10,
              keep_strategy="largest"):
    """
    Args:
        xyz0, xyz1: numpy [N0, 3], [N1, 3]
        N: FPS 下采样数量
        use_cluster: 是否对下采样结果做 DBSCAN 去离群
        eps*, min_samples*: DBSCAN 参数
        keep_strategy: "largest" 只保留最大簇, "all" 保留所有非噪声点

    Returns:
        idx0, idx1: 匹配后的全局索引
        p0_out, p1_out: 匹配后的点云子集 [1, M, 3]
    """
    if mask_internal:
        # internal_mask = estimate_internal_by_density(xyz1)
        internal_mask = mask_by_reference_bbox(
            xyz_ref=xyz0,
            xyz_query=xyz1,
            margin=0.1   # 按你的尺度调
        )
        internal_mask1 = np.asarray(internal_mask).astype(bool)
        keep1 = ~internal_mask1
        xyz1_f = xyz1[keep1]
        idx_map1 = np.nonzero(keep1)[0]   # 新索引 → 原全局索引
    else:
        xyz1_f = xyz1
        idx_map1 = None
    # ===== FPS  =====
    pc0 = torch.from_numpy(xyz0).float().unsqueeze(0).cuda()  # [1, N0, 3]
    # pc1 = torch.from_numpy(xyz1).float().unsqueeze(0).cuda()  # [1, N1, 3]
    pc1 = torch.from_numpy(xyz1_f).float().unsqueeze(0).cuda()

    num_fps = min(pc0.shape[1], pc1.shape[1], N)
    s_idx = farthest_point_sample(pc0, num_fps)   # [1, M]
    e_idx = farthest_point_sample(pc1, num_fps)   # [1, M]
    ps = index_points(pc0, s_idx)[0].cpu().numpy()  # [M,3]
    pe = index_points(pc1, e_idx)[0].cpu().numpy()  # [M,3]

    s_idx = s_idx[0].cpu().numpy()
    e_idx = e_idx[0].cpu().numpy()

    # ===== DBSCAN =====
    def auto_eps(P, k=8, mul=2.5):
        if P.shape[0] <= k: return np.inf
        d = np.linalg.norm(P[:, None] - P[None], axis=-1)
        d.sort(axis=1)
        return float(np.median(d[:, k]) * mul)

    def cluster_mask(P, eps, min_samples, keep):
        if not np.isfinite(eps): return np.ones(P.shape[0], dtype=bool)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
        if keep == "largest":
            vals, cnts = np.unique(labels[labels >= 0], return_counts=True)
            if len(vals) == 0: return np.ones(P.shape[0], dtype=bool)
            keep_val = vals[np.argmax(cnts)]
            return (labels == keep_val)
        else:
            return (labels >= 0)

    if use_cluster:
        if eps0 is None: eps0 = auto_eps(ps)
        if eps1 is None: eps1 = auto_eps(pe)
        ms = cluster_mask(ps, eps0, min_samples0, keep_strategy)
        me = cluster_mask(pe, eps1, min_samples1, keep_strategy)
        ps = ps[ms]
        pe = pe[me]
        s_idx = s_idx[ms]
        e_idx = e_idx[me]

    # ===== 3) 匈牙利匹配（矩阵代价） =====
    with torch.no_grad():
        cost = torch.cdist(torch.tensor(ps)[None], torch.tensor(pe)[None])  # [1, M1, M2]
    row, col = linear_sum_assignment(cost.squeeze().cpu().numpy())

    idx0 = s_idx[row]   # 原始 pc0 中的全局下标
    # idx1 = e_idx[col]   # 原始 pc1 中的全局下标
    idx1_local = e_idx[col]
    idx1 = idx_map1[idx1_local] if idx_map1 is not None else idx1_local

    # ===== 4) 输出点云子集（用于 Chamfer） =====
    device = pc0.device
    p0_out = torch.tensor(ps[row], dtype=torch.float32, device=device).unsqueeze(0)  # [1, M, 3]
    p1_out = torch.tensor(pe[col], dtype=torch.float32, device=device).unsqueeze(0)  # [1, M, 3]

    return idx0, idx1, p0_out, p1_out

def screw_to_axis(screw, raw_screw):
    """
    screw: (6,) → [w1,w2,w3, v1,v2,v3]
    轴方向 = ω
    轴上一点 q = v × ω
    """
    screw_np = screw.detach().cpu().numpy() if hasattr(screw, "detach") else screw.cpu().numpy()
    raw_screw_np = raw_screw.detach().cpu().numpy() if hasattr(raw_screw, "detach") else raw_screw.cpu().numpy()
    w = screw_np[:3]
    q = raw_screw_np[3:]

    return q, w
