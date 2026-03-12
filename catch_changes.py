#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import imageio
import torch
import argparse
import yaml
from argparse import Namespace
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from scene import Scene, GaussianModel
from gaussian_renderer.render import render_gs
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from utils.other_utils import depth2normal
from utils.vis_utils import vis_change_gs, visualize_pcd_matching
from collections import Counter
import json
import open3d as o3d
from utils.vis_utils import vis_screw_axis, vis_point_cloud, vis_two_point_clouds, vis_camera_with_pcd, vis_point_cloud_with_changeid, vis_point_cloud_with_id_axis, vis_mask_with_top_ids
from utils.camera_utils import estimate_intrinsics_from_angle
from utils.geo_utils import depth_to_world_points
from utils.axis_init_utils import icp_rigid_align, rotation_axis_from_R, screw_axis_from_R_t, build_correspondence, load_pcd_with_objid, match_pcd, mask_by_reference_bbox
from articulation.expand_utils import equalize_and_chamfer_from_idx
from articulation.vis_utils import visualize_cd_heatmap_byidx

def save_change_ids_json(change_ids, save_path):
    # convert numpy.int32 to python int
    change_ids = [int(x) for x in change_ids]
    with open(save_path, "w") as f:
        json.dump({"change_ids": change_ids}, f, indent=2)
    print(f"[OK] change_ids saved to {save_path}")

# ----------------------------
# Utility: save normal-as-RGB
# ----------------------------
def save_normal_as_image(normal, path):
    # normal: torch tensor [3,H,W] or numpy [H,W,3]
    if isinstance(normal, torch.Tensor):
        normal = normal.detach().cpu().numpy()
        normal = np.transpose(normal, (1,2,0))  # [H,W,3]

    normal_img = (normal + 1) / 2  # [-1,1] -> [0,1]
    normal_img = np.clip(normal_img, 0, 1)
    imageio.imwrite(path, (normal_img * 255).astype(np.uint8))


def load_depth(path):
    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    depth = depth_mm.astype(np.float32) / 1000.0
    return depth

# ----------------------------
# Depth → Normal (from depth2normal)
# ----------------------------
def depth2normal(depth, focal):
    if depth.ndim == 2:
        depth = depth[None, None]                 # [1,1,H,W]
    elif depth.ndim == 3:
        depth = depth[None]                       # [1,1,H,W]

    H, W = depth.shape[-2:]

    # pad depth
    depth = np.concatenate([depth[:, :, :1], depth, depth[:, :, -1:]], axis=2)
    depth = np.concatenate([depth[:, :, :, :1], depth, depth[:, :, :, -1:]], axis=3)

    depth_t = torch.tensor(depth, dtype=torch.float32)

    # derivative kernels
    kernel = torch.tensor([
        [[0, 0, 0],
         [-0.5, 0, 0.5],
         [0, 0, 0]],
        [[0, -0.5, 0],
         [0, 0, 0],
         [0, 0.5, 0]]
    ], dtype=torch.float32).unsqueeze(1)  # [2,1,3,3]

    # ∂z/∂x and ∂z/∂y
    grad = torch.nn.functional.conv2d(depth_t, kernel)[0]  # [2,H,W]
    grad = grad.permute(1,2,0).numpy()  # [H,W,2]

    # fetch valid depth
    z = depth[0,0,1:-1,1:-1]  # [H,W]

    # compute normal
    nx = -grad[:,:,0] * focal / (z + 1e-6)
    ny = -grad[:,:,1] * focal / (z + 1e-6)
    nz = np.ones_like(z)

    normal = np.stack([nx, ny, nz], axis=0)  # [3,H,W]
    normal /= np.linalg.norm(normal, axis=0, keepdims=True) + 1e-6

    return normal

def compute_psnr(img0, img1, eps=1e-8):
    """
    img0, img1 : [H, W, 3], float32, 0~1
    """
    mse = np.mean((img0 - img1) ** 2)
    if mse < eps:
        return 99.0
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# ----------------------------
# Change detection function
# ----------------------------
# def detect_change(
#     rgb0, rgb1, normal0, normal1, depth0, depth1, id_mask,
#     psnr_thresh=13.0, normal_thresh=0.2, depth_thresh=0.04,
#     pixel_ratio_thresh=0.3,     # 一个物体超过 30% 像素变化就认为变了
#     pixel_count_thresh=1000        # 或者变化像素超过 50
# ):
#     """
#     normal0, normal1: [3,H,W]
#     depth0, depth1:   [H,W]
#     id_mask:          [H,W]  int labels
#     """
#     # ------------------------------
#     # 1. normal difference
#     # ------------------------------
#     cos_sim = (normal0 * normal1).sum(0)
#     normal_diff = 1 - cos_sim
#     normal_mask = normal_diff > normal_thresh

#     # ------------------------------
#     # 2. depth difference
#     # ------------------------------
#     depth_diff = np.abs(depth1 - depth0)
#     depth_mask = depth_diff > depth_thresh

#     # ------------------------------
#     # 3. “变化像素”是 normal 和 depth 的交集
#     # ------------------------------
#     change_mask = np.logical_and(normal_mask, depth_mask)

#     # ------------------------------
#     # 4. 对每个物体（id）检查变化量
#     # ------------------------------
#     unique_ids = np.unique(id_mask)
#     unique_ids = unique_ids[unique_ids != -1]   # 去除非法 id，如果有的话

#     changed_objects = []

#     for obj_id in unique_ids:
#         obj_pixels = (id_mask == obj_id)

#         total_pixels = obj_pixels.sum()
#         if total_pixels == 0:
#             continue

#         changed_pixels = (change_mask & obj_pixels).sum()

#         ratio = changed_pixels / total_pixels

#         # 如果该物体的变化“足够多”，则认为该物体发生了变化
#         if ratio > pixel_ratio_thresh and changed_pixels > pixel_count_thresh:
#             print(f"Object {obj_id} changed by {ratio:.2%} pixels, num {changed_pixels}")
#             changed_objects.append(obj_id)

#     # ------------------------------
#     # 5. 构造 final_mask（所有变动物体的 mask 并集）
#     # ------------------------------
#     final_mask = np.zeros_like(id_mask, dtype=bool)
#     for obj_id in changed_objects:
#         final_mask |= (id_mask == obj_id)
    
#     # =========================================================
#     # 0. PSNR check
#     # =========================================================
#     if rgb0 is not None and rgb1 is not None:
#         img0 = rgb0.astype(np.float32) / 255.0
#         img1 = rgb1.astype(np.float32) / 255.0

#         psnr = compute_psnr(img0, img1)
#         print(f"[detect_change] PSNR = {psnr:.2f} dB")

#         if psnr < psnr_thresh:
#             print(f"[detect_change] PSNR<{psnr_thresh} → Skip change detection")
#             changed_objects = []

#     return final_mask, normal_diff, depth_diff, change_mask, changed_objects


def detect_change(
    rgb0, rgb1, normal0, normal1, depth0, depth1, id_mask,
    psnr_thresh=13.0, normal_thresh=0.2, depth_thresh=0.04,
    pixel_ratio_thresh=0.3,      # 一个物体超过 30% 像素变化就认为变了
    pixel_count_thresh=1000,     # 或者变化像素超过 1000
    min_obj_area_ratio=0.1,     # ✅ 新增：物体在整图至少占 1%
    min_obj_area_pixels=0        # ✅ 新增：物体像素数至少多少（0 表示不启用）
):
    """
    normal0, normal1: [3,H,W]
    depth0, depth1:   [H,W]
    id_mask:          [H,W]  int labels
    """
    H, W = id_mask.shape[:2]
    img_pixels = H * W

    # ------------------------------
    # 1. normal difference
    # ------------------------------
    cos_sim = (normal0 * normal1).sum(0)
    normal_diff = 1 - cos_sim
    normal_mask = normal_diff > normal_thresh

    # ------------------------------
    # 2. depth difference
    # ------------------------------
    depth_diff = np.abs(depth1 - depth0)
    depth_mask = depth_diff > depth_thresh

    # ------------------------------
    # 3. “变化像素”是 normal 和 depth 的交集
    # ------------------------------
    change_mask = np.logical_and(normal_mask, depth_mask)

    # ------------------------------
    # 4. 对每个物体（id）检查变化量
    # ------------------------------
    unique_ids = np.unique(id_mask)
    unique_ids = unique_ids[unique_ids != -1]   # 去除非法 id（如果有）

    changed_objects = []

    for obj_id in unique_ids:
        obj_pixels = (id_mask == obj_id)
        total_pixels = int(obj_pixels.sum())
        if total_pixels == 0:
            continue

        # ✅ 新增：物体面积过滤（占比 + 像素数）
        area_ratio = total_pixels / float(img_pixels)
        if area_ratio < float(min_obj_area_ratio):
            continue
        if min_obj_area_pixels > 0 and total_pixels < int(min_obj_area_pixels):
            continue

        changed_pixels = int((change_mask & obj_pixels).sum())
        ratio = changed_pixels / float(total_pixels)

        if (ratio > pixel_ratio_thresh) and (changed_pixels > pixel_count_thresh):
            print(
                f"Object {obj_id} changed: "
                f"change_ratio={ratio:.2%}, changed_pixels={changed_pixels}, "
                f"obj_area_ratio={area_ratio:.2%}, obj_pixels={total_pixels}"
            )
            changed_objects.append(obj_id)

    # ------------------------------
    # 5. 构造 final_mask（所有变动物体的 mask 并集）
    # ------------------------------
    final_mask = np.zeros_like(id_mask, dtype=bool)
    for obj_id in changed_objects:
        final_mask |= (id_mask == obj_id)

    # =========================================================
    # 0. PSNR check
    # =========================================================
    if rgb0 is not None and rgb1 is not None:
        img0 = rgb0.astype(np.float32) / 255.0
        img1 = rgb1.astype(np.float32) / 255.0

        psnr = compute_psnr(img0, img1)
        print(f"[detect_change] PSNR = {psnr:.2f} dB")

        if psnr < psnr_thresh:
            print(f"[detect_change] PSNR<{psnr_thresh} → Skip change detection")
            changed_objects = []
            # final_mask[:] = False  # ✅ 也清空 final_mask，避免返回旧结果

    return final_mask, normal_diff, depth_diff, change_mask, changed_objects

# ========================
# PCA screw axis estimation
# ========================
def estimate_screw_axis(object_clouds, obj_id):
    P0 = object_clouds[obj_id]["P0"]
    Pks = object_clouds[obj_id]["Pk"]

    if P0 is None or len(Pks) == 0:
        return None, None

    disp_all = []
    for Pk in Pks:
        if len(Pk) == 0:
            continue
        M = min(len(Pk), len(P0))
        disp_all.append(Pk[:M] - P0[:M])

    if len(disp_all) == 0:
        return None, None

    D = np.concatenate(disp_all, axis=0)

    # PCA
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    axis_dir = Vt[0]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-6)

    axis_point = P0.mean(axis=0)
    return axis_point, axis_dir

# ----------------------------
# Main batch processing
# ----------------------------
def catch_changes_by_normal(model_path, name, iteration, views, lp, vis=True):

    base = os.path.join(model_path, name, f"ours_{iteration}")
    rgb_dir = os.path.join(base, "renders")
    normal_dir = os.path.join(base, "normal")
    depth_dir = os.path.join(base, "depth")
    dataset_rgb_dir = os.path.join(lp.dataset_path, "images")
    dataset_id_mask_dir = os.path.join(lp.dataset_path, "id_masks")
    dataset_depth_dir = os.path.join(lp.dataset_path, "depth")

    # 构建固定可复用 colormap（最多 20 种颜色，可扩展）
    idcmap = plt.get_cmap("tab20")
    all_rgb0_list = []
    all_rgb1_list = []
    all_depth0_list = []
    all_normal0_list = []
    all_depth1_list = []
    all_normal1_list = []
    all_changed_objects = []
    # for fname in file_list:
    for idx, cam in enumerate(tqdm(views, desc=f"Rendering {name}")):
        fname = cam.image_name
        rgb0_path = os.path.join(rgb_dir, f"{fname}.png")
        normal0_path = os.path.join(normal_dir, f"{fname}.npy")
        depth0_path = os.path.join(depth_dir, f"{fname}.png")
        rgb1_path = os.path.join(dataset_rgb_dir, f"{fname}.png")
        depth1_path  = os.path.join(dataset_depth_dir, f"{fname}.png")
        id_mask_path = os.path.join(dataset_id_mask_dir, f"{fname}.npy")
        rgb0 = cv2.imread(rgb0_path)
        rgb1 = cv2.imread(rgb1_path)
        normal0 = np.load(normal0_path)              # [3,H,W]

        depth0 = load_depth(depth0_path)
        depth1 = load_depth(depth1_path)

        # Compute focal
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)
        fx = cam.image_width / (2 * tanfovx)
        fy = cam.image_height / (2 * tanfovy)
        focal = (fx + fy) / 2

        # Get normal1 from depth1
        normal1 = depth2normal(depth1, focal)

        id_mask = np.load(id_mask_path)
        # Detect change
        final_mask, normal_diff, depth_diff, change_mask, changed_objects = detect_change(
            rgb0, rgb1, normal0, normal1, depth0, depth1, id_mask
        )
        # print("changed objects:", changed_objects)
        # changed_objects_list.append(changed_objects)
        all_changed_objects.extend(changed_objects)
        all_rgb0_list.append(rgb0)
        all_rgb1_list.append(rgb1)
        all_depth0_list.append(depth0)
        all_normal0_list.append(normal0)
        all_depth1_list.append(depth1)
        all_normal1_list.append(normal1)
        # Save outputs
        # save_normal_as_image(normal0, os.path.join(out_dir, f"{stem}_normal0.png"))
        # save_normal_as_image(normal1, os.path.join(out_dir, f"{stem}_normal1.png"))

        # ------------------------
        # 可视化 Normal0, Normal1, NormalDiff, DepthDiff, Mask
        # ------------------------
        if vis:
            plt.figure(figsize=(18, 14))

            # ================================
            # Row 1: RGBs
            # ================================
            plt.subplot(4, 3, 1)
            plt.title(f"{fname} - RGB0 (first view)")
            plt.imshow(cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(4, 3, 2)
            plt.title(f"{fname} - RGB1 (second view)")
            plt.imshow(cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(4, 3, 3)
            plt.title("RGB Diff (abs)")
            rgb_diff = np.abs(rgb1.astype(np.float32) - rgb0.astype(np.float32))
            rgb_diff = rgb_diff / (rgb_diff.max() + 1e-6)
            plt.imshow(rgb_diff)
            plt.axis("off")

            # ================================
            # Row 2: Normal
            # ================================
            plt.subplot(4, 3, 4)
            plt.title("Normal0")
            plt.imshow((normal0.transpose(1, 2, 0) + 1) / 2)
            plt.axis("off")

            plt.subplot(4, 3, 5)
            plt.title("Normal1")
            plt.imshow((normal1.transpose(1, 2, 0) + 1) / 2)
            plt.axis("off")

            plt.subplot(4, 3, 6)
            plt.title("Normal Difference")
            plt.imshow(np.clip(normal_diff, 0, 1), cmap='jet')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")

            # ================================
            # Row 3: Depth
            # ================================
            plt.subplot(4, 3, 7)
            plt.title("Depth Difference")
            plt.imshow(np.clip(depth_diff, 0, 1), cmap='jet')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")

            # ================================
            # Row 3-4: Masks
            # ================================
            idvmax = id_mask.max() if id_mask.max() > 0 else 1

            plt.subplot(4, 3, 8)
            plt.title("ID Mask")
            plt.imshow(id_mask, cmap=idcmap, vmin=0, vmax=idvmax)
            plt.colorbar()
            plt.axis("off")

            plt.subplot(4, 3, 9)
            plt.title("Change Mask")
            plt.imshow(change_mask.astype(np.float32), cmap='gray')
            plt.axis("off")

            plt.subplot(4, 3, 10)
            plt.title("Final Mask")
            plt.imshow(final_mask.astype(np.float32), cmap='gray')
            plt.axis("off")

            plt.tight_layout()
            plt.show()
    counter = Counter(all_changed_objects)
    print("[INFO] all changed objects: ", all_changed_objects)
    min_votes = 3   # 可调：一个 id 至少出现 {} 次才被认为是真的变化

    change_ids = [obj_id for obj_id, cnt in counter.items() if cnt >= min_votes]
    print("final_change_ids:", change_ids)
    change_ids_path = os.path.join(base, "change_ids.json")
    save_change_ids_json(change_ids, change_ids_path)
    return change_ids

def filter_change_ids_by_pointcount(obj_ids0, obj_ids1, change_ids, min_points=300, max_points=20000):
    """
    根据点数量筛选 change_ids。

    obj_ids     : (N,) 每个点的物体 ID
    change_ids  : list 初步检测到的变化物体 ID
    min_points  : int   认为有效变化的最小点数
    max_points  : int   认为有效变化的最大点数

    返回：
        new_change_ids : 过滤后的变化物体 ID 列表
        stats          : 每个 change_id 的点数量统计
    """

    obj_ids0 = np.asarray(obj_ids0)
    obj_ids1 = np.asarray(obj_ids1) 

    stats = {}
    new_change_ids = []

    for oid in change_ids:
        count0 = int((obj_ids0 == oid).sum())
        count1 = int((obj_ids0 == oid).sum())

        stats[oid] = [count0, count1]

        if count0 >= min_points and count1 >= min_points and count0 <= max_points and count1 <= max_points:
            new_change_ids.append(oid)

    print(f"[INFO] Raw change_ids: {change_ids}")
    print(f"[INFO] Point count stats: {stats}")
    print(f"[INFO] Filtered change_ids (min_points={min_points}): {new_change_ids}")

    return new_change_ids

def init_axis_from_top_cd(P0, eq_idx, cd_eq, ratio=0.6, vis=False):
    """
    从 chamfer 距离最大的 top 点初始化旋转轴（最简单有效版）
    """

    K = len(cd_eq)
    k = max(50, int(K * ratio))

    # 1. 选 CD 最大的一批点
    top_ids = np.argsort(-cd_eq)[:k]
    Q = P0[eq_idx[top_ids]]   # (k,3)

    # 2. PCA —— 第二主成分作为轴方向（第一是边缘方向）
    Q_center = Q.mean(axis=0)
    U, S, Vt = np.linalg.svd(Q - Q_center, full_matrices=False)

    edge_dir = Vt[0]               # 第一主方向 = 点沿门边方向
    axis_dir = Vt[1]               # 第二主方向 ≈ 旋转轴方向
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    edge_dir = edge_dir / np.linalg.norm(edge_dir)


    axis_point = Q_center          # 用点云中心作为轴上的点
    return axis_point, edge_dir, eq_idx[top_ids]

def init_screws(model_path, name, iteration, lp, gaussians, change_ids, min_points=500, max_points=20000, vis=True):
    """
    model_path: 模型目录
    name: "catch"
    iteration: loaded_iter
    lp: ModelParams
    gaussians: GaussianModel
    change_ids: 发生运动的 obj_id 列表
    """

    print(f"[INFO] Screws Initializing...")
    print(f"[INFO] Initial change_ids = {change_ids}")
    # ------------------------ P0 from 3DGS ------------------------
    xyz0 = gaussians.get_xyz.detach().cpu().numpy()
    obj_id0 = gaussians.get_obj_ids.detach().cpu().numpy()

    xyz1, obj_id1 = load_pcd_with_objid(lp.dataset_path)
    
    change_ids = filter_change_ids_by_pointcount(obj_id0, obj_id1, change_ids, min_points=min_points, max_points=max_points)
    # screw_dict = {}
    screw_dict = { oid: {} for oid in change_ids }

    for oid in change_ids:
        print(f"\n==== Processing object {oid} ====")

        # --- 找到该物体的点 ---
        mask0 = (obj_id0 == oid)
        mask1 = (obj_id1 == oid)

        if mask0.sum() < 10 or mask1.sum() < 10:
            print(f"[Skip] object {oid}: too few points")
            continue

        P0 = xyz0[mask0]   # (N0_i, 3)
        P1 = xyz1[mask1]   # (N1_i, 3)

        # --- 关键：用 P0 的 bbox mask P1 的“内部点” ---
        internal1 = mask_by_reference_bbox(
            xyz_ref=P0,
            xyz_query=P1,
            margin=0.1   # 按你的尺度调
        )

        # 保留外部点
        keep1 = ~internal1
        P1_ext = P1[keep1]

        # === 建立索引映射（关键）===
        idx0_obj = np.where(mask0)[0]              # P0[i] → xyz0[idx0_obj[i]]
        idx1_obj = np.where(mask1)[0][keep1]       # P1_ext[j] → xyz1[idx1_obj[j]]
        

        # P0 = xyz0[mask0]
        # P1 = xyz1[mask1]
        
        # if len(P0) < 20 or len(P1) < 20:
        #     print(f"[WARN] object {oid}: too few points, skip")
        #     continue

        # if vis:
        #     vis_two_point_clouds(P0, P1)

        # idx0_local, idx1_local, p0_match, p1_match = match_pcd(P0, P1)
        # visualize_pcd_matching(
        #     P0, P1,
        #     idx0_local, idx1_local,
        #     max_lines=1500
        # )
        # eq_idx0, eq_idx1, cd0_eq, cd1_eq = equalize_and_chamfer_from_idx(
        #     P0, P1,
        #     idx0=idx0_local, 
        #     idx1=idx1_local,
        #     # ① 先做“领域扩张”（真正加入邻域点）
        #     pre_expand_mode="radius",   # 或 "knn" / "none"
        #     pre_radius=0.3,
        #     pre_knn_k=8,
        #     pre_cap0=None,              # 可选上限，防止爆量
        #     pre_cap1=None,
        #     # ② 再做数量对齐
        #     equalize_mode="radius",     # 或 "knn" / "duplicate"
        #     neighbor_radius=0.03,
        #     neighbor_knn_k=8,
        #     jitter=1e-4
        # )
        # # visualize_cd_heatmap_byidx(P0, eq_idx0, cd0_eq)
        # # visualize_cd_heatmap_byidx(P1, eq_idx1, cd1_eq)
        
        # axis_point, axis_dir, top_local_ids = init_axis_from_top_cd(P1, eq_idx1, cd1_eq)
        # print(f"[OK] obj {oid}: axis_point={axis_point}, axis_dir={axis_dir}")
        # # global_top_ids = np.where(mask1)[0][top_local_ids]
        # top_eq0 = eq_idx0[top_local_ids]        # same K points, but mapped to P0

        # # 3. 映射到 xyz0 全局索引
        # global_top_ids = np.where(mask0)[0][top_eq0]

        # # --- 可视化：当前物体的点云 + 轴 ---
        # # mask = (obj_id0 == oid)
        # # cloud_obj = o3d.geometry.PointCloud()
        # # cloud_obj.points = o3d.utility.Vector3dVector(xyz1)
        # # cloud_obj.paint_uniform_color([0.2, 0.8, 0.2])

        # if vis:
        #     print("Vis axis...")
        #     # vis_screw_axis(axis_point, axis_dir, base_cloud=cloud_obj)
        #     vis_point_cloud_with_id_axis(xyz1, obj_id1, change_ids, axis_point, axis_dir)
        # screw_dict[oid]["axis_point"] = axis_point.tolist()
        # screw_dict[oid]["axis_dir"] = axis_dir.tolist()
        # screw_dict[oid]["top_global_ids"] = global_top_ids.tolist()  
        idx0_local, idx1_local, p0_match, p1_match = match_pcd(
            P0, P1_ext
        )      
        visualize_pcd_matching(
            P0, P1_ext,
            idx0_local, idx1_local,
            max_lines=1500
        )
        # eq_idx0, eq_idx1, cd0_eq, cd1_eq = equalize_and_chamfer_from_idx(
        #     P0, P1_ext,
        #     idx0=idx0_local,
        #     idx1=idx1_local,
        #     pre_expand_mode="radius",
        #     pre_radius=0.3,
        #     pre_knn_k=8,
        #     equalize_mode="radius",
        #     neighbor_radius=0.03,
        #     neighbor_knn_k=8,
        #     jitter=1e-4
        # )
        eq_idx0, eq_idx1, cd0_eq, cd1_eq = equalize_and_chamfer_from_idx(
            P0, P1_ext,
            idx0=idx0_local,
            idx1=idx1_local,

            # (A) 领域扩张：尽量拉进邻域点
            pre_expand_mode="knn",
            pre_knn_k=64,          # 原来 8 → 64（很关键）
            pre_radius=0.4,        # 作为兜底
            pre_cap0=None,
            pre_cap1=None,

            # (B) 数量对齐：宁可重复，也要多
            equalize_mode="knn",
            neighbor_knn_k=32,     # 原来 8 → 32
            neighbor_radius=0.06,  # 原来 0.03 → 0.06
            jitter=1e-3            # 防止完全重合
        )
        axis_point, axis_dir, top_local_ids = init_axis_from_top_cd(
            P1_ext, eq_idx1, cd1_eq, ratio=0.6
        )
        # top_local_ids → eq_idx1 → P1_ext local
        top_eq1 = eq_idx1[top_local_ids]

        # P1_ext local → xyz1 global
        global_top_ids_1 = idx1_obj[top_eq1]

        # ===== 从 equalized 匹配中，取对应的 eq_idx0 =====
        top_eq0 = eq_idx0[top_local_ids]          # P0 local indices

        # ===== P0 local → xyz0 global =====
        global_top_ids_0 = idx0_obj[top_eq0]

        if vis:
            vis_point_cloud_with_id_axis(
                xyz1, obj_id1, change_ids,
                axis_point, axis_dir
            )
        vis_mask_with_top_ids(
            xyz0,
            mask0,
            global_top_ids_0
        )

        screw_dict[oid]["axis_point"] = axis_point.tolist()
        screw_dict[oid]["axis_dir"] = axis_dir.tolist()
        screw_dict[oid]["top_global_ids"] = global_top_ids_0.tolist()

    return screw_dict, change_ids

# ===============================
# Config Loader (same as training)
# ===============================
def parse_config(config_path, override_dataset_path=None, override_model_path=None, override_output_path=None, override_min_points=None):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if override_dataset_path is not None:
        cfg_dict["dataset_path"] = override_dataset_path
    if override_model_path is not None:
        cfg_dict["model_path"] = override_model_path
    if override_output_path is not None:
        cfg_dict["output_path"] = override_output_path
    if override_min_points is not None:
        cfg_dict["min_points"] = override_min_points

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    namespace_cfg = Namespace(**cfg_dict)

    model_params = lp.extract(namespace_cfg)
    optim_params = op.extract(namespace_cfg)
    pipe_params = pp.extract(namespace_cfg)

    return namespace_cfg, model_params, optim_params, pipe_params


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    # # 使用end的camera render start训练好的模型，并捕捉变化以初始化铰链轴S
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--min_points", type=int, default=None)

    args = parser.parse_args()

    # load config
    cfg, lp, op, pp = parse_config(
        args.config,
        override_dataset_path=args.dataset_path,
        override_model_path=args.model_path,
        override_output_path=args.output_path,
        override_min_points=args.min_points
    )
    print("[INFO] Rendering using model:", cfg.model_path)
    safe_state(cfg.quiet)
    gaussians = GaussianModel(lp.sh_degree)
    print("[INFO] Loading scene...")
    scene = Scene(lp, gaussians, load_iteration=cfg.iterations, shuffle=False)
    change_ids_path = os.path.join(cfg.output_path, "catch", f"ours_{scene.loaded_iter}", "change_ids.json")
    # if not os.path.exists(change_ids_path):
    change_ids = catch_changes_by_normal(cfg.output_path, "catch", scene.loaded_iter, scene.getTrainCameras(), lp)
        # vis_change_gs(gaussians, change_ids)
    
    with open(change_ids_path, "r") as f:
        change_ids = json.load(f)
    change_ids = change_ids["change_ids"]
    screw_dict, change_ids = init_screws(cfg.output_path, "catch", scene.loaded_iter, lp, gaussians, change_ids, cfg.min_points, cfg.max_points, vis=cfg.vis)
    init_info_dict = {"ids": change_ids, "screw_init": screw_dict}
    init_info_path = os.path.join(cfg.output_path, "catch", f"ours_{scene.loaded_iter}", "screw_init.json")
    with open(init_info_path, "w") as f:
        json.dump(init_info_dict, f, indent=2)
    print(f"[INFO] Screw initialization saved to {init_info_path}")
