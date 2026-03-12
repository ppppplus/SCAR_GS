#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import imageio

from scene import Scene, GaussianModel
from scene.gaussian_model_sc import GaussianModelSC
from gaussian_renderer.render import render_gs
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.vis_utils import visualize_gs_pointcloud
from utils.other_utils import depth2normal
# ===============================
# Helper
# ===============================
def feature_visualize_saving(feature):
    """Normalize ([3,H,W]) feature map for visualization."""
    assert feature.ndim == 3 and feature.shape[0] == 3
    min_val = feature.amin(dim=(1, 2), keepdim=True)
    max_val = feature.amax(dim=(1, 2), keepdim=True)
    feature_norm = (feature - min_val) / (max_val - min_val + 1e-5)
    return feature_norm.permute(1, 2, 0).clamp(0, 1).cpu()


def save_depth(depth_tensor, save_path):
    depth_m = depth_tensor.squeeze().cpu().numpy()   # H×W, float32
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    cv2.imwrite(save_path, depth_mm)


def save_normal(normal: torch.Tensor, path: str):
    """
    normal: Tensor of shape [3, H, W] or [H, W, 3]
    path: save path, e.g. 'normal.png'
    """
    # Rearrange to [H, W, 3]
    if normal.dim() == 3:
        if normal.shape[0] == 3:
            normal = normal.permute(1, 2, 0)   # [H,W,3]
    else:
        raise ValueError("normal must be [3,H,W] or [H,W,3]")

    # detach & move to cpu
    normal_np = normal.detach().cpu().numpy()

    # Normal typically in [-1,1], map to [0,1]
    normal_np = (normal_np + 1.0) / 2.0
    normal_np = np.clip(normal_np, 0.0, 1.0)

    # Save as PNG
    imageio.imwrite(path, (normal_np * 255).astype(np.uint8))
    # print(f"Normal image saved to {path}")

def save_objid_map(semantic_feature, save_path="semantic_objid.png"):
    # semantic_feature : [3, H, W]
    
    # Step 1: take first channel (obj_id)
    sem = semantic_feature[0]  # shape (H,W)
    
    # Step 2: clamp & round to integer
    sem = sem.clamp(min=0)
    sem = sem.round().long()
    
    sem_np = sem.cpu().numpy()
    max_id = int(sem_np.max())
    # print("max obj id:", max_id)

    # Step 3: build colormap [num_classes,3]
    num_classes = max_id + 1
    np.random.seed(42)
    color_map = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)

    # Step 4: map to colors
    colored = color_map[sem_np]  # [H,W,3]

    # Step 5: save
    cv2.imwrite(save_path, colored[..., ::-1])  # RGB→BGR
    print("Saved:", save_path)

# ===============================
# Main rendering function
# ===============================
def render_set(save_root, name, iteration, views, gaussians, pipeline, optim, background):
    """
    save_root/name/ours_xxx/{renders,depth,feature_map}
    """
    base = os.path.join(save_root, name, f"ours_{iteration}")

    folders = {
        "renders": os.path.join(base, "renders"),
        "depth": os.path.join(base, "depth"),
        "feature_map": os.path.join(base, "feature_map"),
        "vis_feature_map": os.path.join(base, "vis_feature_map"),
        "objid_map": os.path.join(base, "objid_map"),
        "normal": os.path.join(base, "normal")
        # "gt_feature_map": os.path.join(base, "gt_feature_map"),
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    print(f"[INFO] Saving results to: {base}")

    for idx, cam in enumerate(tqdm(views, desc=f"Rendering {name}")):
        with torch.no_grad():
            pkg = render_gs(
                cam,
                gaussians,
                pipeline,
                background,
                optim,
            )

            # pkg_obj = render_gs_with_objid(
            #     cam,
            #     gaussians,
            #     pipeline,
            #     background,
            #     optim,
            #     d_xyz=None,
            #     d_rot=None
            # )

            rgb = pkg["render"]
            depth = pkg["depth"]
            fmap = pkg["semantic_feature"]
            # idmap = pkg_obj["semantic_feature"]

            # -------- save RGB ----------
            rgb_path = os.path.join(folders["renders"], f"{idx:04d}.png")
            torchvision.utils.save_image(rgb, rgb_path)

            # -------- save Depth ----------
            depth_path = os.path.join(folders["depth"], f"{idx:04d}.png")
            save_depth(depth, depth_path)

            # -------- save Feature Map ----------
            fmap_path = os.path.join(folders["feature_map"], f"{idx:04d}.npy")
            np.save(fmap_path, fmap.cpu().numpy())

            # # -------- save Feature Map vis --------
            # vis_fmap = feature_visualize_saving(fmap)
            # Image.fromarray((vis_fmap.numpy() * 255).astype(np.uint8)).save(
            #     os.path.join(folders["vis_feature_map"], f"{idx:04d}_feature.png")
            # )

            # -------- save normal ----------
            normal_path = os.path.join(folders["normal"], f"{idx:04d}.npy")
            tanfovx = math.tan(cam.FoVx * 0.5)
            tanfovy = math.tan(cam.FoVy * 0.5)
            fx = cam.image_width / (2 * tanfovx)
            fy = cam.image_height / (2 * tanfovy)
            focal = (fx + fy) / 2
            normal = depth2normal(depth, focal)
            np.save(normal_path, normal.detach().cpu().numpy())

            save_normal(normal, os.path.join(folders["normal"], f"{idx:04d}.png"))

            # -------- save Obj id Map ----------
            # objid_map_path = os.path.join(folders["objid_map"], f"{idx:04d}.png")
            # save_objid_map(fmap, objid_map_path)

    # visualize_gs_pointcloud(gaussians, color_mode="id")


# ===============================
# Config Loader (same as training)
# ===============================
def parse_config(config_path, override_dataset_path=None, override_model_path=None, override_output_path=None):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if override_dataset_path is not None:
        cfg_dict["dataset_path"] = override_dataset_path
    if override_model_path is not None:
        cfg_dict["model_path"] = override_model_path
    if override_output_path is not None:
        cfg_dict["output_path"] = override_output_path

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

    args = parser.parse_args()

    # load config
    cfg, lp, op, pp = parse_config(
        args.config,
        override_dataset_path=args.dataset_path,
        override_model_path=args.model_path,
        override_output_path=args.output_path
    )

    print("[INFO] Rendering using model:", cfg.model_path)

    safe_state(cfg.quiet)

    gaussians = GaussianModel(lp.sh_degree)

    print("[INFO] Loading scene...")
    scene = Scene(lp, gaussians, load_iteration=cfg.iterations, shuffle=False)
    scene.gaussians.training_setup(op)
    scene.gaussians.prune_by_obj_id([1, 2])
    background = torch.tensor(
        [1, 1, 1] if lp.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda"
    )
    render_set(
        cfg.output_path, "catch", scene.loaded_iter,
        scene.getTrainCameras(), gaussians, pp, op, background
    )

    print("[INFO] Rendering complete.")
