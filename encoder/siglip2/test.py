#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from siglip2.siglip2_utils import SIGLIP2NetWork

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 保持数据加载工具函数不变 ---
def resolve_paths(base_path, frame_id):
    s = str(frame_id).strip()
    stem = s.zfill(4) if s.isdigit() else s
    return (os.path.join(base_path, "images", f"{stem}.png"),
            os.path.join(base_path, "id_masks", f"{stem}.npy"),
            os.path.join(base_path, "siglip2_feat", f"{stem}_ids.npy"),
            os.path.join(base_path, "siglip2_feat", f"{stem}_f.npy"), stem)

def load_rgb(path): return np.array(Image.open(path).convert("RGB"))
def load_id_mask(path):
    m = np.load(path, allow_pickle=True)
    return m[..., 0].astype(np.int32) if m.ndim == 3 else m.astype(np.int32)

def load_obj_feats(feat_p, id_p):
    arr = np.load(feat_p, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object: arr = arr.item()
    if isinstance(arr, dict):
        ids = np.array(sorted(arr.keys()), dtype=np.int32)
        feats = np.stack([np.asarray(arr[i], np.float32).reshape(-1) for i in ids])
        return ids, feats
    ids = np.load(id_p).astype(np.int32).reshape(-1)
    return ids, arr.astype(np.float32)

def l2_normalize(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def sigmoid01(x): return 1.0 / (1.0 + np.exp(-x))

def top1_object(obj_ids, sims, bg_id):
    valid = (obj_ids != int(bg_id))
    if not np.any(valid): return None, None
    idx = np.argmax(np.where(valid, sims, -1e9))
    return int(obj_ids[idx]), float(sims[idx])

def soft_mask_from_id(id_mask, target_id, blur_ksize=51):
    hard = (id_mask == int(target_id)).astype(np.uint8) * 255
    if hard.sum() == 0: return np.zeros(id_mask.shape, dtype=np.float32)
    soft = cv2.GaussianBlur(hard, (blur_ksize, blur_ksize), 0)
    return np.clip(soft.astype(np.float32) / 255.0, 0.0, 1.0)

# --- 核心可视化逻辑修改 ---

def joint_thermal_render(rgb: np.ndarray, soft_masks: list, sims: list, bg_dim: float = 0.5):
    """
    将多个 Query 匹配到的 Top-1 物体画在一起，呈现热力红区效果
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    # 1. 灰度暗背景：只保留微弱的轮廓
    gray = (0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2])[..., None]
    out = np.repeat(gray, 3, axis=-1) * bg_dim 
    
    # 2. 叠加热力层
    for sm, sim in zip(soft_masks, sims):
        # 计算该目标的增益（亮度）
        gain = float(sigmoid01(np.array([(sim - 0.2) * 8.0], np.float32))[0])
        
        # 生成 JET 热力图 (中心红，边缘蓝)
        mask_8u = (sm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_8u, cv2.COLORMAP_JET)
        heatmap_f = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 混合权重：结合蒙版透明度和匹配增益
        alpha_map = (sm * gain * 0.8)[..., None]
        out = out * (1.0 - alpha_map) + heatmap_f * alpha_map
        
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, required=True)
    ap.add_argument("--id", type=str, required=True)
    ap.add_argument("--query", action="append")
    ap.add_argument("--blur_ksize", type=int, default=51)
    args = ap.parse_args()

    # 加载数据
    img_p, mask_p, id_p, feat_p, _ = resolve_paths(args.base_path, args.id)
    rgb = load_rgb(img_p)
    id_mask = load_id_mask(mask_p)
    obj_ids, obj_feats = load_obj_feats(feat_p, id_p)
    obj_feats = l2_normalize(obj_feats)

    # 模型准备
    extractor = SIGLIP2NetWork(device=device)
    
    best_soft_masks = []
    best_sims = []

    for q in args.query:
        # 提取文本特征
        t_feat = extractor.extract_text_features(q)
        if torch.is_tensor(t_feat): t_feat = t_feat.detach().float().cpu().numpy().reshape(-1)
        t_feat = l2_normalize(t_feat)

        # 匹配 Top-1
        sims = obj_feats @ t_feat
        bid, bsim = top1_object(obj_ids, sims, bg_id=0)
        
        if bid is not None:
            best_soft_masks.append(soft_mask_from_id(id_mask, bid, args.blur_ksize))
            best_sims.append(bsim)

    # 生成联合热力图
    joint_view = joint_thermal_render(rgb, best_soft_masks, best_sims)

    # 展示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(joint_view)
    plt.title(f"Joint Heatmap: {', '.join(args.query)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()