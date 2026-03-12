import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from siglip2.utils import get_image_paths
from siglip2.siglip2_utils import SIGLIP2NetWork
from siglip2.sam_utils import build_sam, masks_to_segmap_and_crops
class SAMSigLIP2Dataset(Dataset):
    """
    - 输入：图像目录 `path`
    - 处理：SAM 分割 -> 为每个实例用 SigLIP2 编码 -> 回填成 per-pixel 特征 [H,W,C]
    - 缓存：每张图保存 *_s.npy (seg_map) 和 *_f.npy (features[K,C])
    - 输出：在 __getitem__ 中按 [N*H*W, C] 的向量逐条返回
    """
    def __init__(self, path: str,
                 sam_ckpt_path: str = "/home/ubuntu/Documents/TJH/model_zoo/sam/sam_vit_h_4b8939.pth",
                 device: str = "cuda",
                 feature_dim: int = 768,
                 cache_subdir: str = "siglip2_features_saml",
                 resize_if_taller_than: int = 1080):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        self.image_paths = get_image_paths(path)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images under {path}")

        # 模型
        self.siglip = SIGLIP2NetWork(device=device)
        self.mask_generator = build_sam(sam_ckpt_path, device=device)

        # 缓存目录
        self.cache_dir = Path(path) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.global_feat_path = self.cache_dir / "cache.npy"   # [N, dim]
        self.segmaps_path     = self.cache_dir / "segmaps.npy"        # [num_imgs, 2] (start, end)
        self.try_load(resize_if_taller_than)

    def try_load(self, resize_if_taller_than: int):
        try:
            # self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.global_feat_path, mmap_mode="r")
        except (FileNotFoundError, ValueError):
            self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.global_feat_path, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]  # [C] (CPU)

    # ---------- 核心流程 ----------
    def _build_instance_features(self, resize_if_taller_than: int):
        # if self.global_feat_path.exists() and self.offsets_path.exists():
        #     return  # 缓存已存在

        feats_chunks = []   # 每张图的 [K_i, dim]
        # offsets = []
        segmaps = []
        cursor = 0

        for img_path in tqdm(self.image_paths, desc="Build global features [N, dim]", leave=False):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")

            # 可选：高图下采样到 <=1080p 以加速 SAM
            H0, W0 = img_bgr.shape[:2]
            if resize_if_taller_than > 0 and H0 > resize_if_taller_than:
                scale = H0 / float(resize_if_taller_than)
                img_bgr = cv2.resize(img_bgr, (int(W0/scale), int(H0/scale)))

            # --- SAM 分割 ---
            masks = self.mask_generator.generate(img_bgr)
            # 转实例 crops（方形 256x256，RGB）
            seg_map, crops = masks_to_segmap_and_crops(masks, img_bgr)

            # --- SigLIP2 编码每个实例 ---
            # if len(crops) == 0:
            #     offsets.append((cursor, cursor))
            #     continue

            feats = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
            for i, crop_rgb in enumerate(crops):
                t = torch.from_numpy(crop_rgb).permute(2, 0, 1)     # [3,256,256] uint8
                f = self.siglip.encode_image_tensor(t)              # [dim] CPU float (已 normalize)
                feats[i] = f.numpy()

            feats_chunks.append(feats)               # [K_i, dim]
            segmaps.append(seg_map)
            # offsets.append((cursor, cursor + feats.shape[0]))
            # cursor += feats.shape[0]

        # 拼接并保存全局缓存
        if len(feats_chunks):
            feats_all = np.concatenate(feats_chunks, axis=0)        # [N, dim]
        else:
            feats_all = np.zeros((0, self.feature_dim), dtype=np.float32)
        segmaps_all = np.stack(segmaps, axis=0)
        np.save(self.global_feat_path, feats_all.astype(np.float32))
        np.save(self.segmaps_path, segmaps_all)

class SAMSigLIP2PatchDataset(Dataset):
    """
    - 输入：图像目录 `path`
    - 处理：SAM 分割 -> 为每个实例用 SigLIP2 编码 -> 回填成 per-pixel 特征 [H,W,C]
    - 缓存：每张图保存 *_s.npy (seg_map) 和 *_f.npy (features[K,C])
    - 输出：在 __getitem__ 中按 [N*H*W, C] 的向量逐条返回
    """
    def __init__(self, path: str,
                 sam_ckpt_path: str = "/home/ubuntu/Documents/TJH/model_zoo/sam/sam_vit_h_4b8939.pth",
                 device: str = "cuda",
                 feature_dim: int = 768,
                 cache_subdir: str = "siglip2_features_saml",
                 resize_if_taller_than: int = 1080):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        self.image_paths = get_image_paths(path)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images under {path}")

        # 模型
        self.siglip = SIGLIP2NetWork(device=device)
        self.mask_generator = build_sam(sam_ckpt_path, device=device)

        # 缓存目录
        self.cache_dir = Path(path) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.global_feat_path = self.cache_dir / "cache.npy"   # [N, dim]
        self.segmaps_path     = self.cache_dir / "segmaps.npy"        # [num_imgs, 2] (start, end)
        self.try_load(resize_if_taller_than)

    def try_load(self, resize_if_taller_than: int):
        try:
            # self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.global_feat_path, mmap_mode="r")
        except (FileNotFoundError, ValueError):
            self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.global_feat_path, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]  # [C] (CPU)

    # ---------- 核心流程 ----------
    def _build_instance_features(self, resize_if_taller_than: int):
        # if self.global_feat_path.exists() and self.offsets_path.exists():
        #     return  # 缓存已存在

        feats_chunks = []   # 每张图的 [K_i, dim]
        # offsets = []
        segmaps = []
        cursor = 0

        for img_path in tqdm(self.image_paths, desc="Build global features [N, dim]", leave=False):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")

            # 可选：高图下采样到 <=1080p 以加速 SAM
            H0, W0 = img_bgr.shape[:2]
            if resize_if_taller_than > 0 and H0 > resize_if_taller_than:
                scale = H0 / float(resize_if_taller_than)
                img_bgr = cv2.resize(img_bgr, (int(W0/scale), int(H0/scale)))

            # --- SAM 分割 ---
            masks = self.mask_generator.generate(img_bgr)
            # 转实例 crops（方形 256x256，RGB）
            seg_map, crops = masks_to_segmap_and_crops(masks, img_bgr)

            # --- SigLIP2 编码每个实例 ---
            # if len(crops) == 0:
            #     offsets.append((cursor, cursor))
            #     continue

            feats = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
            for i, crop_rgb in enumerate(crops):
                t = torch.from_numpy(crop_rgb).permute(2, 0, 1)     # [3,256,256] uint8
                f = self.siglip.encode_image_tensor(t)              # [dim] CPU float (已 normalize)
                feats[i] = f.numpy()

            feats_chunks.append(feats)               # [K_i, dim]
            segmaps.append(seg_map)
            # offsets.append((cursor, cursor + feats.shape[0]))
            # cursor += feats.shape[0]

        # 拼接并保存全局缓存
        if len(feats_chunks):
            feats_all = np.concatenate(feats_chunks, axis=0)        # [N, dim]
        else:
            feats_all = np.zeros((0, self.feature_dim), dtype=np.float32)
        segmaps_all = np.stack(segmaps, axis=0)
        np.save(self.global_feat_path, feats_all.astype(np.float32))
        np.save(self.segmaps_path, segmaps_all)


class SAMSigLIP2VQDataset(Dataset):
    """
    - 输入：图像目录 `path`
    - 处理：SAM 分割 -> 为每个实例用 SigLIP2 编码 -> 回填成 per-pixel 特征 [H,W,C]
    - 缓存：N个图的patch向量保存为[sum(1024*K_n), 768]的cache.npy和[sum(K_n), 768]的features.npy
    - 输出：在 __getitem__ 中按 [B, C] 的向量逐条返回
    """
    def __init__(self, path: str,
                 sam_ckpt_path: str = "/home/ubuntu/Documents/TJH/model_zoo/sam/sam_vit_h_4b8939.pth",
                 device: str = "cuda",
                 feature_dim: int = 768,
                 cache_subdir: str = "siglip2_features_saml_vq",
                 resize_if_taller_than: int = 1080):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        self.image_paths = get_image_paths(path)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images under {path}")

        # 模型
        self.siglip = SIGLIP2NetWork(device=device)
        self.mask_generator = build_sam(sam_ckpt_path, device=device)

        # 缓存目录
        self.cache_dir = Path(path) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.patch_feats_path = self.cache_dir / "cache.npy"   # [N, dim]
        self.features_path     = self.cache_dir / "features.npy"        # [num_imgs, 2] (start, end)
        self.try_load(resize_if_taller_than)

    def try_load(self, resize_if_taller_than: int):
        try:
            # self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.patch_feats_path, mmap_mode="r")
        except (FileNotFoundError, ValueError):
            self._build_instance_features(resize_if_taller_than)
            self.data = np.load(self.patch_feats_path, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]  # [C] (CPU)

    # ---------- 核心流程 ----------
    def _build_instance_features(self, resize_if_taller_than: int):
        # if self.global_feat_path.exists() and self.offsets_path.exists():
        #     return  # 缓存已存在

        feats_chunks = []   # 每张图的 [K_i, dim]
        # offsets = []
        patch_feature_chunks = []
        cursor = 0

        for img_path in tqdm(self.image_paths, desc="Build global features [N, dim]", leave=False):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")

            # 可选：高图下采样到 <=1080p 以加速 SAM
            H0, W0 = img_bgr.shape[:2]
            if resize_if_taller_than > 0 and H0 > resize_if_taller_than:
                scale = H0 / float(resize_if_taller_than)
                img_bgr = cv2.resize(img_bgr, (int(W0/scale), int(H0/scale)))

            # --- SAM 分割 ---
            masks = self.mask_generator.generate(img_bgr)
            # 转实例 crops（方形 256x256，RGB）
            seg_map, crops = masks_to_segmap_and_crops(masks, img_bgr)

            # --- SigLIP2 编码每个实例 ---
            # if len(crops) == 0:
            #     offsets.append((cursor, cursor))
            #     continue

            feats = []
            feat_maps = []
            for i, crop_rgb in enumerate(crops):
                t = torch.from_numpy(crop_rgb).permute(2, 0, 1)     # [3,256,256] uint8
                fmap, feat = self.siglip.encode_image_patch(t)              # [dim] CPU float (已 normalize)
                feats.append(feat.numpy())
                feat_maps.append(fmap.numpy())
            feats = np.array(feats) # [K_i,768]
            feat_maps = np.array(feat_maps) # [K_i, 1024, 768]

            feats_chunks.append(feats)               # [K_i, dim]
            patch_feature_chunks.append(feat_maps.reshape(-1, 768)) 
            # offsets.append((cursor, cursor + feats.shape[0]))
            # cursor += feats.shape[0]

        # 拼接并保存全局缓存
        if len(feats_chunks):
            feats_all = np.concatenate(feats_chunks, axis=0)        # [sum(K_i), dim]
            patch_feature_chunks_all = np.concatenate(patch_feature_chunks, axis=0) # [sum(K_i*1024), 768]
        else:
            print("No instance found in the dataset.")
            exit(0)
        # segmaps_all = np.stack(segmaps, axis=0)
        np.save(self.features_path, feats_all.astype(np.float32))
        np.save(self.patch_feats_path, patch_feature_chunks_all)