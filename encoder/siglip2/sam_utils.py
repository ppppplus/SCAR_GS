import torch
import numpy as np
import cv2
from typing import List, Tuple
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch.nn.functional as F

def build_sam(mask_ckpt_path: str, device="cuda"):
    sam = sam_model_registry["vit_h"](checkpoint=mask_ckpt_path).to(device)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    return gen

def masks_to_segmap_and_crops(masks: List[dict], image_bgr_uint8: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    返回:
      seg_map: [H,W] int32, 每个像素是实例 index (-1 表示背景)
      crops:   长度 K 的 list，每个元素是裁成方形并 resize 到 256x256 的 uint8 RGB 小图 (H,W,3)
    """
    H, W = image_bgr_uint8.shape[:2]
    seg_map = -np.ones((H, W), dtype=np.int32)
    crops = []

    def crop_square(mask):
        seg = mask['segmentation']            # [H,W] bool
        img = image_bgr_uint8.copy()
        img[~seg] = 0
        x, y, w, h = np.int32(mask['bbox'])
        patch = img[y:y+h, x:x+w, :]
        # pad to square
        hh, ww = patch.shape[:2]
        L = max(hh, ww)
        canvas = np.zeros((L, L, 3), dtype=np.uint8)
        if hh >= ww:
            s = (hh - ww) // 2
            canvas[:, s:s+ww, :] = patch
        else:
            s = (ww - hh) // 2
            canvas[s:s+hh, :, :] = patch
        patch256 = cv2.resize(canvas, (256, 256))
        return cv2.cvtColor(patch256, cv2.COLOR_BGR2RGB)  # to RGB

    # 填 seg_map & 生成 crops
    for i, m in enumerate(masks[3]):
        seg_map[m['segmentation']] = i
        crops.append(crop_square(m))

    return seg_map, crops

def masks_to_segmap_and_crops_no_background(masks: List[dict], image_bgr_uint8: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    返回:
      seg_map: [H,W] int32, 每个像素是实例 index (从 0 开始)
      crops:   长度 K 的 list，每个元素是裁成方形并 resize 到 256x256 的 uint8 RGB 小图 (H,W,3)
      
    说明:
      - 背景填充为中性灰 (127,127,127)，避免黑边导致特征偏移
    """
    H, W = image_bgr_uint8.shape[:2]
    seg_map = -np.ones((H, W), dtype=np.int32)
    crops = []

    def crop_square(mask):
        seg = mask['segmentation']            # [H,W] bool
        x, y, w, h = np.int32(mask['bbox'])
        patch = image_bgr_uint8[y:y+h, x:x+w, :].copy()
        
        # pad to square with neutral gray background
        hh, ww = patch.shape[:2]
        L = max(hh, ww)
        canvas = np.full((L, L, 3), 127, dtype=np.uint8)  # 中性灰背景

        if hh >= ww:
            s = (hh - ww) // 2
            canvas[:, s:s+ww, :] = patch
        else:
            s = (ww - hh) // 2
            canvas[s:s+hh, :, :] = patch

        patch256 = cv2.resize(canvas, (256, 256))
        return cv2.cvtColor(patch256, cv2.COLOR_BGR2RGB)  # 转 RGB

    # 填 seg_map & 生成 crops
    for i, m in enumerate(masks[3]):
        seg_map[m['segmentation']] = i
        crops.append(crop_square(m))

    return seg_map, crops

def masks_to_segmap_and_crops_tensor(
    masks: List[dict],
    image_tensor: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    输入:
      masks: 每个元素包含 { 'segmentation': bool array [H,W], 'bbox': [x,y,w,h] }
      image_tensor: torch.Tensor, [3,H,W], float32, [0,1], RGB (ToTensor 后的输出)

    返回:
      seg_map: [H,W] int32 tensor, 每个像素是实例 index (-1 表示背景)
      crops:   长度 K 的 list，每个元素是 [3,256,256] float32 tensor, RGB, [0,1]
    """
    assert image_tensor.ndim == 3 and image_tensor.shape[0] == 3, "Expect [3,H,W] tensor"
    _, H, W = image_tensor.shape

    seg_map = torch.full((H, W), -1, dtype=torch.int32, device=image_tensor.device)
    crops = []

    def crop_square(mask):
        seg = torch.as_tensor(mask['segmentation'], dtype=torch.bool, device=image_tensor.device)  # [H,W]
        img = image_tensor.clone()
        img[:, ~seg] = 0.0

        x, y, w, h = map(int, mask['bbox'])
        patch = img[:, y:y+h, x:x+w]  # [3,h,w]

        hh, ww = patch.shape[1:]
        L = max(hh, ww)

        # pad 成方形
        canvas = torch.zeros((3, L, L), dtype=img.dtype, device=img.device)
        if hh >= ww:
            s = (hh - ww) // 2
            canvas[:, :, s:s+ww] = patch
        else:
            s = (ww - hh) // 2
            canvas[:, s:s+hh, :] = patch

        # resize 到 256x256
        patch256 = F.interpolate(canvas.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        return patch256.squeeze(0)  # [3,256,256]

    # 填 seg_map & 生成 crops
    for i, m in enumerate(masks[3]):
        seg = torch.as_tensor(m['segmentation'], dtype=torch.bool, device=image_tensor.device)
        seg_map[seg] = i
        crops.append(crop_square(m))

    return seg_map, crops