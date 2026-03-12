# 测试siglip2的query

import os, sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15
import argparse
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)
from siglip2.siglip2_utils import SIGLIP2NetWork
from siglip2.sam_utils import build_sam, masks_to_segmap_and_crops
from quantizer import VectorQuantizer
from quantizer_ae import VectorAE
from quantizer_vqae import VectorQuantizerAE

from clip.clip_utils import to_tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def text_image_match_heatmap(id_mask, seg_feats, text_feat, feature_extractor):
    """
    计算每个像素对应的文本匹配概率
    Args:
        id_mask: [H,W] numpy, 每个像素的物体ID
        seg_feats: [num_objects, C] torch tensor, 每个ID的特征
        text: str, query
    Returns:
        prob_map: [H,W] torch tensor
    """

    text_prob = feature_extractor.match_features(seg_feats, text_feat)
    print(text_prob.shape)

    H, W = id_mask.shape
    
    prob_map = torch.zeros((H, W), device=device)
    curr_id = 0
    for obj_id in np.unique(id_mask):
        if obj_id == -1:
            continue
        prob_map[id_mask == obj_id] = text_prob[curr_id].float()
        curr_id += 1

    return prob_map

def plot_heatmap(prob_map, image=None, alpha=0.6, cmap="jet"):
    prob_map = prob_map.cpu().numpy()
    plt.figure(figsize=(12, 8))
    if image is not None:
        plt.imshow(image)
    plt.imshow(prob_map, cmap=cmap, alpha=alpha)
    plt.colorbar(label="Text Match Probability")
    plt.axis("off")
    plt.show()

def get_image_features(data, id_mask, cursor, feature_dim=768):
    num = np.max(id_mask) + 1
    start = cursor
    end = cursor+num
    if end <= start:
        out = np.empty((0, feature_dim))
        return out, (start, end)

    # memmap 切片 → 拷贝成常规 ndarray（避免后续进程/线程问题）
    feats_np = np.array(data[start:end], copy=True)  # [K, dim]

    return feats_np, end

def sam_masks_to_id_mask(
    masks,
    shape=None,                 # (H,W)。不传则从第一个mask里推断
    order="score",              # "score" | "area_desc" | "area_asc" | "input"
    background_id=-1,           # 背景ID
    start_id=0,                 # 实例ID起始编号
    non_overwrite=False,        # True: 保留已赋值像素；False: 按顺序覆盖
):
    """
    将 SAM 的 masks(list of dict) 转换为 id_mask(int32)，并返回 id->mask 的索引映射。
    每个元素典型包含：
      m['segmentation'] : HxW bool
      m['area']         : int
      m['predicted_iou'], m['stability_score'] : float
    """
    if len(masks) == 0:
        if shape is None:
            raise ValueError("shape 必须提供（masks 为空）")
        return np.full(shape, background_id, dtype=np.int32), []

    # 推断尺寸
    if shape is None:
        seg0 = masks[0]["segmentation"]
        shape = seg0.shape

    # 排序策略（决定重叠像素谁留下）
    if order == "score":
        # 分数高的后覆盖（pred_iou * stability 更稳）
        def _score(m): return float(m.get("predicted_iou", 0.0)) * float(m.get("stability_score", 0.0))
        masks_sorted = sorted(masks, key=_score)  # 先低后高 → 后者覆盖
    elif order == "area_desc":
        masks_sorted = sorted(masks, key=lambda m: int(m.get("area", 0)))
    elif order == "area_asc":
        masks_sorted = sorted(masks, key=lambda m: int(m.get("area", 0)), reverse=True)
    elif order == "input":
        masks_sorted = list(masks)
    else:
        raise ValueError(f"未知 order: {order}")

    id_mask = np.full(shape, background_id, dtype=np.int32)
    keep_ids = []

    curr_id = start_id
    for m in masks_sorted:
        seg = m["segmentation"]
        if seg.shape != shape:
            raise ValueError("所有 segmentation 尺寸必须一致")
        if non_overwrite:
            # 只在尚未赋值的位置写入
            write = seg & (id_mask == background_id)
            if not write.any():
                continue
            id_mask[write] = curr_id
        else:
            # 直接用当前实例覆盖（排序越靠后优先级越高）
            id_mask[seg] = curr_id
        keep_ids.append(curr_id)
        curr_id += 1

    return id_mask, keep_ids

def visualize_image_and_mask_array(
    image: np.ndarray,              # [H,W,3] uint8 或 float，RGB 或 BGR
    mask: np.ndarray,               # [H,W] (int: id mask) 或 [H,W] (float: prob/gray)
    *,
    image_is_bgr: bool = False,     # True 则会转成 RGB 再显示
    alpha: float = 0.5,             # 叠加透明度
    cmap: str = "tab20",            # id mask 调色板
    background_id: int = -1,        # id mask 背景值
    draw_boundaries: bool = True,
    boundary_color=(1, 1, 1),       # 边界颜色 (RGB, 0~1)
    boundary_thickness: int = 1,
):
    """直接用数组可视化 image+mask 叠加，返回 [H,W,3] 的 float32(0~1)。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image 必须是 [H,W,3]，得到 {image.shape}")
    H, W = image.shape[:2]

    img = image.copy()
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    # 颜色空间
    if image_is_bgr:
        img = img[..., ::-1]  # BGR->RGB
    # 归一化到 0..1
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # 尺寸对齐
    if mask.shape[:2] != (H, W):
        if np.issubdtype(mask.dtype, np.integer):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    is_id_mask = np.issubdtype(mask.dtype, np.integer)

    overlay = img.copy()
    if is_id_mask:
        import matplotlib
        cmap_obj = matplotlib.cm.get_cmap(cmap)
        ids = np.unique(mask)
        ids = ids[ids != background_id]

        color_layer = np.zeros_like(overlay)
        alpha_layer = np.zeros((H, W), dtype=np.float32)

        for i, k in enumerate(ids):
            color = np.array(cmap_obj(i / max(len(ids), 1)))[:3]  # RGB
            region = (mask == int(k))
            color_layer[region] = color
            alpha_layer[region] = alpha

        vis = overlay * (1 - alpha_layer[..., None]) + color_layer * alpha_layer[..., None]

        if draw_boundaries and len(ids) > 0:
            kernel = np.ones((3, 3), np.uint8)
            for k in ids:
                region = (mask == int(k)).astype(np.uint8)
                if region.sum() == 0:
                    continue
                edge = cv2.dilate(region, kernel, iterations=boundary_thickness) - region
                vis[edge.astype(bool)] = boundary_color
    else:
        m = mask.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        heat = plt.get_cmap("jet")(m)[..., :3]  # 伪彩
        vis = overlay * (1 - alpha) + heat * alpha

        if draw_boundaries:
            thr = (m > 0.5).astype(np.uint8)
            edge = cv2.dilate(thr, np.ones((3, 3), np.uint8), iterations=boundary_thickness) - thr
            vis[edge.astype(bool)] = boundary_color

    vis = np.clip(vis, 0.0, 1.0).astype(np.float32)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis)
    plt.axis("off")
    plt.show()
    
from sklearn.preprocessing import StandardScaler, normalize
def plot_cosine_hist(X, pairs=5000):
    # cosine on L2 normalized features
    Xn = X.cpu().numpy()
    print(Xn.shape)
    Xn = normalize(Xn, norm="l2", axis=1)
    n = len(Xn)
    pairs = min(pairs, n * (n - 1) // 2)
    sims = []
    for _ in range(pairs):
        i, j = np.random.randint(0, n, 2)
        while j == i:
            j = np.random.randint(0, n)
        sims.append((Xn[i] * Xn[j]).sum())
    sims = np.array(sims)
    plt.figure()
    plt.hist(sims, bins=60)
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title("Random-pair cosine similarity")
    # savefig(os.path.join(outdir, "02_cosine_hist.png"))
    plt.show()


def main(img_paths, query_text, feature_extractor, mask_generator, resize_if_taller_than=1080):
    seg_feats_list = []
    for img_path in img_paths:
        image = load_image(img_path)
        img_bgr = cv2.imread(img_path)
        text_feat = feature_extractor.extract_text_features(query_text)
        if img_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")
        H0, W0 = img_bgr.shape[:2]
        if resize_if_taller_than > 0 and H0 > resize_if_taller_than:
            scale = H0 / float(resize_if_taller_than)
            img_bgr = cv2.resize(img_bgr, (int(W0/scale), int(H0/scale)))
        masks = mask_generator.generate(img_bgr)
        seg_map, crops = masks_to_segmap_and_crops(masks, img_bgr)

        # --- SigLIP2 编码每个实例 ---
        # if len(crops) == 0:
        #     offsets.append((cursor, cursor))
        #     continue
        feats = []
        for i, crop_rgb in enumerate(crops):
            t = torch.from_numpy(crop_rgb).permute(2, 0, 1)     # [3,256,256] uint8
            # t = Image.fromarray(crop_rgb.astype("uint8"))
            # t = to_tensor(t)
            f = feature_extractor.encode_image_tensor(t)              # [dim] CPU float (已 normalize)
            feats.append(f.numpy())
        feats = np.array(feats)
        print(seg_map.shape, feats.shape)
        # features, cursor = get_image_features(data, id_mask, cursor)
        seg_feats = torch.from_numpy(feats).float().to(device)
        seg_feats_list.append(seg_feats)
    all_feats = torch.cat(seg_feats_list, dim=0)
    
    plot_cosine_hist(all_feats)
    # prob_map = text_image_match_heatmap(seg_map, seg_feats, text_feat, feature_extractor)
    # plot_heatmap(prob_map, image=img_bgr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument('--image_path1', type=str, default="/home/ubuntu/TJH/Work/aff_ws/LEGaussians/my_scripts/data/0000.png")
    parser.add_argument('--image_path2', type=str, default="/home/ubuntu/TJH/Work/aff_ws/LEGaussians/my_scripts/data/0000.png")
    
    parser.add_argument('--query', type=str, default="bicycle")

    args = parser.parse_args()
    # feat_cache = np.load(os.path.join(cache_dir, "cache.npy"))
    # offset_cache = np.load(os.path.join(cache_dir, "offsets.npy"))
    # print(feat_cache.shape, offset_cache.shape)
    # id_mask_folder = os.path.join(args.source_path, "id_masks")
    # feature_folder = os.path.join(args.source_path, "samsiglip2_embeddings")
    # query_text = f"a {args.query}"
    sam_ckpt_path = "/home/ubuntu/Documents/TJH/model_zoo/sam/sam_vit_h_4b8939.pth"
    
    feature_extractor = SIGLIP2NetWork(device=device)
    mask_generator = build_sam(sam_ckpt_path, device=device)
    main([args.image_path1, args.image_path2], args.query, feature_extractor, mask_generator)
