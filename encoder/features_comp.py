import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import configargparse

from siglip2.siglip2_utils import SIGLIP2NetWork
from feat_comp.ae import VectorAE
from feat_comp.vqae import VectorQuantizerAE
from feat_comp.lqc import VectorQuantizerLQC

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#   工具：根据 id_mask 获取每个 object 的 bbox
# ============================================================
def crop_instance_from_id_mask(image_bgr_uint8: np.ndarray, id_mask: np.ndarray, oid: int):
    """
    参照 SAM 的 crop_square() 实现:
      - 抹掉 mask 以外区域（填 0）
      - 裁出 bbox
      - pad 成正方形
      - resize 到 256x256
      - 输出 RGB uint8

    输入：
        image_bgr_uint8: 原图 BGR uint8
        id_mask: [H,W] int
        oid: object id

    返回：
        crop_rgb: [256,256,3] uint8 RGB
    """
    # 1) 得到 binary mask
    seg = (id_mask == oid)    # bool mask
    if seg.sum() == 0:
        return None

    H, W = id_mask.shape

    # 2) 抹掉非 mask 区域
    img = image_bgr_uint8.copy()
    img[~seg] = 0

    # 3) bbox
    ys, xs = np.where(seg)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    patch = img[y1:y2+1, x1:x2+1, :]    # BGR

    # 4) pad 成正方形
    hh, ww = patch.shape[:2]
    L = max(hh, ww)
    canvas = np.zeros((L, L, 3), dtype=np.uint8)

    if hh >= ww:
        s = (hh - ww) // 2
        canvas[:, s:s+ww, :] = patch
    else:
        s = (ww - hh) // 2
        canvas[s:s+hh, :, :] = patch

    # 5) resize → 256
    patch256 = cv2.resize(canvas, (256, 256))

    # 6) BGR → RGB
    patch256_rgb = cv2.cvtColor(patch256, cv2.COLOR_BGR2RGB)

    return patch256_rgb

def idmask_to_crops(id_mask: np.ndarray, image_bgr_uint8: np.ndarray):
    """
    输入:
        id_mask: [H,W] int
        image_bgr_uint8: 原始 BGR 图

    输出:
        obj_ids: list of object id
        crops: list of 256×256 RGB crop
    """
    obj_ids = np.unique(id_mask)
    obj_ids = obj_ids[obj_ids >= 0]  # 去除背景 -1

    crops = []
    for oid in obj_ids:
        crop_rgb = crop_instance_from_id_mask(image_bgr_uint8, id_mask, oid)
        if crop_rgb is not None:
            crops.append(crop_rgb)

    return obj_ids.tolist(), crops

# ============================================================
#   提取单张图像所有 object 的特征
# ============================================================
def extract_object_features(img, id_mask, feature_extractor, f_dim, device="cuda"):
    """
    输入:
        img:      [H,W,3]  BGR numpy
        id_mask:  [H,W]    每像素 object id
        feature_extractor.encode_image_tensor(x) → [f_dim]

    输出:
        obj_feats:  [N_obj, f_dim] torch.float32
        obj_ids:    [N_obj]        torch.int32
    """
    obj_ids, crops = idmask_to_crops(id_mask, img)

    feats = []
    if len(crops) == 0:
        return torch.zeros(0, f_dim), torch.zeros(0, dtype=torch.int32)
    for crop_rgb in crops:
        t = torch.from_numpy(crop_rgb).permute(2, 0, 1)  # [3,256,256]
        feat = feature_extractor.encode_image_tensor(t)   # [feat_dim]
        feats.append(feat.cpu())

    feats = torch.stack(feats, dim=0)  # [N_obj, f_dim]
    obj_ids = torch.tensor(obj_ids)

    return feats, obj_ids

# ============================================================
#   降维：将 [C,H,W] → [e_dim,H,W]
# ============================================================
def compress_object_features_to_map(obj_feats, obj_ids, id_mask, compressor, device="cuda"):
    """
    输入:
        obj_feats: [N_obj, f_dim]     object-level features
        obj_ids:   [N_obj]            对应 id_mask 的 object id
        id_mask:   [H, W]             每个像素属于哪个 object
        compressor: AE / VQAE / LQC

    输出:
        low_map:   [e_dim, H, W]      降维后的 feature map
    """

    H, W = id_mask.shape
    obj_feats = obj_feats.to(device)

    # ---- 1) object-level 降维 ----
    with torch.no_grad():
        if isinstance(compressor, VectorAE):
            lat = compressor.encode_vectors(obj_feats)         # [N_obj, e_dim]

        elif isinstance(compressor, VectorQuantizerAE):
            _, hq = compressor.encode_vectors(obj_feats[None])          # [1, N_obj, e_dim]
            lat = hq[0]

        elif isinstance(compressor, VectorQuantizerLQC):
            _, hq = compressor.encode_quantize_vectors(obj_feats[None]) # [1, N_obj, e_dim]
            lat = hq[0]

        else:
            raise ValueError("Unknown compressor type.")

    # lat: [N_obj, e_dim]
    lat = lat.cpu().numpy()
    obj_ids = obj_ids.cpu().numpy()

    # ---- 2) 拼回一个[e_dim, H, W] ----
    e_dim = lat.shape[1]
    low_map = np.zeros((e_dim, H, W), dtype=np.float32)

    for oid, vec in zip(obj_ids, lat):
        # vec shape: [e_dim]
        mask = (id_mask == oid)
        for c in range(e_dim):
            low_map[c, mask] = vec[c]

    return low_map


# ============================================================
#   主函数
# ============================================================
def main(args):
    for state in ["start", "end"]:
        source_path = os.path.join(args.source_path, state)
        if not os.path.exists(source_path):
            print(f"[WARN] Missing {source_path}")
            exit()
        id_mask_dir = os.path.join(source_path, "id_masks")
        image_dir = os.path.join(source_path, "images")

        # 保存路径
        feat_dir = os.path.join(source_path, f"{args.model_type}_feat")
        comp_dir = os.path.join(source_path, f"{args.model_type}_feat_dim{args.e_dim}")

        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)

        if args.model_type == "siglip2":
            feature_extractor = SIGLIP2NetWork(device=device)
        # elif args.model_type == "clip":
        #     feature_extractor = OpenCLIPNetwork(
        #         OpenCLIPNetworkConfig(
        #             clip_model_type="ViT-B-16",
        #             clip_model_pretrained="laion2b_s34b_b88k"
        #         )
        #     )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        if args.method == "ae":
            compressor = VectorAE(args.f_dim, args.e_dim, device)
        elif args.method == "vqae":
            compressor = VectorQuantizerAE(args.f_dim, args.n_e, args.e_dim, args.beta, device)
        elif args.method == "lqc":
            compressor = VectorQuantizerLQC(args.f_dim, args.n_e, args.e_dim, args.beta, device)
        else:
            raise ValueError("Unknown method")

        if args.ckpt is not None:
            sd = torch.load(args.ckpt, map_location=device)
            compressor.load_state_dict(sd["model_state"])
            compressor.to(device).eval()

        # 全部 mask 文件
        mask_files = sorted([f for f in os.listdir(id_mask_dir) if f.endswith(".npy")])

        for fname in tqdm(mask_files, desc=f"Processing masks of {state}"):

            base = fname.replace(".npy", "")
            mask_path = os.path.join(id_mask_dir, fname)
            image_path = os.path.join(image_dir, base + ".png")

            id_mask = np.load(mask_path)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[WARN] Missing image {image_path}")
                continue

            # ① 提取 feature_map
            obj_feats, obj_ids = extract_object_features(img, id_mask, feature_extractor, args.f_dim)
            np.save(os.path.join(feat_dir, base + "_f.npy"), obj_feats.cpu().numpy())
            np.save(os.path.join(feat_dir, base + "_ids.npy"), obj_ids.cpu().numpy())


            # ② 降维
            comp_map = compress_object_features_to_map(obj_feats, obj_ids, id_mask, compressor)
            np.save(os.path.join(comp_dir, base + ".npy"), comp_map)

        print("Done.")

def parse_args():
    parser = configargparse.ArgParser(
        description="Feature extraction + compression",
        # config_file_parser_class=configargparse.YAMLConfigFileParser,
        args_for_setting_config_path=["--config"],
    )
    # parser.add_argument('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument("--source_path", type=str, required=True, help="目录，其中含 id_mask 与 images")
    parser.add_argument("--model_type", type=str, default="siglip2")
    parser.add_argument("--method", type=str, default="lqc",
                        choices=["ae", "vqae", "lqc"])
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--f_dim", type=int, default=768)
    parser.add_argument("--n_e", type=int, default=128)
    parser.add_argument("--e_dim", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.)
    args, unknown = parser.parse_known_args()
    return args

# ============================================================
#   输入source_path， 输出特征和降维特征并保存
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)