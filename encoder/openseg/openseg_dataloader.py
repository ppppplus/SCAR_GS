import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
# import tensorflow as tf2
# import tensorflow._api.v2.compat.v1 as tf


class OpenSegDataset(Dataset):
    """
    - 输入：OpenSeg 提取后的 lang_features 目录
    - 每张图片对应一个 {i:04d}.npy 文件，存储 [K_i, 768] 特征
    - 加载后拼接为 [N, 768] 全局特征矩阵
    - 输出：每次 __getitem__ 返回一个 [768] 向量（float32）
    """

    def __init__(self,
                 data_root: str,
                 openseg_ckpt_path: str = "/home/ubuntu/Documents/TJH/model_zoo/openseg/openseg_exported_clip",
                 device: str = "cuda",
                 feature_dim: int = 768,
                 cache_subdir: str = "openseg_features"):
        super().__init__()
        self.data_root = data_root
        self.model_path = openseg_ckpt_path
        self.feature_dim = feature_dim
        self.device = device
        self.image_dir = self.data_root

        # 缓存目录
        self.cache_dir = os.path.join(self.data_root, cache_subdir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "cache.npy")

        self.try_load()

    def try_load(self):
        """优先加载缓存，否则重新拼接"""
        try:
            # self.data = np.load(self.cache_path, mmap_mode="r")
            self.data = self.load_cache()
        except (FileNotFoundError, ValueError):
            self._build_cache()
            # self.data = np.load(self.cache_path, mmap_mode="r")
            self.data = self.load_cache()

    def load_cache(self):
        """加载缓存"""
        cache_dir = Path(self.cache_dir)
        npy_files = sorted([p for p in cache_dir.glob("*.npy") if p.is_file()])
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {cache_dir}")

        feats_all = []
        for p in tqdm(npy_files, desc=f"Building cache"):
            arr = np.load(p, mmap_mode="r")
            # 自动适配维度 [768,H,W] 或 [H,W,768]
            if arr.ndim != 3:
                raise ValueError(f"{p} has shape {arr.shape}, expected 3D array.")
            if arr.shape[-1] == 768:
                arr = np.transpose(arr, (2, 0, 1))  # -> [768,H,W]
            elif arr.shape[0] != 768:
                raise ValueError(f"{p} has invalid shape {arr.shape}")
            # feats = arr.reshape(-1, 768)
            feats_all.append(arr.astype(dtype=np.float32, copy=False))

        # 拼接为大矩阵 [N,768]
        # all_feats = np.concatenate(feats_all, axis=0)
        return feats_all

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].astype(np.float32)

    # ---------- 关键逻辑 ----------
    def _build_cache(self):
        print(f"[OpenSegDataset] Extracting dense features from {self.image_dir} ...")

        # ---- 初始化 TensorFlow 环境 ----
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # ---- 加载 OpenSeg 模型 ----
        openseg = tf2.saved_model.load(
            str(self.model_path),
            tags=[tf.saved_model.tag_constants.SERVING]
        )

        # ---- 获取图像文件列表 ----
        img_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".png"))
        ])

        all_feats = []
        for idx, img_name in enumerate(tqdm(img_files, desc="Extracting OpenSeg features")):
            img_path = os.path.join(self.image_dir, img_name)

            # ---- 读取图像 ----
            with tf.gfile.GFile(str(img_path), 'rb') as f:
                np_image_string = np.array([f.read()])

            # ---- 前向推理 ----
            text_emb = tf.zeros([1, 1, self.feature_dim])  # 空文本嵌入
            results = openseg.signatures["serving_default"](
                inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
                inp_text_emb=text_emb
            )

            img_info = results['image_info']
            crop_sz = [
                int(img_info[0, 0] * img_info[2, 0]),
                int(img_info[0, 1] * img_info[2, 1])
            ]

            # 提取 dense feature map
            image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
            target_h = int(img_info[0, 0] // 2)
            target_w = int(img_info[0, 1] // 2)

            feat_2d = tf.cast(
                tf.image.resize(
                    image_embedding_feat, (target_h, target_w),
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )[0],
                dtype=tf.float32
            ).numpy()
            # feat_2d = tf.cast(
            #     tf.image.resize_nearest_neighbor(
            #         image_embedding_feat,
            #         (int(img_info[0, 0]), int(img_info[0, 1])),
            #         align_corners=True
            #     )[0],
            #     dtype=tf.float32
            # ).numpy()

            feat_2d = np.transpose(feat_2d, (2, 0, 1))  # [768, H, W]
            feat_2d = feat_2d / np.linalg.norm(feat_2d, axis=0, keepdims=True)

            # ---- 保存每张图的特征文件 ----
            np.save(os.path.join(self.cache_dir, f"{idx:04d}.npy"), feat_2d)
            # all_feats.append(feat_2d.reshape(self.feature_dim, -1).T)  # [H*W, 768]

        # ---- 拼接为全局缓存 ----
        # feats_all = np.concatenate(all_feats, axis=0)
        # np.save(self.cache_path, feats_all.astype(np.float32))
        # print(f"[OpenSegDataset] Cached {feats_all.shape[0]} features to {self.cache_path}")
