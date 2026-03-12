import os, sys
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
from loguru import logger
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)
from sam_encoder.sam_mask_extractor import SamVideoSegmenter
from siglip2_encoder.siglip2_extractor import SIGLIP2FeatureExtractor
from feat_comp_lqc.feature_compression import CVQFeatureCompressor

class SamSiglip2FeatureExtractor:
    def __init__(self, args):
        """
        Args:
            sam1_ckpt: SAM1 checkpoint path
            sam2_ckpt: SAM2 checkpoint path
            siglip2_config: siglip2 的配置文件路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_path = args.source_path
        # --- SAM part ---
        self.sam_extractor = SamVideoSegmenter(args)
        # --- Detic part ---
        self.feat_extractor = SIGLIP2FeatureExtractor()
        # --- featcomp part ---
        self.feat_comp = CVQFeatureCompressor()
        # --- output dirs ---
        self.start_dir = os.path.join(self.source_path, "start")
        self.end_dir = os.path.join(self.source_path, "end")
        self.start_out_dir = os.path.join(self.start_dir, "samsiglip2_embeddings")
        self.end_out_dir = os.path.join(self.end_dir, "samsiglip2_embeddings")
        os.makedirs(self.start_out_dir, exist_ok=True)
        os.makedirs(self.end_out_dir, exist_ok=True)
        # Step1: 拼接两种状态的 images
        self.start_names, self.start_paths = self.load_frames(os.path.join(self.start_dir, "images"))
        self.end_names, self.end_paths = self.load_frames(os.path.join(self.end_dir, "images"))
        self.all_frame_names = self.start_names + self.end_names
        self.all_frame_paths = self.start_paths + self.end_paths

    def load_frames(self, video_dir):
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        try:
            frame_names.sort(
                key=lambda p: int(os.path.splitext(p)[0])            )
        except:
            frame_names.sort(
                key=lambda p: os.path.splitext(p)[0]            )

        frame_paths = [os.path.join(video_dir, p) for p in frame_names]
        return frame_names, frame_paths
    
    def get_mask_patches(self, image, masks, target_size=(224, 224)):
        patches = []
        for mask in masks:
            # 获取裁剪 bbox
            x, y, w, h = map(int, mask['bbox'])
            patch = image[y:y+h, x:x+w]

            # pad 成正方形
            h_patch, w_patch, _ = patch.shape
            l = max(h_patch, w_patch)
            pad_img_array = np.zeros((l, l, 3), dtype=np.uint8)
            if h_patch > w_patch:
                pad_img_array[:, (l - w_patch)//2 : (l - w_patch)//2 + w_patch, :] = patch
            else:
                pad_img_array[(l - h_patch)//2 : (l - h_patch)//2 + h_patch, :, :] = patch

            # resize 到 target_size
            resized_patch = cv2.resize(pad_img_array, target_size)

            # 转为 torch tensor, C,H,W 并归一化到 [0,1]
            patch_tensor = torch.from_numpy(resized_patch.astype(np.float32)).permute(2,0,1) / 255.0
            patches.append(patch_tensor)

        if patches:
            return torch.stack(patches, dim=0)
        else:
            return torch.empty(0, 3, *target_size)
    
    def extract_features(self, frame_names, frame_paths):
        for frame_idx in tqdm(range(0, len(frame_paths)), desc="Sam+Detic"):
            img_path = frame_paths[frame_idx]
            # fname = os.path.basename(img_path)
            img_name = frame_names[frame_idx]
            
            # --- 判断属于 start 还是 end ---
            if img_path in self.start_paths:
                state = "start"
            elif img_path in self.end_paths:
                state = "end"
            else:
                raise ValueError(f"not valid: {img_path}")

            # --- 读取原始图像 ---
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --- Detic feature extraction ---
            detic_result = self.detic_extractor.extract_features(image)
            featuremap = detic_result["featuremap"]  # [512, H, W]

            idmask_path = os.path.join(
                os.path.dirname(img_path).replace("images", "id_masks"),
                f"{os.path.splitext(img_name)[0]}.npy"
            )
            if not os.path.exists(idmask_path):
                logger.warning(f"ID mask not found: {idmask_path}, skipping...")
                continue
            idmask = np.load(idmask_path)  # [Hi, Wi]

            # 根据idmask得到含有物体区域的tiles

            # --- 将 idmask resize 到 featuremap 大小 ---
            Hf, Wf = featuremap.shape[1:]
            idmask_resized = cv2.resize(
                idmask.astype(np.int32), (Wf, Hf), interpolation=cv2.INTER_NEAREST
            )
            pooled_featuremap = torch.zeros((4, featuremap.shape[1], featuremap.shape[2]),
                                dtype=torch.float32, device=self.device)

            for obj_id in np.unique(idmask_resized):
                if obj_id == -1:
                    continue
                mask = idmask_resized == obj_id  # [Hf, Wf]
                if mask.sum() == 0:
                    continue
                mask_tensor = torch.from_numpy(mask).to(featuremap.device)  # 转 tensor
                feat = featuremap[:, mask_tensor]       # [512, num_pixels]
                pooled_feat = feat.mean(dim=-1)         # [512]
                pooled_feat = pooled_feat / (pooled_feat.norm(p=2) + 1e-6)
                with torch.no_grad():
                    comp_pooled_feat = self.feat_comp.encode(pooled_feat.unsqueeze(0).to(self.device))
                    comp_pooled_feat = comp_pooled_feat.squeeze(0)
                pooled_featuremap[:, mask_tensor] = comp_pooled_feat[:, None]
            # with torch.no_grad():
            #     comp_featuremap = self.feat_comp.encode(pooled_featuremap.to(self.device))
            if state == "start":
                torch.save(pooled_featuremap,
                    os.path.join(self.start_out_dir, f"{os.path.splitext(img_name)[0]}.pt"))
            else:
                torch.save(pooled_featuremap,
                    os.path.join(self.end_out_dir, f"{os.path.splitext(img_name)[0]}.pt"))
            
            logger.info(f"Saved pooled featuremap for {state}/{img_name}")

    def process_dataset(self):
        if not os.path.exists(os.path.join(self.start_dir, "id_masks")) or not os.path.exists(os.path.join(self.end_dir, "id_masks")):
            self.sam_extractor.run()
        # self.extract_features(self.all_frame_names, self.all_frame_paths)      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,required=True)
    # parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--sam1_checkpoint",type=str,default="/home/ubuntu/TJH/Work/aff_ws/LangScene-X/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam2_checkpoint",type=str,default="/home/ubuntu/TJH/Work/aff_ws/LangScene-X/ckpt/sam2_hiera_large.pt")
    parser.add_argument("--detect_stride",type=int,default=5)
    parser.add_argument("--use_other_level",type=int,default=1)
    parser.add_argument("--postnms",type=int,default=1)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    parser.add_argument("--reverse", action="store_true")
    
    level_dict = {
        "default": 0,
        "small": 1, 
        "middle": 2,
        "large": 3
    }
    args = parser.parse_args()

    samsiglip2_extractor = SIGLIP2FeatureExtractor(args)
    samsiglip2_extractor.process_dataset()