import argparse
import os
import random
import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam_utils import Prompts, get_video_segments, save_idmasks, search_new_obj, cal_no_mask_area_ratio, masks_update
# 这里直接用之前已有的工具函数 (mask_nms, masks_update, Prompts, get_video_segments, save_idmasks, etc.)
# 你可以放到一个 utils.py 文件里，然后 import 进来


class SamVideoSegmenter:
    def __init__(self, args):
        self.args = args
        self.source_path = args.source_path
        self.start_dir = os.path.join(self.source_path, "start")
        self.end_dir = os.path.join(self.source_path, "end")
        self.start_out_dir = os.path.join(self.start_dir, "id_masks")
        self.end_out_dir = os.path.join(self.end_dir, "id_masks")
        self.load = args.load
        self.save = args.save
        # os.makedirs(self.start_out_dir, exist_ok=True)
        # os.makedirs(self.end_out_dir, exist_ok=True)
        self.level_dict = {"default": 0, "small": 1, "middle": 2, "large": 3}
        
        # os.makedirs(args.output_dir, exist_ok=True)
        # logger.add(os.path.join(args.output_dir, f'{args.level}.log'), rotation="500 MB")
        logger.info(f"[Init] Source={self.source_path}, start={self.start_dir}, end={self.end_dir}")

        # SAM1
        self.sam = sam_model_registry["vit_h"](checkpoint=args.sam1_checkpoint).to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=args.pred_iou_thresh,
            box_nms_thresh=args.box_nms_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

        # SAM2
        self.predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", args.sam2_checkpoint)
        self.sam2 = build_sam2("sam2_hiera_l.yaml", args.sam2_checkpoint, device="cuda", apply_postprocessing=False)

    def load_frames(self, video_dir):
        if not os.path.exists(video_dir):
            return [], []
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        try:
            frame_names.sort(
                key=lambda p: int(os.path.splitext(p)[0]),
                reverse=self.args.reverse
            )
        except:
            frame_names.sort(
                key=lambda p: os.path.splitext(p)[0],
                reverse=self.args.reverse
            )

        frame_paths = [os.path.join(video_dir, p) for p in frame_names]
        return frame_names, frame_paths

    def run(self):
        # --- 1. 加载 start + end 图片和权重---
        start_names, start_paths = self.load_frames(os.path.join(self.start_dir, "images"))
        end_names, end_paths = self.load_frames(os.path.join(self.end_dir, "images"))
        # all_frame_names = start_names + end_names
        all_frame_paths = start_paths + end_paths
        logger.info(f"[Run] start={len(start_names)} frames, end={len(start_names)} frames, total={len(all_frame_paths)}")

        if self.load:
            state_path = os.path.join(self.start_dir, "sam_state.pt")
            if os.path.exists(state_path):
                logger.info(f"[Run] Loading saved SAM2 propagation state from {state_path}")
                prompts_loader, inference_state = self.load_state(state_path)
            else:
                raise FileNotFoundError(f"[Error] load_state=True but no state file found: {state_path}")
            # --- 2. 只传播保存end部分 ---
            video_segments = self.process_frames(end_paths, prompts_loader, inference_state)
            self.save_results(frame_names=end_names, video_segments=video_segments, stage=1, offset=len(start_names))

        else:
            logger.info("[Run] Running propagation from scratch")
            inference_state = self.predictor.init_state_from_frames(frame_paths=all_frame_paths)
            prompts_loader = Prompts(bs=self.args.batch_size)

            # --- 2. 一次性mask传播 ---
            video_segments = self.process_frames(all_frame_paths, prompts_loader, inference_state)

            # --- 3. 拆分保存 ---
            self.save_results(frame_names=start_names, video_segments=video_segments, stage=0, offset=0)
            self.save_results(frame_names=end_names, video_segments=video_segments, stage=1, offset=len(start_names))

        # --- 4. 保存传播状态 ---
        if self.save:
            self.save_state(prompts_loader, inference_state, os.path.join(self.start_dir, "sam_state.pt"))

    def process_frames(self, frame_paths, prompts_loader, inference_state):
        """
        跟原来的 process_folder 类似，但处理拼接后的所有帧
        """
        now_frame = 0
        masks_from_prev, sum_id = [], 0
        while True:
            logger.info(f"[process] frame: {now_frame}")
            sum_id = prompts_loader.get_obj_num()
            # image_path = os.path.join(self.start_dir if now_frame < len(frame_names)//2 else self.end_dir, frame_names[now_frame])
            image_path = frame_paths[now_frame]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[0] > 1080:
                scale = 1080 / image.shape[0]
                h = int(image.shape[0] * scale)
                w = int(image.shape[1] * scale)
                image = cv2.resize(image, (w, h))

            all_masks = self.mask_generator.generate(image)
            masks = all_masks[self.level_dict[self.args.level]]
            if self.args.postnms:
                masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

            if now_frame == 0:
                for ann_obj_id, m in enumerate(masks):
                    prompts_loader.add(ann_obj_id, 0, m["segmentation"])
            else:
                new_mask_list = search_new_obj(masks_from_prev=masks_from_prev, mask_list=masks, mask_ratio_thresh=mask_ratio_thresh)
                for id, m in enumerate(masks_from_prev):
                    if m.sum() > 0:
                        prompts_loader.add(id, now_frame, m[0])
                for i, new_mask in enumerate(new_mask_list):
                    prompts_loader.add(sum_id + i, now_frame, new_mask["segmentation"])

            if now_frame == 0 or len(new_mask_list) != 0:
                video_segments = get_video_segments(prompts_loader, self.predictor, inference_state)

            max_area_no_mask = (0, -1)
            for out_frame_idx in range(now_frame, len(frame_paths), self.args.detect_stride):
                out_mask_list = list(video_segments[out_frame_idx].values())
                no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
                if now_frame == out_frame_idx:
                    mask_ratio_thresh = no_mask_ratio
                if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
                    masks_from_prev = out_mask_list
                    max_area_no_mask = (no_mask_ratio, out_frame_idx)
                    break
            if max_area_no_mask[1] == -1:
                break
            now_frame = max_area_no_mask[1]
        return get_video_segments(prompts_loader, self.predictor, inference_state, final_output=True)

    def save_results(self, frame_names, video_segments, stage=0, offset=0):
        # save_dir = os.path.join(self.output_dir, self.args.level, f"{tag}-final-output")
        save_dir = self.start_out_dir if stage == 0 else self.end_out_dir
        os.makedirs(save_dir, exist_ok=True)

        for local_idx, frame_name in enumerate(frame_names):
            global_idx = local_idx + offset
            out_obj_id_list = list(video_segments[global_idx].keys())
            out_mask_list = list(video_segments[global_idx].values())
            save_idmasks(out_obj_id_list, out_mask_list, local_idx, save_dir)

        logger.info(f"Saved {len(frame_names)} frames to {save_dir}")
    
    def save_state(self, prompts_loader, inference_state, save_path):
        state = {
            "prompts": prompts_loader.save(),  # 你需要在 utils.Prompts 实现 save()
            "inference_state": inference_state,  # 是 dict，可以直接保存
        }
        torch.save(state, save_path)
        logger.info(f"[State Saved] {save_path}")

    def load_state(self, load_path):
        state = torch.load(load_path, map_location="cuda")

        # 恢复 prompts_loader
        prompts_loader = Prompts(bs=self.args.batch_size)
        prompts_loader.load(state["prompts"])  # 你要在 utils.Prompts 里实现 load()

        # 恢复 SAM2 推理状态
        inference_state = state["inference_state"]

        logger.info(f"[State Loaded] {load_path}")
        return prompts_loader, inference_state



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--level", choices=["default","small","middle","large"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sam1_checkpoint", type=str, required=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--detect_stride", type=int, default=5)
    parser.add_argument("--use_other_level", type=int, default=1)
    parser.add_argument("--postnms", type=int, default=1)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.7)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    seg = SamVideoSegmenter(args)
    seg.run()