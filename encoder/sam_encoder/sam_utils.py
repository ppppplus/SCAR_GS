import argparse
import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    mask_chunk_size = 20
    
    mask_chunks = masks_ord.split(mask_chunk_size, dim=0)
    area_chunks = masks_area.split(mask_chunk_size, dim=0)

    iou_matrix = []
    inner_iou_matrix = []

    for i_areas, i_chunk in zip(area_chunks, mask_chunks):
        row_iou_matrix = []
        row_inner_iou_matrix = []
        for j_areas, j_chunk in zip(area_chunks, mask_chunks):
            intersection = torch.logical_and(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            union = torch.logical_or(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            local_iou_mat = intersection / union 
            row_iou_matrix.append(local_iou_mat)

            row_inter_mat = intersection / i_areas[:, None]
            col_inter_mat = intersection / j_areas[None, :]

            inter = torch.logical_and(row_inter_mat < 0.5, col_inter_mat >= 0.85)

            local_inner_iou_mat = torch.zeros((len(i_areas), len(j_areas)))
            local_inner_iou_mat[inter] = 1 - row_inter_mat[inter] * col_inter_mat[inter]
            row_inner_iou_matrix.append(local_inner_iou_mat)

        row_iou_matrix = torch.cat(row_iou_matrix, dim=1)
        row_inner_iou_matrix = torch.cat(row_inner_iou_matrix, dim=1)
        iou_matrix.append(row_iou_matrix)
        inner_iou_matrix.append(row_inner_iou_matrix)
    iou_matrix = torch.cat(iou_matrix, dim=0)
    inner_iou_matrix = torch.cat(inner_iou_matrix, dim=0)

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if isinstance(masks_lvl, tuple):
            masks_lvl = masks_lvl[0]  # If it's a tuple, take the first element
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
            
        # Check if masks_lvl is a list of dictionaries
        if isinstance(masks_lvl[0], dict):
            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        else:
            # If it's a direct list of masks, use them directly
            seg_pred = torch.from_numpy(np.stack(masks_lvl, axis=0))
            # Create default values for cases without iou and stability
            iou_pred = torch.ones(len(masks_lvl))
            stability = torch.ones(len(masks_lvl))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def get_patch(mask, image):
    image = image.copy()
    image[mask==0] = np.array([0, 0,  0], dtype=np.uint8)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = image[y0:y1, x0:x1]
    return patch

def idmask2patches(image: np.array, id_mask: np.array, patch_size=256):
    """
    Get image patches by idmask    

    Args:
        image (np.array): has shape (H, W, 3)
        masks (np.array): has shape (num_masks, H, W)
    Returns:
        patches (np.array): has shape (num_masks, patch_size, patch_size)
    """
    unique_ids = np.unique(id_mask)
    unique_ids = unique_ids[unique_ids != -1] 
    patch_list = []
    for i, obj_id in enumerate(unique_ids):
        mask_bool = id_mask == obj_id
        if mask_bool.sum() == 0:
            continue
        patch = get_patch(mask_bool, image)
        pad_patch = cv2.resize(pad_img(patch), (patch_size,patch_size))
        patch_list.append(pad_patch)
    patches = np.stack(patch_list, axis=0)
    return patches

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # Create image object
    image = Image.fromarray(image_array[0])

    # Save image
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

def save_masks(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),np.array(mask_list))

def save_idmasks(id_list, mask_list, frame_idx, save_dir, background_id=-1):
    # os.makedirs(save_dir, exist_ok=True)
    _, H, W = mask_list[0].shape
    instance_map = np.full((H, W), background_id, dtype=np.int32)

    # 将掩码写入 instance map
    for obj_id, mask in zip(id_list, mask_list):
        mask_2d = mask.squeeze()  
        instance_map[mask_2d.astype(bool)] = obj_id

    # 保存为 npy 文件
    save_path = os.path.join(save_dir, f"{frame_idx:04d}.npy")
    np.save(save_path, instance_map)
    # print(f"[INFO] Saved {save_path}, shape={instance_map.shape}")

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def make_enlarge_bbox(origin_bbox, max_width,max_height,ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0]-width*(ratio-1)/2,0),max(origin_bbox[1]-height*(ratio-1)/2,0)]
    new_box.append(min(width*ratio,max_width-new_box[0]))
    new_box.append(min(height*ratio,max_height-new_box[1]))
    return new_box

def sample_points(masks, enlarge_bbox,positive_num=1,negtive_num=40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x]==True and positive_count>0:
            output_points.append((x,y,1))
            positive_count-=1
        elif masks[y][x]==False and negtive_count>0:
            output_points.append((x,y,0))
            negtive_count-=1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points

def sample_points_from_mask(mask):
    # Get indices of all True values
    true_indices = np.argwhere(mask)

    # Check if there are any True values
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")

    # Randomly select a point from True value indices
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]

    return tuple(sample_point)

def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # Calculate mask_none, representing areas not included in any previous masks
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
    logger.info(len(new_mask_list))
    logger.info("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    logger.info("expected ratios:",mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: # Still a lot of gaps, greater than current thresh
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    logger.info(len(new_mask_list))

    return new_mask_list

def get_bbox_from_mask(mask):
    # Get row and column indices of non-zero elements
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find min and max indices of non-zero rows and columns
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Calculate width and height
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    return xmin, ymin, width, height

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


# class Prompts:
#     def __init__(self,bs:int):
#         self.batch_size = bs
#         self.prompts = {}
#         self.obj_list = []
#         self.key_frame_list = []
#         self.key_frame_obj_begin_list = []

#     def add(self,obj_id,frame_id,mask):
#         if obj_id not in self.obj_list:
#             new_obj = True
#             self.prompts[obj_id] = []
#             self.obj_list.append(obj_id)
#         else:
#             new_obj = False
#         self.prompts[obj_id].append((frame_id,mask))
#         if frame_id not in self.key_frame_list and new_obj:
#             # import ipdb; ipdb.set_trace()
#             self.key_frame_list.append(frame_id)
#             self.key_frame_obj_begin_list.append(obj_id)
#             logger.info("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
#     def get_obj_num(self):
#         return len(self.obj_list)
    
#     # def __len__(self):
#     #     if self.obj_list % self.batch_size == 0:
#     #         return len(self.obj_list) // self.batch_size
#     #     else:
#     #         return len(self.obj_list) // self.batch_size +1
#     def __len__(self):
#         n = len(self.obj_list)
#         return (n + self.batch_size - 1) // self.batch_size
    
#     def __iter__(self):
#         # self.batch_index = 0
#         self.start_idx = 0
#         self.iter_frameindex = 0
#         return self

#     def __next__(self):
#         if self.start_idx < len(self.obj_list):
#             if self.iter_frameindex == len(self.key_frame_list)-1:
#                 end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
#             else:
#                 if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
#                     end_idx = self.start_idx+self.batch_size
#                 else:
#                     end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
#                     self.iter_frameindex+=1
#                 # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
#             batch_keys = self.obj_list[self.start_idx:end_idx]
#             batch_prompts = {key: self.prompts[key] for key in batch_keys}
#             self.start_idx = end_idx
#             return batch_prompts
#         # if self.batch_index * self.batch_size < len(self.obj_list):
#         #     start_idx = self.batch_index * self.batch_size
#         #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
#         #     batch_keys = self.obj_list[start_idx:end_idx]
#         #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
#         #     self.batch_index += 1
#         #     return batch_prompts
#         else:
#             raise StopIteration
    
#     def save(self):
#         data = {
#             "batch_size": self.batch_size,
#             "obj_list": self.obj_list,
#             "key_frame_list": self.key_frame_list,
#             "key_frame_obj_begin_list": self.key_frame_obj_begin_list,
#             "prompts": {},
#         }

#         # prompts[obj_id] = [(frame_idx, mask), ...]
#         for obj_id, plist in self.prompts.items():
#             data["prompts"][obj_id] = []
#             for frame_idx, mask in plist:
#                 # mask 转为 numpy uint8 以便保存
#                 data["prompts"][obj_id].append(
#                     (frame_idx, mask.astype(np.uint8))
#                 )

#         return data

#     def load(self, data):
#         self.batch_size = data["batch_size"]
#         self.obj_list = data["obj_list"]
#         self.key_frame_list = data["key_frame_list"]
#         self.key_frame_obj_begin_list = data["key_frame_obj_begin_list"]

#         self.prompts = {}
#         for obj_id, plist in data["prompts"].items():
#             self.prompts[obj_id] = []
#             for frame_idx, mask in plist:
#                 self.prompts[obj_id].append(
#                     (frame_idx, mask.astype(np.uint8))
#                 )

class Prompts:
    def __init__(self, bs: int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        # 这里存的是 “在 obj_list 里的起始 index”，不是 obj_id
        self.key_frame_obj_begin_list = []

    def add(self, obj_id, frame_id, mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False

        self.prompts[obj_id].append((frame_id, mask))

        # 当某一帧第一次引入新物体时，认为它是关键帧，并记录该帧新物体起点在 obj_list 的 index
        if new_obj and (frame_id not in self.key_frame_list):
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(len(self.obj_list) - 1)  # ✅ index
            logger.info(f"key_frame_obj_begin_list: {self.key_frame_obj_begin_list}")

    def get_obj_num(self):
        return len(self.obj_list)

    def get_next_id(self):
        return (max(self.obj_list) + 1) if len(self.obj_list) > 0 else 0

    def __len__(self):
        n = len(self.obj_list)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx >= len(self.obj_list):
            raise StopIteration

        # 默认按 batch_size 切
        end_idx = min(self.start_idx + self.batch_size, len(self.obj_list))

        # 但如果下一个关键帧起点落在本 batch 中间，就截断到关键帧起点
        if self.iter_frameindex < len(self.key_frame_obj_begin_list) - 1:
            next_begin = self.key_frame_obj_begin_list[self.iter_frameindex + 1]
            if self.start_idx < next_begin < end_idx:
                end_idx = next_begin
                self.iter_frameindex += 1

        batch_keys = self.obj_list[self.start_idx:end_idx]
        batch_prompts = {k: self.prompts[k] for k in batch_keys}
        self.start_idx = end_idx
        return batch_prompts
    
    def save(self):
        data = {
            "batch_size": self.batch_size,
            "obj_list": self.obj_list,
            "key_frame_list": self.key_frame_list,
            "key_frame_obj_begin_list": self.key_frame_obj_begin_list,
            "prompts": {},
        }

        # prompts[obj_id] = [(frame_idx, mask), ...]
        for obj_id, plist in self.prompts.items():
            data["prompts"][obj_id] = []
            for frame_idx, mask in plist:
                # mask 转为 numpy uint8 以便保存
                data["prompts"][obj_id].append(
                    (frame_idx, mask.astype(np.uint8))
                )

        return data

    def load(self, data):
        self.batch_size = data["batch_size"]
        self.obj_list = data["obj_list"]
        self.key_frame_list = data["key_frame_list"]
        self.key_frame_obj_begin_list = data["key_frame_obj_begin_list"]

        self.prompts = {}
        for obj_id, plist in data["prompts"].items():
            self.prompts[obj_id] = []
            for frame_idx, mask in plist:
                self.prompts[obj_id].append(
                    (frame_idx, mask.astype(np.uint8))
                )
        
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False):

    video_segments = {}
    for batch_prompts in tqdm(prompts_loader,desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
        # start_frame_idx = 0 if final_output else None
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = { }
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
        
        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments

def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        reroll = True
        iter_cnt = 0
        while reroll and iter_cnt < 100:
            iter_cnt += 1
            reroll = False
            color = tuple(random.randint(1, 255) for _ in range(3))
            for selected_color in colors:
                if np.linalg.norm(np.array(color) - np.array(selected_color)) < 70:
                    reroll = True
                    break
        colors.append(color)
    return colors
