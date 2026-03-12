# SAM1+CLIP preprocessing code
import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')
from PIL import Image

import torch
import torchvision
from torch import nn
from open_clip import create_model_from_pretrained, get_tokenizer 

class SIGLIP2NetWork:
    def __init__(self, device="cuda"):
        self.device = device
        local_model_path = "local-dir:/home/ubuntu/Documents/TJH/model_zoo/siglip2/timm-ViT-B-16-SigLIP2-512"
        self.model, self.preprocess = create_model_from_pretrained(model_name=local_model_path, device=device)
        self.tokenizer = get_tokenizer(local_model_path)
    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        feature = self.model.encode_image(image, normalize=True)
        return feature
def plot_patches(patch_images, cols=5):
    """
    绘制所有 patch
    """
    rows = (len(patch_images) + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for i, patch in enumerate(patch_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(patch)
        plt.axis("off")
    plt.show()
def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=768
    timer = 0
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            features, seg_map, patches = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)   # 均含有四个层次，img_embed为N*512的编码，seg_map为值为不同索引数字的原图大小的mask
            # show_all_mask(i, img, img_embed, seg_map)
            # img_embed = img_embed
            # plot_patches(patches)
        except:
            raise ValueError(timer)

        # num = img_embed.shape[0]
        # img_embeds[i, :num] = img_embed    # 将embed放入整个大列表中
    
        # seg_maps[i] = seg_map   # N_imgs,H,W
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        save_numpy(save_path, features, seg_map)
        print(f"Saved features&segmap of fig {data_list[i]}")
        # break

    # mask_generator.predictor.model.to('cpu')
        
    # for i in range(img_embeds.shape[0]):
    #     save_path = os.path.join(save_folder, data_list[i].split('.')[0])
    #     assert total_lengths[i] == int(seg_maps[i].max() + 1)
    #     curr = {
    #         'feature': img_embeds[i, :total_lengths[i]],
    #         'seg_maps': seg_maps[i]
    #     }
    #     sava_numpy(save_path, curr)

def save_numpy(save_path, features, seg_map):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, seg_map)
    np.save(save_path_f, features)

def _embed_clip_sam_tiles(image, sam_encoder, mode="default", fdim=768, device="cuda"):
    aug_imgs = torch.cat([image])   # [1,3,h,w]
    seg_images, seg_map = sam_encoder(aug_imgs)

    # clip_embeds = {}
    # for mode in ['default', 's', 'm', 'l']:
    #     tiles = seg_images[mode]
    #     tiles = tiles.to("cuda")
    #     with torch.no_grad():
    #         clip_embed = model.encode_image(tiles)
    #     clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
    #     clip_embeds[mode] = clip_embed.detach().cpu().half()
    tiles = seg_images[mode]
    clip_embeds = np.zeros((tiles.shape[0], fdim))
    # tiles = tiles.to("cuda")
    for i in range(tiles.shape[0]):
        patch_img = Image.fromarray(tiles[i].astype("uint8")) 
        with torch.no_grad():
            clip_embed = model.encode_image(patch_img)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embed = clip_embed.detach().cpu()
        clip_embeds[i] = clip_embed.unsqueeze(0).numpy()
    
    return clip_embeds, seg_map[mode], tiles   # [N, fdim], [h, w], [N, 256, 256, 3]

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

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

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

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
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
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

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (256,256))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        # seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    # if len(masks_s) != 0:
    #     seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    # if len(masks_m) != 0:
    #     seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    # if len(masks_l) != 0:
    #     seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="/home/ubuntu/TJH/Work/aff_ws/LangScene-X/ckpt/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    source_path = args.source_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(source_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()
    model = SIGLIP2NetWork()
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]    # (1,C,H,W）
    imgs = torch.cat(images)    

    save_folder = os.path.join(source_path, 'siglip2_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)