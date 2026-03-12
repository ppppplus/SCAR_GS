#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
import cv2
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    gt_semantic_feature = cam_info.semantic_feature
    id_mask = cam_info.id_mask
    target_mask = cam_info.target_mask
    # image size will the same as feature map size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    elif args.resolution == 0:
        assert gt_semantic_feature is not None, \
            "args.resolution=0 requires semantic_feature"
        _, H, W = gt_semantic_feature.shape
        resolution = (W, H)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if gt_semantic_feature is not None and gt_semantic_feature.shape[2]!= resolution[0]:
        gt_semantic_feature = torch.nn.functional.interpolate(
            gt_semantic_feature.unsqueeze(0),
            size=resolution[::-1],  # (H, W)
            mode="nearest"  # 或 bilinear，取决于语义类型
        ).squeeze(0)
        id_mask = cv2.resize(id_mask, resolution, interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(target_mask, resolution, interpolation=cv2.INTER_NEAREST)
    depth = cam_info.depth
    if depth is not None:
        depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)
        depth = np.expand_dims(depth, axis=0)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device, depth=depth, 
                  semantic_feature=gt_semantic_feature, 
                  id_mask=id_mask,
                  target_mask=target_mask)
    return cam


def load_correspondence(args, cam, corr_dir):
    if corr_dir is None:
        return
    H = cam.image_height * args.resolution if args.resolution != -1 else cam.image_height
    corr_list = np.load(corr_dir, allow_pickle=True)['data']
    def rev_pixel(pixel, H):
        return pixel
        # return pixel * np.array([1, -1]).reshape(1, 2) + np.array([0, H - 1]).reshape(1, 2)

    for corr in corr_list:
        src_name, tgt_name = list(corr.keys())
        src_pixel, tgt_pixel = corr[src_name], corr[tgt_name]  # smaller coords are at the top - the same index to use for images
        src_pixel = torch.from_numpy(rev_pixel(src_pixel, H)).long()
        tgt_pixel = torch.from_numpy(rev_pixel(tgt_pixel, H)).long()
        tgt_id = tgt_name.split('_')[1]
        cam.corr[tgt_id] = [src_pixel, tgt_pixel]


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )

def estimate_intrinsics_from_angle(angle_x, angle_y, width, height):
    fx = 0.5 * width / np.tan(angle_x / 2)
    # fy = fx  # assume square pixels
    fy = 0.5 * height / np.tan(angle_y / 2)
    cx = width / 2
    cy = height / 2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)