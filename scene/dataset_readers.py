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

import os
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.geo_utils import depth_to_world_points
from utils.camera_utils import estimate_intrinsics_from_angle
from utils.graphics_utils import BasicPointCloud, ObjPointCloud
import torch
import torch.nn.functional as F
import sys
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    c2w: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    id_mask: np.array
    semantic_feature: torch.tensor 
    target_mask: torch.tensor
    # semantic_feature_path: str 
    # semantic_feature_name: str 
    image_path: str
    image_name: str
    width: int
    height: int
    depth: Optional[np.array] = None
    


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    obj_ids: np.array
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    obj_ids_path: str
    semantic_feature_dim: int 
    # train_cameras_2s: list
    # test_cameras_2s: list


def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}
    
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, semantic_feature_folder, load_depth=True):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # w2c = np.eye(4) 
        # w2c[:3, :3] = R.T
        # w2c[:3, 3] = T
        # c2w = np.linalg.inv(w2c)
        # c2w[:3, 1:3] *= -1 
        # c2w[:3, 1:3] *= -1
        # w2c = np.linalg.inv(c2w)
        # R = w2c[:3, :3].T
        # T = w2c[:3, 3]
        
        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path).convert("RGB")
        oriimage = Image.open(image_path)

        # try:
        #     im_data = np.array(oriimage)
        # except:
        #     print(f'{image_path} is damaged')
        #     continue
        # ho = im_data.shape[0]
        # wo = im_data.shape[1]
        
        semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap.pt' 
        semantic_feature = torch.load(semantic_feature_path, weights_only=True)
        # semantic_feature = semantic_feature.unsqueeze(0)  # (1, 3, h, w)
        # semantic_feature = F.interpolate(semantic_feature, size=(ho, wo), mode='bilinear', align_corners=False)

        # # 去掉 batch 维度
        # semantic_feature = semantic_feature.squeeze(0)  #
        # semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
        # semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap.pt' 
        # semantic_feature = torch.load(semantic_feature_path, weights_only=True) 
        # semantic_feature = semantic_feature.squeeze(0)  #

        depth_path = modify_image_path(image_path, "depth")
    
        if load_depth and os.path.exists(depth_path):
            depth = cv.imread(depth_path, -1) / 1e3
            h, w = depth.shape
            # if depth.size == mask.size:
            #     depth[mask[..., 0] < 0.5] = 0
            # else:
            # depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            depth[depth < 0.1] = 0
        else:
            depth = None
        
        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       semantic_feature=semantic_feature,
        #                     #   semantic_feature_path=semantic_feature_path,
        #                     #   semantic_feature_name=semantic_feature_name,
        #                       target_mask=None) 
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=oriimage, depth=depth, mono_depth=None, semantic_feature=semantic_feature,
                                        image_path=image_path, image_name=image_name, width=width, height=height, fid=0))
        # cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    if all(attr in vertices.data.dtype.names for attr in ['nx', 'ny', 'nz']):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # print(f"Loaded pcd with {positions.shape[0]} points.")
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchObjPly(path):
    ply_path = os.path.join(path, "points3d.ply")
    obj_path = os.path.join(path, "obj_ids.npy")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing file: {ply_path}")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Missing file: {obj_path}")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    if all(attr in vertices.data.dtype.names for attr in ['nx', 'ny', 'nz']):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    # -------- load obj ids --------
    obj_ids = np.load(obj_path)

    if len(obj_ids) != len(positions):
        raise ValueError(
            f"obj_ids length ({len(obj_ids)}) does not match "
            f"point count ({len(positions)})"
        )

    return ObjPointCloud(
        points=positions,
        colors=colors,
        normals=normals,
        obj_ids=obj_ids,
    )

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def load_ply2dict(path, max_sh_degree=0, device="cuda", apply_activation=True):
    plydata = PlyData.read(path)
    # xyz
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # f_dc
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # semantic
    count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
    semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1)
    semantic_feature = np.expand_dims(semantic_feature, axis=-1)  # [N,3,1]

    # f_rest
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    # scale
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # rotation
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # ---- 转 tensor ----
    xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
    opacity = torch.tensor(opacities, dtype=torch.float32, device=device)
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    rotations = torch.tensor(rots, dtype=torch.float32, device=device)

    # ---- 可选激活 ----
    if apply_activation:
        opacity = torch.sigmoid(opacity)            # [N,1]
        scales = torch.exp(scales)                  # [N,3]
        rotations = F.normalize(rotations, dim=-1)  # [N,4]

    features_dc = torch.tensor(features_dc, dtype=torch.float32, device=device).transpose(1, 2).contiguous()
    features_extra = torch.tensor(features_extra, dtype=torch.float32, device=device).transpose(1, 2).contiguous()
    semantic_feature = torch.tensor(semantic_feature, dtype=torch.float32, device=device)

    pc_dict = {
        "xyz": xyz,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "features": torch.cat([features_dc, features_extra], dim=1),  # [N, SH, 3]
        "semantic_feature": semantic_feature,  # [N,3,1]
        "active_sh_degree": int(np.sqrt(features_extra.shape[1] + 1) - 1)
    }

    return pc_dict

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images

    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "lseg_embeddings" 
    elif foundation_model =='detic':
        semantic_feature_dir = "detic_embeddings" 
    # temp_path = path.replace(ft_name, '')
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), semantic_feature_folder=os.path.join(path, semantic_feature_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    semantic_feature_dim = cam_infos[0].semantic_feature.shape[0]

    if eval:
        train_cam_infos = cam_infos
        try:
            test_cameras_extrinsic_file = os.path.join(path, "test/sparse/0", "images.bin")
            test_cameras_intrinsic_file = os.path.join(path, "test/sparse/0", "cameras.bin")
            test_cam_extrinsics = read_extrinsics_binary(test_cameras_extrinsic_file)
            test_cam_intrinsics = read_extrinsics_binary(test_cameras_intrinsic_file)
        except Exception as e:
            print(f"Exception occurred: {e}")
            cmd = f"colmap model_converter --input_path {os.path.join(path,'test/sparse/0')} --output_path {os.path.join(path,'test/sparse/0')} --output_type TXT"
            os.system(cmd)
            test_cameras_extrinsic_file = os.path.join(path, "test/sparse/0", "images.txt")
            test_cameras_intrinsic_file = os.path.join(path, "test/sparse/0", "cameras.txt")
            test_cam_extrinsics = read_extrinsics_text(test_cameras_extrinsic_file)
            test_cam_intrinsics = read_intrinsics_text(test_cameras_intrinsic_file)
        test_reading_dir = "test/images"
        # reading_dir_F = "language_feature" if language_feature == None else language_feature
        test_cam_infos_unsorted = readColmapCameras(cam_extrinsics=test_cam_extrinsics, cam_intrinsics=test_cam_intrinsics, images_folder=os.path.join(path, test_reading_dir), semantic_feature_folder=os.path.join(path, semantic_feature_dir))
        test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

# def mergeplyfromdepth(train_cam_infos, downsample=True, voxel_size=0.005):
#     all_points, all_colors = [], []
#     for cam in train_cam_infos:
#         color = np.array(cam.image)
#         depth = np.array(cam.depth)
#         c2w = cam.c2w
#         K = estimate_intrinsics_from_angle(cam.FovX, cam.FovY, cam.width, cam.height)
#         # K = np.array([[cam.FovX/2, 0, cam.width/2], [0, cam.FovY/2, cam.height/2], [0, 0, 1]])
#         pts, u, v = depth_to_world_points(depth, K, c2w, stride=2)
#         colors = color[v, u]
#         all_points.append(pts)
#         all_colors.append(colors)
#     xyz = np.concatenate(all_points, axis=0)
#     rgb = np.concatenate(all_colors, axis=0)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.colors = o3d.utility.Vector3dVector(rgb)
#     if downsample:
#         pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#     return np.array(pcd.points), np.array(pcd.colors)
def mergeplyfromdepth(train_cam_infos, downsample=True, voxel_size=0.005):
    all_points, all_colors = [], []
    all_obj_ids = []

    # ===== 是否使用 id_mask =====
    use_obj_id = (
        hasattr(train_cam_infos[0], "id_mask") 
        and train_cam_infos[0].id_mask is not None
    )

    for cam in train_cam_infos:
        color = np.array(cam.image)       # (H, W, 3)
        depth = np.array(cam.depth)       # (H, W)

        c2w = cam.c2w
        K = estimate_intrinsics_from_angle(cam.FovX, cam.FovY, cam.width, cam.height)

        pts, u, v = depth_to_world_points(depth, K, c2w, stride=4)
        colors = color[v, u]

        all_points.append(pts)
        all_colors.append(colors)

        if use_obj_id:
            id_mask = cam.id_mask   # (H, W)
            obj_ids = id_mask[v, u]
            all_obj_ids.append(obj_ids)

    # ===== 合并 xyz / rgb =====
    xyz = np.concatenate(all_points, axis=0)
    rgb = np.concatenate(all_colors, axis=0)

    # ===== 不使用 obj_id =====
    if not use_obj_id:
        if downsample:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            return np.asarray(pcd.points), np.asarray(pcd.colors), None
        else:
            return xyz, rgb, None

    # ===== 合并 obj_id =====
    obj_ids = np.concatenate(all_obj_ids, axis=0)

    # ===== Downsample 同步 obj_id =====
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        ds_xyz = np.asarray(pcd.points)
        ds_rgb = np.asarray(pcd.colors)

        # 最近邻匹配 obj_id
        nbrs = NearestNeighbors(n_neighbors=1).fit(xyz)
        _, idx = nbrs.kneighbors(ds_xyz)
        ds_obj_ids = obj_ids[idx[:, 0]]

        return ds_xyz, ds_rgb, ds_obj_ids

    return xyz, rgb, obj_ids

def modify_image_path(image_path, name):
    parts = image_path.split('/')
    # Replace the second last element with "depth"
    parts[-2] = name
    return '/'.join(parts)

def readCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_folder, id_masks_folder, extension=".png", no_bg=False, load_depth=True, load_mono_depth=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = contents["camera_angle_y"]

        frames = contents["frames"]
        # frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        frames = sorted(frames, key=lambda x: x['file_path'])
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"]
            # if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'images')):
            #     cam_name = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'images', os.path.basename(cam_name)).replace('.jpg', '.png')
            # if cam_name.endswith('jpg') or cam_name.endswith('png'):
            #     cam_name = os.path.join(path, cam_name)
            # else:
            #     cam_name = os.path.join(path, cam_name + extension)
            # frame_time = frame['time']

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            oriimage = Image.open(image_path)

            # try:
            #     im_data = np.array(oriimage)
            # except:
            #     print(f'{image_path} is damaged')
            #     continue

            FovY = fovy
            FovX = fovx
            
            # ho = im_data.shape[0]
            # wo = im_data.shape[1]
            
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '.npy' 
            semantic_feature = np.load(semantic_feature_path)
            semantic_feature = torch.from_numpy(semantic_feature).float()   # [fdim, h ,w]

            id_mask_path = os.path.join(id_masks_folder, image_name) + '.npy'
            if os.path.exists(id_mask_path):
                id_mask_np = np.load(id_mask_path)
                id_mask = torch.from_numpy(id_mask_np) 
                target_mask = (id_mask != -1) 
            else:
                id_mask = None
                target_mask = None
            # target_mask = semantic_feature
            # target_mask = (semantic_feature != 0).any(dim=0)
            
    
            depth_path = modify_image_path(image_path, "depth")
        
            if load_depth and os.path.exists(depth_path):
                # depth = cv.imread(depth_path, -1) / 1e3
                depth_mm = cv.imread(depth_path, cv.IMREAD_UNCHANGED).astype(np.uint16)
                depth = depth_mm.astype(np.float32) / 1000.0
                h, w = depth.shape
                # if depth.size == mask.size:
                #     depth[mask[..., 0] < 0.5] = 0
                # else:
                # depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, c2w=c2w, FovY=FovY, FovX=FovX, image=oriimage, depth=depth, semantic_feature=semantic_feature, id_mask=id_mask,
                                        target_mask=target_mask, image_path=image_path, image_name=image_name, width=oriimage.size[0], height=oriimage.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, ftype, fdim,  white_background, eval, extension=".png", no_bg=True):
    if ftype =='siglip2':
        semantic_feature_dir = f"siglip2_feat_dim{fdim}" 
    else:
        raise ValueError("Invalid foundation model")
    print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []
    train_cam_infos = readCamerasFromTransforms(
        path, f"transforms_train.json", white_background, os.path.join(path, semantic_feature_dir), os.path.join(path, "id_masks"), extension, no_bg=no_bg)
    try:
        test_cam_infos = readCamerasFromTransforms(
            path, f"transforms_test.json", white_background, os.path.join(path, semantic_feature_dir), os.path.join(path, "id_masks"), extension, no_bg=no_bg)
    except:
        test_cam_infos = []
    # if not eval:
    #     train_infos.extend(test_infos)
    # train_cam_infos.append(train_infos)
    # test_cam_infos.append(test_infos)
    
    print(f"Read train transforms with {len(train_cam_infos)} cameras")
    print(f"Read test transforms with {len(test_cam_infos)} cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    obj_ids_path = os.path.join(path, "obj_ids.npy")
    if not os.path.exists(ply_path):
        if os.path.exists(os.path.join(path, "depth")):
            ## get pcd from depth images
            xyz, rgb, obj_ids = mergeplyfromdepth(train_cam_infos, downsample=True, voxel_size=0.03)

            if obj_ids is not None:
                print("id_mask detected, obj_ids generated.")
                np.save(obj_ids_path, obj_ids)

            print(f"Generated pcd from depth images with {xyz.shape[0]} points")
            storePly(ply_path, xyz, rgb)
        else:
            ## generate random pcd
            num_pts = 1500
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # pcd = fetchPly(ply_path)
    try:
        pcd = fetchPly(ply_path)
        obj_ids = np.load(obj_ids_path).astype(np.int32)
    except:
        pcd = None
        obj_ids = None
    scene_info = SceneInfo(point_cloud=pcd,
                           obj_ids = obj_ids,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           obj_ids_path=obj_ids_path,
                           semantic_feature_dim=fdim)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
