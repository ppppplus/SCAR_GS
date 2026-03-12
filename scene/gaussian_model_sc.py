################################### 1212 objid版本（后续改为objlogits）
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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation_inverse
from utils.dual_quaternion import quaternion_mul

def _to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dytpes=torch.float32, device=device)
    return x

def load_ply(path, f_dim=3, max_sh_degree=0):
    plydata = PlyData.read(path)
    # print(plydata.elements[0].properties)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
    semantic = np.zeros((xyz.shape[0], f_dim))
    for idx, attr_name in enumerate(fea_names):
        semantic[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, opacities, features_dc, features_extra, scales, rots, semantic

class GaussianModelSC:
    # GaussianModel with scale constrained
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._semantic_feature = None

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = nn.functional.normalize
    
    def param_names(self):
        return ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'max_radii2D', 'xyz_gradient_accum']
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._semantic_feature,
            self._obj_ids,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):

        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._semantic_feature,
            self._obj_ids,
            self.max_radii2D,
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
        ) = model_args
        
        # self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    # @classmethod
    # def build_from(cls, gs, **kwargs):
    #     new_gs = GaussianModel(**kwargs)
    #     new_gs._xyz = nn.Parameter(gs._xyz)
    #     new_gs._features_dc = nn.Parameter(torch.zeros_like(gs._features_dc))
    #     new_gs._features_rest = nn.Parameter(torch.zeros_like(gs._features_rest))
    #     new_gs._scaling = nn.Parameter(gs._scaling)
    #     new_gs._rotation = nn.Parameter(gs._rotation)
    #     new_gs._opacity = nn.Parameter(gs._opacity)
    #     new_gs._semantic_feature = nn.Parameter(gs._semantic_feature)
    #     # new_gs.feature = nn.Parameter(gs.feature)
    #     new_gs._obj_ids = gs._obj_ids
    #     new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")
    #     return new_gs

    @property
    def get_scaling(self):
        s = self.scaling_activation(self._scaling)  # (N,1)
        return s.expand(-1, 3)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_features_dc(self):
        return self._features_dc
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_rgb(self):
        return SH2RGB(self._features_dc.squeeze(1))
    
    @property
    def get_semantic_feature(self):
        if self._semantic_feature is not None:
            return self._semantic_feature
        else:
            raise ValueError('no semantic feature')
    
    @property
    def get_obj_ids(self):
        if self._obj_ids is not None:
            return self._obj_ids
        else:
            raise ValueError('no object ids')
        
    @property
    def get_max_obj_id(self):
        if not hasattr(self, "_obj_ids") or self._obj_ids is None:
            raise ValueError('no object ids')

        obj_ids = self._obj_ids
        if isinstance(obj_ids, torch.Tensor):
            if obj_ids.numel() == 0:
                return None
            return int(obj_ids.max().item())
        else:  # numpy
            if obj_ids.size == 0:
                return None
            return int(obj_ids.max())
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    
    def get_covariance(self, d_rot, scaling_modifier=1):
        rotation = quaternion_mul(d_rot, self.get_rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)
    
    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, obj_ids: np.ndarray, spatial_lr_scale: float=5., semantic_feature_size: int=3, print_info=True, max_point_num=150_000):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        if type(obj_ids) == np.ndarray:
            fused_obj_ids = torch.tensor(np.asarray(obj_ids), dtype=torch.int32, device="cuda")
        else:
            fused_obj_ids = obj_ids
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], semantic_feature_size).float().cuda() 

        if print_info:
            print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._scaling = nn.Parameter(scales[:, :1])  # 只保留一个
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic_feature = nn.Parameter(self._semantic_feature.contiguous().requires_grad_(True))
        self._obj_ids = fused_obj_ids

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._semantic_feature.shape[1]):  
            l.append('semantic_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        
        # --- 修改开始 ---
        # 即使内部是 (N,1)，保存时为了兼容性，扩展为 (N,3)
        scale = self._scaling.detach().repeat(1, 3).cpu().numpy()
        # --- 修改结束 ---
        
        rotation = self._rotation.detach().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().cpu().numpy()

        # obj_ids logic ... (保持不变)
        if hasattr(self, "_obj_ids") and self._obj_ids is not None:
            obj_ids = self._obj_ids.detach().cpu().numpy().astype(np.int32).reshape(-1, 1)
        else:
            obj_ids = None

        # --- 修改属性列表构造 ---
        # 不要调用 self.construct_list_of_attributes()，因为那个函数会依据 shape 只生成 scale_0
        # 我们手动指定标准格式
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        
        # 强制写入 scale_0, scale_1, scale_2
        for i in range(3):
            l.append('scale_{}'.format(i))
            
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._semantic_feature.shape[1]):  
            l.append('semantic_{}'.format(i))
        # ----------------------

        dtype_full = [(attribute, 'f4') for attribute in l]

        if obj_ids is not None:
            dtype_full.append(("obj_id", "i4"))

        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature),
            axis=1
        )

        if obj_ids is not None:
            attributes = np.concatenate((attributes, obj_ids), axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)
        vertex = plydata.elements[0].data

        # === xyz ===
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

        # === opacity ===
        opacities = vertex["opacity"][..., np.newaxis]

        # === features_dc (3,1) ===
        features_dc = np.stack([
            vertex["f_dc_0"],
            vertex["f_dc_1"],
            vertex["f_dc_2"]
        ], axis=1)
        features_dc = np.expand_dims(features_dc, axis=-1)  # (P,3,1)

        # === semantic_feature ===
        semantic_names = [n for n in vertex.dtype.names if n.startswith("semantic_")]
        semantic_names = sorted(semantic_names, key=lambda x: int(x.split("_")[1]))
        semantic_feature = np.stack([vertex[n] for n in semantic_names], axis=1)
        # semantic_feature = np.expand_dims(semantic_feature, axis=-1)  # (P,C,1)

        # === f_rest ===
        extra_f_names = [n for n in vertex.dtype.names if n.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[2]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.stack([vertex[n] for n in extra_f_names], axis=1)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        # === scale ===
        # scale_names = sorted([n for n in vertex.dtype.names if n.startswith("scale_")])
        # scales = np.stack([vertex[n] for n in scale_names], axis=1)
        # === scale ===
        scale_names = sorted([n for n in vertex.dtype.names if n.startswith("scale_")])
        scales = np.stack([vertex[n] for n in scale_names], axis=1)
        
        # 如果读取到的是标准的 3 维 scaling，强制转换为 1 维
        if scales.shape[1] == 3:
            # 可以取平均，或者直接取第一维 (通常初始点云生成时三者是相等的)
            scales = scales[:, :1] 

        # === rotation ===
        rot_names = sorted([n for n in vertex.dtype.names if n.startswith("rot")])
        rots = np.stack([vertex[n] for n in rot_names], axis=1)

        # === obj_id ===
        if "obj_id" in vertex.dtype.names:
            obj_ids = vertex["obj_id"].astype(np.int32)  # enforce int32
            # self._obj_ids = obj_ids  # keep as numpy (recommended)
            self._obj_ids = torch.tensor(obj_ids, dtype=torch.int32, device="cuda")
        else:
            print("[load_ply] No obj_id found in PLY.")
            self._obj_ids = None

        # ===== Convert to torch.nn.Parameters =====
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda")
                                        .transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device="cuda")
                                        .transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float32, device="cuda"), requires_grad=True)

        # === initialize runtime buffers ===
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
 
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        if hasattr(self, "_obj_ids"):
            self._obj_ids = self._obj_ids[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_semantic_feature):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "semantic_feature": new_semantic_feature}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 
        
        # print(f"[DENSIFICATION_POSTFIX] Final semantic feature shape: {self._semantic_feature.shape}")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2, selected_pts_mask=None, without_prune=False):
        if selected_pts_mask is None:
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        s = self.get_scaling[selected_pts_mask][:, :1]
        new_scaling = self.scaling_inverse_activation(s.repeat(N,1) / (0.8 * N))

        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1) 
        # new_obj_ids = self.obj_ids[selected_pts_mask].repeat_interleave(N)
        if hasattr(self, "_obj_ids"):
            selected_obj_ids = self._obj_ids[selected_pts_mask]      # shape K
            new_obj_ids = selected_obj_ids.repeat_interleave(N)     # shape K*N
        else:
            new_obj_ids = None
        
        # print(f"[DENSIFY_SPLIT] Original semantic feature shape: {self._semantic_feature.shape}")
        # print(f"[DENSIFY_SPLIT] Selected points: {selected_pts_mask.sum()}")
        # print(f"[DENSIFY_SPLIT] New semantic feature shape: {new_semantic_feature.shape}")
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature)

        if new_obj_ids is not None:
            self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)
        
        if not without_prune:
            nsel = int(selected_pts_mask.sum().item())
            prune_filter = torch.cat((selected_pts_mask,
                                    torch.zeros(N * nsel, device="cuda", dtype=torch.bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 
        
        # print(f"[DENSIFY_CLONE] Original semantic feature shape: {self._semantic_feature.shape}")
        # print(f"[DENSIFY_CLONE] Selected points: {selected_pts_mask.sum()}")
        # print(f"[DENSIFY_CLONE] New semantic feature shape: {new_semantic_feature.shape}")

        if hasattr(self, "_obj_ids"):
            new_obj_ids = self._obj_ids[selected_pts_mask]
        else:
            new_obj_ids = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature)

        if new_obj_ids is not None:
            self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask_opacity = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask_opacity
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask_opacity, big_points_vs), big_points_ws)
        # if prune_mask.sum() > 10000:
        #     N_total, N_prune, N_prune_opacity = self.get_xyz.shape[0], prune_mask.sum(), prune_mask_opacity.sum()
        #     print(f"Pruning {N_prune} points, {N_prune / N_total * 100:.2f}%, {N_prune_opacity} by opacity.")
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1