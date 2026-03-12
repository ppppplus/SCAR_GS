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
import torch.nn.functional as F
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

class GaussianModel:
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
        self._obj_logits = None
        self.num_objects = None  # K

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
        # if self.include_feature:
        #     assert self._semantic_feature is not None, "no semantic feature"
        #     return (
        #         self.active_sh_degree,
        #         self._xyz,
        #         self._features_dc,
        #         self._features_rest,
        #         self._scaling,
        #         self._rotation,
        #         self._opacity,
        #         self._semantic_feature,
        #         self.max_radii2D,
        #         self.xyz_gradient_accum,
        #         self.denom,
        #         self.optimizer.state_dict(),
        #         self.spatial_lr_scale,
        #     )
        # else:
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._semantic_feature,
            # self._obj_ids,
            self._obj_logits,
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
            # self._obj_ids,
            self._obj_logits,
            self.max_radii2D,
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
        ) = model_args
        if self._obj_logits is not None:
            self.num_objects = self._obj_logits.shape[1]
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    #     # new_gs._obj_ids = gs._obj_ids
    #     new_gs._obj_logits = nn.Parameter(gs._obj_logits)
    #     new_gs.num_objects = gs._obj_logits.shape[1]
    #     new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")
    #     return new_gs

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

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
    def get_rgb(self):
        return SH2RGB(self._features_dc.squeeze(1))
    
    @property
    def get_semantic_feature(self):
        if self._semantic_feature is not None:
            return self._semantic_feature
        else:
            raise ValueError('no semantic feature')
    
    # @property
    # def get_obj_ids(self):
    #     if self._obj_ids is not None:
    #         return self._obj_ids
    #     else:
    #         raise ValueError('no object ids')
    @property
    def get_obj_probs(self):
        return torch.softmax(self._obj_logits, dim=-1)

    @property
    def get_obj_id(self):
        return torch.argmax(self._obj_logits, dim=-1)

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
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic_feature = nn.Parameter(self._semantic_feature.contiguous().requires_grad_(True))
        # self._obj_ids = fused_obj_ids
        unique_ids = torch.unique(fused_obj_ids[fused_obj_ids >= 0])
        self.num_objects = int(unique_ids.max().item() + 1)
        print("Number of objs at initialization: ", self.num_objects)
        N = fused_obj_ids.shape[0]
        obj_logits = torch.zeros((N, self.num_objects), device="cuda")

        valid = fused_obj_ids >= 0
        obj_logits[valid, fused_obj_ids[valid]] = 5.0  # strong prior
        obj_logits[~valid] = 0.0  # background / unknown

        self._obj_logits = nn.Parameter(obj_logits, requires_grad=True)

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
            {'params': [self._obj_logits], 'lr': training_args.obj_logits_lr, "name": "obj_logits"},
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
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().cpu().numpy()
        obj_ids = self.get_obj_id.detach().cpu().numpy().astype(np.int32).reshape(-1, 1)
        # obj_ids (must be int32 for PLY)
        # if hasattr(self, "_obj_ids"):
        #     obj_ids = self._obj_ids
        #     if isinstance(obj_ids, torch.Tensor):
        #         obj_ids = obj_ids.detach().cpu().numpy()
        #     obj_ids = obj_ids.astype(np.int32).reshape(-1, 1)
        # else:
        #     obj_ids = None


        # original dtype list
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # add obj_id dtype (int32)
        if obj_ids is not None:
            dtype_full.append(("obj_id", "i4"))

        # base attributes
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature),
            axis=1
        )

        # append obj_ids into attribute matrix
        if obj_ids is not None:
            attributes = np.concatenate((attributes, obj_ids), axis=1)

        # write to ply
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
        scale_names = sorted([n for n in vertex.dtype.names if n.startswith("scale_")])
        scales = np.stack([vertex[n] for n in scale_names], axis=1)

        # === rotation ===
        rot_names = sorted([n for n in vertex.dtype.names if n.startswith("rot")])
        rots = np.stack([vertex[n] for n in rot_names], axis=1)

        # === obj_logit ===
        if "obj_id" in vertex.dtype.names:
            obj_ids_np = vertex["obj_id"].astype(np.int32)
            obj_ids = torch.from_numpy(obj_ids_np).to(device="cuda", dtype=torch.long)

            valid = obj_ids >= 0
            if valid.any():
                self.num_objects = int(obj_ids[valid].max().item() + 1)
            else:
                self.num_objects = 1  # 或者直接不训练 obj_logits
            N = obj_ids.shape[0]

            obj_logits = torch.zeros((N, self.num_objects), device="cuda")
            if valid.any():
                obj_logits[valid, obj_ids[valid]] = 5.0

            self._obj_logits = nn.Parameter(obj_logits, requires_grad=True)
        else:
            self._obj_logits = None
            self.num_objects = None

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
        self._obj_logits = optimizable_tensors["obj_logits"]

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
                              new_rotation, new_semantic_feature, new_obj_logits):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "semantic_feature": new_semantic_feature,
             "obj_logits": new_obj_logits}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 
        self._obj_logits = optimizable_tensors["obj_logits"]
        
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
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)
        # new_obj_ids = self.obj_ids[selected_pts_mask].repeat_interleave(N)
        # if hasattr(self, "_obj_ids"):
        #     selected_obj_ids = self._obj_ids[selected_pts_mask]      # shape K
        #     new_obj_ids = selected_obj_ids.repeat_interleave(N)     # shape K*N
        # else:
        #     new_obj_ids = None
        new_obj_logits = self._obj_logits[selected_pts_mask].repeat(N, 1)
        
        # print(f"[DENSIFY_SPLIT] Original semantic feature shape: {self._semantic_feature.shape}")
        # print(f"[DENSIFY_SPLIT] Selected points: {selected_pts_mask.sum()}")
        # print(f"[DENSIFY_SPLIT] New semantic feature shape: {new_semantic_feature.shape}")

        # new_feature = self.feature[selected_pts_mask].repeat(N, 1) if self.fea_dim > 0 else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature, new_obj_logits)

        # if new_obj_ids is not None:
        #     self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)
        
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

        # new_feature = self.feature[selected_pts_mask] if self.fea_dim > 0  else None
        
        # print(f"[DENSIFY_CLONE] Original semantic feature shape: {self._semantic_feature.shape}")
        # print(f"[DENSIFY_CLONE] Selected points: {selected_pts_mask.sum()}")
        # print(f"[DENSIFY_CLONE] New semantic feature shape: {new_semantic_feature.shape}")

        # if hasattr(self, "_obj_ids"):
        #     new_obj_ids = self._obj_ids[selected_pts_mask]
        # else:
        #     new_obj_ids = None
        new_obj_logits = self._obj_logits[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature, new_obj_logits)

        # if new_obj_ids is not None:
        #     self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)

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

    # def semantic_merge(
    #     self,
    #     groups=None,                  # Optional[list[LongTensor]]: each tensor is indices to merge
    #     same_obj_only=True,           # if True, only cluster within same obj_id
    #     cos_thr=0.95,                 # cosine similarity threshold for auto grouping
    #     min_cluster_size=2,           # only merge clusters with >=2 points
    #     max_cluster_size=None,        # optional cap for greedy grouping
    #     weight_by="opacity",          # "opacity" or "uniform"
    #     keep_obj_id="rep",            # "rep" | "min" | "majority"
    #     verbose=False,
    # ):
    #     """
    #     Merge Gaussians based on semantic similarity, while keeping optimizer state.

    #     Key idea:
    #     1) choose one representative index per group (rep)
    #     2) compute merged parameters (weighted average) and write into rep (in-place)
    #     3) prune all other indices via prune_points(mask) so optimizer/buffers/obj_ids stay aligned

    #     Returns:
    #         stats dict
    #     """
    #     device = self._xyz.device
    #     N = self._xyz.shape[0]

    #     if N == 0:
    #         return {"merged_groups": 0, "merged_points": 0}

    #     if self._semantic_feature is None:
    #         raise ValueError("[semantic_merge] self._semantic_feature is None")

    #     if not hasattr(self, "_obj_ids") or self._obj_ids is None:
    #         # 允许无 obj_id：same_obj_only 会失效
    #         same_obj_only = False

    #     # ------------------------------------------------------------
    #     # Helper: build groups automatically (greedy) if groups is None
    #     # ------------------------------------------------------------
    #     if groups is None:
    #         sem = self.get_semantic_feature.detach()
    #         sem = F.normalize(sem, dim=1, eps=1e-8)  # (N,C)

    #         if same_obj_only:
    #             obj_ids = self._obj_ids.detach()
    #             unique_oids = torch.unique(obj_ids)
    #         else:
    #             unique_oids = torch.tensor([0], device=device)

    #         assigned = torch.zeros((N,), dtype=torch.bool, device=device)
    #         groups = []

    #         for oid in unique_oids:
    #             if same_obj_only:
    #                 idxs = torch.where(self._obj_ids == oid)[0]
    #             else:
    #                 idxs = torch.arange(N, device=device)

    #             # 只在子集里做 greedy
    #             sub_assigned = assigned[idxs]
    #             sub_idxs = idxs[~sub_assigned]
    #             if sub_idxs.numel() == 0:
    #                 continue

    #             # greedy: pick a seed, group all with cos >= thr
    #             while sub_idxs.numel() > 0:
    #                 seed = sub_idxs[0]
    #                 if assigned[seed]:
    #                     sub_idxs = sub_idxs[1:]
    #                     continue

    #                 sims = (sem[sub_idxs] @ sem[seed])  # (M,)
    #                 g = sub_idxs[sims >= cos_thr]

    #                 if max_cluster_size is not None and g.numel() > max_cluster_size:
    #                     g = g[:max_cluster_size]

    #                 if g.numel() >= min_cluster_size:
    #                     groups.append(g)
    #                     assigned[g] = True

    #                 # remove assigned from sub_idxs
    #                 sub_idxs = sub_idxs[~assigned[sub_idxs]]

    #     # sanitize groups
    #     cleaned_groups = []
    #     for g in groups:
    #         if g is None:
    #             continue
    #         if not torch.is_tensor(g):
    #             g = torch.tensor(g, device=device, dtype=torch.long)
    #         g = g.unique()
    #         g = g[(g >= 0) & (g < N)]
    #         if g.numel() >= min_cluster_size:
    #             cleaned_groups.append(g)
    #     groups = cleaned_groups

    #     if len(groups) == 0:
    #         return {"merged_groups": 0, "merged_points": 0}

    #     # ------------------------------------------------------------
    #     # Helper: choose representative + compute weights
    #     # ------------------------------------------------------------
    #     def _weights_for(g_idx: torch.Tensor):
    #         if weight_by == "opacity":
    #             w = self.get_opacity.detach().squeeze(-1)[g_idx]  # (K,)
    #             w = torch.clamp(w, min=1e-6)
    #         else:
    #             w = torch.ones((g_idx.numel(),), device=device)
    #         w = w / (w.sum() + 1e-9)
    #         return w  # (K,)

    #     def _choose_rep(g_idx: torch.Tensor):
    #         # pick the max opacity as representative (stable in practice)
    #         opa = self.get_opacity.detach().squeeze(-1)[g_idx]
    #         rep = g_idx[torch.argmax(opa)]
    #         return rep

    #     # ------------------------------------------------------------
    #     # Prepare prune mask (True means "remove this point")
    #     # ------------------------------------------------------------
    #     prune_mask = torch.zeros((N,), dtype=torch.bool, device=device)

    #     merged_groups = 0
    #     merged_points = 0

    #     # ------------------------------------------------------------
    #     # Apply each merge group: in-place write into rep, prune others
    #     # ------------------------------------------------------------
    #     with torch.no_grad():
    #         for g in groups:
    #             if g.numel() < min_cluster_size:
    #                 continue

    #             rep = _choose_rep(g)
    #             others = g[g != rep]
    #             if others.numel() == 0:
    #                 continue

    #             # Optional: enforce same obj_id inside group if requested
    #             if same_obj_only and self._obj_ids is not None:
    #                 oid0 = int(self._obj_ids[rep].item())
    #                 if not torch.all(self._obj_ids[g] == oid0):
    #                     # 这个 group 跨 obj_id，跳过（骨架策略）
    #                     continue

    #             w = _weights_for(g)  # (K,)

    #             # ---- merged params ----
    #             # xyz: weighted average
    #             xyz_m = (self._xyz[g] * w[:, None]).sum(dim=0)  # (3,)

    #             # features
    #             fdc_m = (self._features_dc[g] * w[:, None, None]).sum(dim=0)      # (1,3,1)?? actually _features_dc shape is (N,1,3?) in your code it's (N,1,3?)? you use transpose(1,2) earlier.
    #             frest_m = (self._features_rest[g] * w[:, None, None]).sum(dim=0)

    #             # scaling: average in log-space (since scaling_activation=exp)
    #             sca_m = (self._scaling[g] * w[:, None]).sum(dim=0)

    #             # rotation: weighted average then normalize (quaternion-like)
    #             rot_m = (self._rotation[g] * w[:, None]).sum(dim=0)
    #             rot_m = rot_m / (rot_m.norm() + 1e-9)

    #             # opacity: average in logit space is better, but skeleton uses average in raw param space
    #             opa_m = (self._opacity[g] * w[:, None]).sum(dim=0)

    #             # semantic_feature
    #             sem_m = (self._semantic_feature[g] * w[:, None]).sum(dim=0)

    #             # optional extra feature
    #             if self.fea_dim > 0:
    #                 feat_m = (self.feature[g] * w[:, None]).sum(dim=0)

    #             # ---- write into representative (in-place) ----
    #             self._xyz.data[rep] = xyz_m
    #             self._features_dc.data[rep] = fdc_m
    #             self._features_rest.data[rep] = frest_m
    #             self._scaling.data[rep] = sca_m
    #             self._rotation.data[rep] = rot_m
    #             self._opacity.data[rep] = opa_m
    #             self._semantic_feature.data[rep] = sem_m
    #             if self.fea_dim > 0:
    #                 self.feature.data[rep] = feat_m

    #             # ---- decide obj_id for rep (optional policy) ----
    #             if hasattr(self, "_obj_ids") and self._obj_ids is not None:
    #                 if keep_obj_id == "min":
    #                     self._obj_ids[rep] = torch.min(self._obj_ids[g]).to(self._obj_ids.dtype)
    #                 elif keep_obj_id == "majority":
    #                     # majority vote
    #                     vals, cnt = torch.unique(self._obj_ids[g], return_counts=True)
    #                     self._obj_ids[rep] = vals[torch.argmax(cnt)].to(self._obj_ids.dtype)
    #                 else:
    #                     # "rep": keep original
    #                     pass

    #             # ---- mark others for prune ----
    #             prune_mask[others] = True

    #             merged_groups += 1
    #             merged_points += int(others.numel())

    #     # ------------------------------------------------------------
    #     # Prune all "others" in one shot (keeps optimizer state aligned)
    #     # ------------------------------------------------------------
    #     if prune_mask.any():
    #         # prune_points expects mask=True means prune
    #         self.prune_points(prune_mask)

    #     if verbose:
    #         print(f"[semantic_merge] merged_groups={merged_groups}, merged_points={merged_points}, new_N={self._xyz.shape[0]}")

    #     return {
    #         "merged_groups": merged_groups,
    #         "merged_points": merged_points,
    #         "new_num_points": int(self._xyz.shape[0]),
    #     }


class StandardGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(sh_degree, fea_dim, with_motion_mask)
        self.all_the_same = all_the_same
    
    @property
    def get_scaling(self):
        scaling = self._scaling.mean()[None, None].expand_as(self._scaling) if self.all_the_same else self._scaling.mean(dim=1, keepdim=True).expand_as(self._scaling)
        return self.scaling_activation(scaling)
