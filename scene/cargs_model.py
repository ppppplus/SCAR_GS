import torch
import os
import numpy as np
from torch import nn
from plyfile import PlyData
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p
from utils.general_utils import inverse_sigmoid
from scene.gaussian_model import GaussianModel

class CARGaussianModel(GaussianModel):
    """
    Extend the original GaussianModel to support mobility&articulation.
    add change_id, mobility
    """
    def __init__(self, sh_degree, n_screws=10):
        super().__init__(sh_degree)
        # raw mobility (before sigmoid), shape (N,1)
        self._mobility = torch.empty(0)

        # activation
        self.mobility_activation = torch.sigmoid
        self.inverse_mobility_activation = inverse_sigmoid

        self.change_ids = None
        self._mob_mask = None
        self._internal_mask = None
        # self.initialized_articulation = False
        self.grad_hooks = [] 

    def init_mobility_from_change_ids(self, screw_init):
        """
        Initialize mobility based on obj_id and change_ids.

        screw_init:
        {
            "change_ids": [29, 31, 45],
            "screws": {
                "29": { "axis_point": [...], "axis_dir": [...], "top_ids": [...] },
                ...
            }
        }

        self.get_obj_ids: (N,) LongTensor
        """

        device = self.get_xyz.device
        obj_ids = self.get_obj_ids   # (N,)

        # --------------------------------------------------
        # 1. parse change_ids
        # --------------------------------------------------
        change_ids = screw_init["ids"]
        if not torch.is_tensor(change_ids):
            change_ids = torch.tensor(change_ids, device=device)

        self.change_ids = change_ids
        # --------------------------------------------------
        # 2. build internal mask: which gaussians are learnable
        # --------------------------------------------------
        mob_mask = torch.zeros_like(obj_ids, dtype=torch.bool)

        for cid in change_ids:
            mob_mask |= (obj_ids == cid)

        self._mob_mask = mob_mask   # (N,)

        # --------------------------------------------------
        # 3. initialize mobility in [0,1]
        # --------------------------------------------------
        # default: static = 0
        N = obj_ids.shape[0]
        mobility_init = torch.zeros((N, 1), device=device, dtype=torch.float32)
        mobility_init[mob_mask, 0] = 0.5

        mobility_raw = self.inverse_mobility_activation(mobility_init.clamp(1e-4, 1 - 1e-4))
        self._mobility = nn.Parameter(mobility_raw)  # (N,1)


        print(
            f"[Mobility Init] total={len(obj_ids)}, "
            f"trainable={mob_mask.sum().item()}, "
            f"static={(~mob_mask).sum().item()}"
        )
        
    # =======  Getter functions  =======
    @property
    def get_change_ids(self):
        return self.change_ids

    @property
    def get_mob_mask(self):
        if self._mob_mask is None:
            mob_mask = torch.zeros_like(self.get_obj_ids, dtype=torch.bool)
            for cid in self.change_ids:
                mob_mask |= (self.get_obj_ids == cid)
            self._mob_mask = mob_mask 
    
        return self._mob_mask
    
    @property
    def get_mobility(self):
        """sigmoid(mobility_raw) in [0,1]"""
        return self.mobility_activation(self._mobility)

    def param_names(self):
        return super().param_names() + ['_mobility']
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        # add mobility optimizer group
        self.optimizer.add_param_group(
            {
                "params": [self._mobility],
                "lr": training_args.mobility_lr,
                "name": "mobility",
            }
        )
        # Ensure optimizer state is initialized for the newly added mobility parameter
        # if self.optimizer.state.get(self._mobility, None) is None:
        #     # Initialize state if not already initialized
        #     self.optimizer.state[self._mobility] = {
        #         "exp_avg": torch.zeros_like(self._mobility, device=self._mobility.device),
        #         "exp_avg_sq": torch.zeros_like(self._mobility, device=self._mobility.device),
        #     }

    def prune_points(self, mask):
        valid_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._mobility = optimizable_tensors["mobility"]

        if hasattr(self, "_obj_ids"):
            self._obj_ids = self._obj_ids[valid_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]
    
    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_semantic_feature,
        new_mobility,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "semantic_feature": new_semantic_feature,
            "mobility": new_mobility,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._mobility = optimizable_tensors["mobility"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def capture(self):
        """
        Returns a tuple of model parameters for checkpointing.
        Order is important and must match restore().
        """
        base = super().capture()
        return base + (
            self._mobility,     # Tensor / Parameter
            self.change_ids,    # list[int] or Tensor (metadata)
        )

    # def restore(self, model_args, training_args):
    #     *base_args, mobility = model_args

    #     super().restore(tuple(base_args), training_args)
    #     self._mobility = nn.Parameter(mobility.requires_grad_(True))
    def restore(self, model_args, training_args):
        """
        Restore model from checkpoint.
        """

        *base_args, mobility, change_ids = model_args

        # restore base Gaussian parameters
        super().restore(tuple(base_args), training_args)

        # restore mobility (trainable)
        self._mobility = nn.Parameter(mobility.requires_grad_(True))

        # restore change_ids (NOT trainable)
        self.change_ids = change_ids

    def extract_gaussians(
        self,
        mask_or_idx=None,
        detach=False,
        device="cuda",
    ):
        """
        Extract a subset of Gaussians and return all fields needed
        for densification_postfix.

        Args:
            mask_or_idx: BoolTensor [N] or LongTensor [K], or None to extract all
            detach: whether to detach tensors
            device: target device (None = keep original)

        Returns:
            dict with keys:
                xyz
                f_dc
                f_rest
                opacity
                scaling
                rotation
                semantic_feature
                mobility   (RAW mobility, not sigmoid!)obj_ids    (optional)
        """
        # --------------------------------------------------
        # resolve indices
        # --------------------------------------------------
        if mask_or_idx is None:
            # 如果 mask_or_idx 为 None，提取所有高斯核
            idx = torch.arange(self.get_xyz.shape[0], device=self.get_xyz.device)
        elif mask_or_idx.dtype == torch.bool:
            idx = mask_or_idx.nonzero(as_tuple=False).squeeze(-1)
        else:
            idx = mask_or_idx

        if idx.numel() == 0:
            raise ValueError("extract_gaussians: empty selection")

        def _proc(x):
            if x is None:
                return None
            if detach:
                x = x.detach()
            if device is not None:
                x = x.to(device)
            return x

        # --------------------------------------------------
        # extract fields
        # --------------------------------------------------
        data = {
            "xyz": _proc(self.get_xyz[idx]),
            "f_dc": _proc(self.get_features_dc[idx]),
            "f_rest": _proc(self.get_features_rest[idx]),
            "opacity": _proc(self._opacity[idx]),
            "scaling": _proc(self._scaling[idx]),
            "rotation": _proc(self._rotation[idx]),
            "semantic_feature": (
                _proc(self.get_semantic_feature[idx])
                if self.get_semantic_feature is not None
                else None
            ),
            # IMPORTANT: raw mobility, not sigmoid
            "mobility": _proc(self._mobility[idx]),
        }

        if hasattr(self, "_obj_ids"):
            data["obj_ids"] = _proc(self._obj_ids[idx])
        else:
            data["obj_ids"] = None

        return data

    def add_gaussians_from_data(
        self,
        data,
        invalidate_mask=True,
    ):
        """
        Add Gaussians from extracted data dict and correctly merge mobility mask.

        data must contain:
            xyz
            f_dc
            f_rest
            opacity
            scaling
            rotation
            semantic_feature (optional)
            mobility   (RAW mobility, not sigmoid)
            obj_ids    (optional)

        This method:
            - uses densification_postfix
            - syncs obj_ids
            - invalidates mobility/internal masks
        """
        # --- 新增逻辑：强制初始化新点的 Mobility 为 0.5 ---
        # 0.5 是激活后的值，我们需要将其转换为 raw 空间 (通过 inverse_sigmoid)
        # 注意：data["mobility"] 传入的通常是 Raw 值
        # n_new = data["xyz"].shape[0]
        # init_mob_val = 0.5
        
        # # 使用 inverse_sigmoid 将 0.5 转为 raw 值 (约为 0.0)
        # # 这里使用 clamp 1e-4 防止极端值，虽然 0.5 很安全
        # raw_init_mob = inverse_sigmoid(torch.full((n_new,), init_mob_val, device=data["xyz"].device))
        
        # # 覆盖 data 中的 mobility
        # data["mobility"] = raw_init_mob
        # -------------------------------
        # 1. add gaussian parameters
        # -------------------------------

        self.densification_postfix(
            data["xyz"],
            data["f_dc"],
            data["f_rest"],
            data["opacity"],
            data["scaling"],
            data["rotation"],
            data.get("semantic_feature", None),
            data["mobility"],
        )

        # -------------------------------
        # 2. sync obj_ids
        # -------------------------------
        if data.get("obj_ids", None) is not None:
            if not hasattr(self, "_obj_ids"):
                raise RuntimeError("add_gaussians_from_data: model has no _obj_ids")

            self._obj_ids = torch.cat(
                [self._obj_ids, data["obj_ids"]],
                dim=0
            )

        # -------------------------------
        # 3. invalidate cached masks
        # -------------------------------
        if invalidate_mask:
            # mobility mask must be recomputed
            self._mob_mask = None

        
        # 4. update internal mask (marking internal points)
        new_internal_mask = torch.ones(data["xyz"].shape[0], device=self._internal_mask.device, dtype=torch.bool)

        if hasattr(self, "_internal_mask"):
            self._internal_mask = torch.cat([self._internal_mask, new_internal_mask], dim=0)
        else:
            self._internal_mask = new_internal_mask
        self._register_mobility_grad_hook()


    def freeze_except_mobility(self):
        for group in self.optimizer.param_groups:
            if group.get("name", "") != "mobility":
                group["lr"] = 0.0

    # def _register_mobility_grad_hook(self):
    #     """
    #     Zero-out gradients for non-target mobility
    #     """
    #     if self._mobility is None:
    #         return

    #     mask = self.get_mob_mask

    #     def hook(grad):
    #         # grad: (N, 1) or (N,)
    #         grad = grad.clone()
    #         grad[~mask] = 0.0
    #         return grad

    #     self.grad_hooks.append(self._mobility.register_hook(hook))
    
    def _register_mobility_grad_hook(self):
        # 1. 先清理掉旧的 hooks，防止重复叠加导致内存泄漏或逻辑错误
        if hasattr(self, "mobility_hook_handles"):
            for handle in self.mobility_hook_handles:
                handle.remove()
        self.mobility_hook_handles = []

        if self._mobility is None:
            return

        # 定义 Hook 闭包
        def hook(grad):
            # 实时获取最新的 mask，确保新增的点如果不在 mask 内也会被冻结
            # 注意：这里的 self.get_mob_mask 必须返回一个与当前 grad 形状一致的 Tensor
            current_mask = self.get_mob_mask 
            
            # 即使 self._mobility 长度变了，只要 get_mob_mask 逻辑正确，这里就能匹配
            new_grad = grad.clone()
            # new_grad[~current_mask] = 0.0
            new_grad[~current_mask, :] = 0.0
            return new_grad

        # 注册并记录句柄
        handle = self._mobility.register_hook(hook)
        self.mobility_hook_handles.append(handle)
    
    def apply_mob_mask_hook(self):
        """
        Register gradient hooks so that ONLY gaussians inside mob_mask
        will receive gradients (others are frozen).
        """
        # --------------------------------------------------
        # 1. clear old hooks
        # --------------------------------------------------
        if hasattr(self, "mob_grad_hooks"):
            for h in self.mob_grad_hooks:
                h.remove()
        self.mob_grad_hooks = []

        # --------------------------------------------------
        # 2. get current mob_mask
        # --------------------------------------------------
        mob_mask = self.get_mob_mask  # BoolTensor (N,)

        if mob_mask is None:
            raise RuntimeError("apply_mob_mask_hook: mob_mask is None")

        # --------------------------------------------------
        # 3. define hook
        # --------------------------------------------------
        def mask_grad_hook(grad):
            if grad is None:
                return None
            new_grad = grad.clone()
            new_grad[~mob_mask] = 0.0
            return new_grad

        # --------------------------------------------------
        # 4. register hook for all gaussian params
        # --------------------------------------------------
        params = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ]

        for p in params:
            if p is not None and p.requires_grad:
                h = p.register_hook(mask_grad_hook)
                self.mob_grad_hooks.append(h)
    
    def apply_internal_mask_hook(self):
        """
        为所有需要优化的参数注册 Hook，确保只有 internal_mask 为 True 的点产生梯度。
        """
        # 1. 清理旧 Hook，防止累加
        if hasattr(self, "internal_grad_hooks"):
            for h in self.internal_grad_hooks:
                h.remove()
        self.internal_grad_hooks = []

        # 2. 获取当前的 mask
        # 确保 mask 长度与当前点数一致
        mask = self._internal_mask 

        def freeze_hook(grad):
            # 复制梯度并抹除 mask 之外的部分
            if grad is None: return None
            new_grad = grad.clone()
            new_grad[~mask] = 0.0
            return new_grad

        # 3. 为所有参与训练的属性注册 Hook
        # 属性列表包括：_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation
        trainable_params = [
            self._xyz, 
            self._features_dc, 
            self._features_rest, 
            self._opacity, 
            self._scaling, 
            self._rotation
        ]

        for param in trainable_params:
            if param is not None:
                self.internal_grad_hooks.append(param.register_hook(freeze_hook))

    def add_random_points_inside_object(self, num_new_points=100, expansion_factor=0.1, ratio=0.5, init_opacity=0.3):
        """
        Adds new points **inside** the object defined by `mob_mask`.
        The new points are initialized with mobility 0 (static) and an `internal_mask` indicating they are internal points.
        
        Args:
            num_new_points (int): The number of new points to generate.
            expansion_factor (float): The factor by which to expand the bounding box (in each dimension).
            ratio (float): The ratio of new points to be added based on the existing points.
        
        Returns:
            new_xyz (Tensor): The newly generated points in xyz space.
            new_internal_mask (Tensor): Mask indicating which points are internal.
        """
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype
        for id in self.change_ids:
            N = self.get_xyz.shape[0]
            # 1. Get the points selected by `mob_mask` (these points are part of the dynamic object)
            selected_points = self.get_xyz[self.get_mob_mask].detach()  # (N, 3)
            num_selected = selected_points.shape[0]

            if num_selected == 0:
                print("No points selected by mobility mask.")
                return None, None

            # 2. 基础降噪（保留最大聚类，剔除离群点）
            kmeans = KMeans(n_clusters=1, random_state=0).fit(selected_points.cpu().numpy())
            cluster_labels = torch.tensor(kmeans.labels_, device=device)
            cluster_sizes = torch.bincount(cluster_labels)
            largest_cluster_label = torch.argmax(cluster_sizes)
            filtered_points = selected_points[cluster_labels == largest_cluster_label]

            # 3. 计算 Bounding Box
            min_coords = filtered_points.min(dim=0)[0]  # (3,)
            max_coords = filtered_points.max(dim=0)[0]  # (3,)
            
            # 4. 扩展 Bounding Box
            # expansion_factor=0.1 意味着每一边向外扩张 10% 的尺寸
            bbox_size = max_coords - min_coords
            min_coords = min_coords + bbox_size * expansion_factor
            max_coords = max_coords - bbox_size * expansion_factor
            
            # 5. 确定新增点数
            points_to_add = max(num_new_points, int(num_selected * ratio))

            # 6. 在 Bounding Box 内均匀随机分布采样
            # torch.rand 生成 [0, 1] 之间的随机数，然后线性映射到 [min, max]
            random_offsets = torch.rand(points_to_add, 3, device=device, dtype=dtype)
            new_points = min_coords + random_offsets * (max_coords - min_coords)
            # 7. Create a new internal_mask (True for internal points)
            new_internal_mask = torch.ones(points_to_add, device=device, dtype=torch.bool)  # (points_to_add,)

            # 8. Flatten the new_xyz and new_internal_mask
            new_xyz = new_points.view(-1, 3)  # (points_to_add, 3)
            new_fdc = torch.zeros((points_to_add, 1, 3), device=device)
            new_frest = torch.zeros(
                (points_to_add, (self.active_sh_degree + 1) ** 2 - 1, 3),
                device=device,
                dtype=dtype        
                )
            new_opacity = inverse_sigmoid(
                    torch.full((points_to_add, 1), init_opacity, device=device)
                )

            new_scaling = torch.full((points_to_add, 3), -4.0, device=device, dtype=dtype)
            new_rotation = torch.zeros((points_to_add, 4), device=device, dtype=dtype)
            new_rotation[:, 0] = 1.0

            selected_semantic_features = self.get_semantic_feature[self.get_mob_mask].detach()
            new_sem = selected_semantic_features.mean(dim=0).expand(points_to_add, -1)

            # new_sem = torch.zeros(
            #         (points_to_add, self._semantic_feature.shape[1]),
            #         device=device,
            #         dtype=dtype
            #     )
            # 9. Initialize new mobility to 0 (these points are internal, so they are not movable initially)
            # new_mobility = torch.full((points_to_add,), 0.0, device=device)  # mobility = 0 (static)
            new_mobility = inverse_sigmoid(torch.full((points_to_add,1), 1e-4, device=device))
            
            # 10. Combine the new points with the original ones
            # combined_xyz = torch.cat([self.get_xyz, new_xyz], dim=0)  # (N + points_to_add, 3)
            # combined_mobility = torch.cat([self._mobility, new_mobility], dim=0)  # (N + points_to_add,)
            
            # 11. Update the model with the new points
            new_obj_ids = id.expand(points_to_add)

            self.densification_postfix(
                new_xyz=new_xyz,
                new_features_dc=new_fdc,  # Use the existing features for simplicity
                new_features_rest=new_frest,
                new_opacities=new_opacity,  # Same opacity
                new_scaling=new_scaling,  # Same scaling
                new_rotation=new_rotation,  # Same rotation
                new_semantic_feature=new_sem,  # Same semantic feature
                new_mobility=new_mobility,
            )
            if new_obj_ids is not None:
                self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)
            # 12. Update the internal mask (internal points added)
            if self._internal_mask is None:
                self._internal_mask = torch.cat([torch.zeros(N, dtype=torch.bool, device=device), new_internal_mask], dim=0)
            else:
                self._internal_mask = torch.cat([self._internal_mask, new_internal_mask], dim=0)
            # reset mob_mask
            self._mob_mask = None
        return new_xyz, self._internal_mask
    
    def prune_internal_points(self):
        """
        删除 self._internal_mask 中标记为 True 的所有高斯点。
        """
        # 1. 检查是否存在内部掩码
        if self._internal_mask is None:
            return

        # 2. 统计需要删除的点数
        n_remove = self._internal_mask.sum().item()
        if n_remove == 0:
            return

        print(f"[Internal Prune] Removing {n_remove} internal points.")

        # 3. 确定要删除的掩码 (True 代表要删除)
        mask_to_prune = self._internal_mask

        # 4. 手动更新 _internal_mask 自身
        # 逻辑：保留那些标记为 False (非内部) 的点
        # 结果：self._internal_mask 将变短，且剩下的值全为 False (因为 True 的都删了)
        valid_mask = ~mask_to_prune
        self._internal_mask = self._internal_mask[valid_mask]

        # 5. 调用父类/基类的 prune_points 删除物理参数 (xyz, color, etc.)
        # 注意：这会自动更新 _obj_ids
        self.prune_points(mask_to_prune)

        # 6. 后处理：清理缓存和 Hooks
        self._mob_mask = None  # 索引变了，缓存失效
        
        # 清除针对内部点的梯度 Hook (如果有)
        if hasattr(self, "internal_grad_hooks"):
            for h in self.internal_grad_hooks:
                h.remove()
            self.internal_grad_hooks = []

    def create_from_ckpt(self, ckpt_data, obj_ids=None, change_ids=None, spatial_lr_scale=5.0, semantic_feature_size=3, max_points=None):
        """
        从检查点数据创建 CARGaussianModel。
        优先级：ckpt_data["object_ids"] > 外部传入的 obj_ids > 默认全0
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --------------------------------------------------
        # 1. 从检查点提取基础参数
        # --------------------------------------------------
        xyz = ckpt_data.get("means")
        if xyz is not None: xyz = xyz.to(device)
        
        scaling = ckpt_data.get("scales")
        if scaling is not None: scaling = scaling.to(device)
        
        rotation = ckpt_data.get("quats")
        if rotation is not None: rotation = rotation.to(device)
        
        features_dc = ckpt_data.get("features_dc")
        if features_dc is not None:
            features_dc = features_dc.to(device)
            if features_dc.dim() == 2:
                features_dc = features_dc.unsqueeze(1)
        
        features_rest = ckpt_data.get("features_rest")
        if features_rest is not None: features_rest = features_rest.to(device)
        
        opacity = ckpt_data.get("opacities")
        if opacity is not None: opacity = opacity.to(device)
        
        mobility = ckpt_data.get("mobilities")
        if mobility is not None: mobility = mobility.to(device)

        semantic_feature = None
        if "semantic_feature" in ckpt_data and ckpt_data["semantic_feature"] is not None:
            semantic_feature = ckpt_data["semantic_feature"].to(device)

        # --- [NEW] 提取权重中的 object_ids ---
        ckpt_obj_ids = ckpt_data.get("object_ids")
        if ckpt_obj_ids is not None:
            # 如果权重中存在，则覆盖掉外部传入的参数
            obj_ids = ckpt_obj_ids.to(device)
            # 如果维度是 (N, 1)，压缩为 (N,) 以匹配内部惯例
            if obj_ids.dim() == 2 and obj_ids.shape[1] == 1:
                obj_ids = obj_ids.squeeze(1)

        # --------------------------------------------------
        # [NEW] 1.5 根据 max_points 进行降采样
        # --------------------------------------------------
        if max_points is not None and max_points > 0 and xyz is not None:
            num_points = xyz.shape[0]
            if num_points > max_points:
                print(f"[Create from Ckpt] 正在进行降采样: {num_points} -> {max_points}")
                indices = torch.randperm(num_points, device=device)[:max_points]
                
                xyz = xyz[indices]
                if scaling is not None: scaling = scaling[indices]
                if rotation is not None: rotation = rotation[indices]
                if features_dc is not None: features_dc = features_dc[indices]
                if features_rest is not None: features_rest = features_rest[indices]
                if opacity is not None: opacity = opacity[indices]
                if mobility is not None: mobility = mobility[indices]
                if semantic_feature is not None: semantic_feature = semantic_feature[indices]

                # 同步裁切 obj_ids (此时 obj_ids 可能是权重里的，也可能是外部传入的)
                if obj_ids is not None:
                    if isinstance(obj_ids, (torch.Tensor, np.ndarray)):
                        # 如果是数组形式且长度匹配，进行切片
                        if len(obj_ids) == num_points:
                            obj_ids = obj_ids[indices]

        # --------------------------------------------------
        # 2-3. 初始化基础参数和语义 (逻辑不变)
        # --------------------------------------------------
        if xyz is not None: self._xyz = nn.Parameter(xyz.requires_grad_(True))
        if features_dc is not None: self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        if features_rest is not None: self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        if scaling is not None: self._scaling = nn.Parameter(scaling.requires_grad_(True))
        if rotation is not None: self._rotation = nn.Parameter(rotation.requires_grad_(True))
        if opacity is not None: self._opacity = nn.Parameter(opacity.requires_grad_(True))
        
        if semantic_feature is not None:
            self._semantic_feature = nn.Parameter(semantic_feature.requires_grad_(True))
        else:
            N = xyz.shape[0] if xyz is not None else 0
            self._semantic_feature = nn.Parameter(
                torch.zeros((N, semantic_feature_size), device=device, dtype=torch.float32).requires_grad_(True)
            )
        
        # --------------------------------------------------
        # 4. 初始化对象 ID (核心逻辑修改点)
        # --------------------------------------------------
        N = xyz.shape[0] if xyz is not None else 0
        if obj_ids is not None:
            # 处理 int, float, array, Tensor 等各种可能情况
            if isinstance(obj_ids, (int, float)):
                self._obj_ids = torch.full((N,), int(obj_ids), dtype=torch.int32, device=device)
            elif isinstance(obj_ids, np.ndarray):
                self._obj_ids = torch.tensor(obj_ids, dtype=torch.int32, device=device)
            elif isinstance(obj_ids, torch.Tensor):
                self._obj_ids = obj_ids.to(device).to(torch.int32)
        else:
            self._obj_ids = torch.zeros(N, dtype=torch.int32, device=device)

        # --------------------------------------------------
        # 5. 初始化移动性参数
        # --------------------------------------------------
        if mobility is not None:
            self._mobility = nn.Parameter(mobility.requires_grad_(True))
        else:
            # 如果检查点中没有 mobility，初始化为 0（静态）
            N = xyz.shape[0] if xyz is not None else 0
            mobility_init = torch.zeros((N, 1), device=device, dtype=torch.float32)
            mobility_raw = self.inverse_mobility_activation(mobility_init.clamp(1e-4, 1 - 1e-4))
            self._mobility = nn.Parameter(mobility_raw.requires_grad_(True))
        
        # --------------------------------------------------
        # 6. 初始化 change_ids
        # --------------------------------------------------
        if change_ids is not None:
            if not torch.is_tensor(change_ids):
                change_ids = torch.tensor(change_ids, device=device)
            self.change_ids = change_ids
        else:
            # 如果没有提供 change_ids，使用所有唯一的 obj_ids
            self.change_ids = torch.unique(self._obj_ids)
        
        # --------------------------------------------------
        # 7. 初始化其他必要的属性
        # --------------------------------------------------
        N = xyz.shape[0] if xyz is not None else 0
        self.max_radii2D = torch.zeros(N, device=device)
        self.spatial_lr_scale = spatial_lr_scale
        
        # 初始化梯度相关的属性
        self.xyz_gradient_accum = torch.zeros((N, 1), device=device)
        self.denom = torch.zeros((N, 1), device=device)
        
        # 重建 mob_mask
        self._mob_mask = None 
        
        # 初始化内部掩码
        self._internal_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        print(
            f"[Create from Ckpt] 成功加载 {N} 个高斯核\n"
            f"  ✓ xyz 形状: {self._xyz.shape}\n"
            f"  ✓ features_dc 形状: {self._features_dc.shape}\n"
            f"  ✓ features_rest 形状: {self._features_rest.shape}\n"
            f"  ✓ scaling 形状: {self._scaling.shape}\n"
            f"  ✓ rotation 形状: {self._rotation.shape}\n"
            f"  ✓ opacity 形状: {self._opacity.shape}\n"
            f"  ✓ mobility 形状: {self._mobility.shape}\n"
            f"  ✓ semantic_feature 形状: {self._semantic_feature.shape}\n"
            f"  ✓ change_ids: {self.change_ids.tolist() if torch.is_tensor(self.change_ids) else self.change_ids}"
        )
        
    # def create_from_ckpt(self, ckpt_data, obj_ids=None, change_ids=None, spatial_lr_scale=5.0, semantic_feature_size=3):
    #     """
    #     从检查点数据创建 CARGaussianModel。
        
    #     Args:
    #         ckpt_data (dict): 检查点数据，包含以下键：
    #             - means: Tensor (N, 3) - 高斯核位置
    #             - scales: Tensor (N, 3) - 缩放参数
    #             - quats: Tensor (N, 4) - 旋转四元数
    #             - features_dc: Tensor (N, 3) - 颜色 DC 分量
    #             - features_rest: Tensor (N, 15, 3) - 颜色高阶分量
    #             - opacities: Tensor (N, 1) - 不透明度
    #             - mobilities: Tensor (N, 1) - 移动性参数（RAW 空间）
    #         obj_ids (Tensor or np.ndarray, optional): 对象 ID，形状 (N,)
    #         change_ids (list or Tensor, optional): 可动对象的 ID 列表
    #         spatial_lr_scale (float): 空间学习率缩放因子
    #         semantic_feature_size (int): 语义特征维度
        
    #     Returns:
    #         None (直接修改 self)
    #     """
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
        
    #     # --------------------------------------------------
    #     # 1. 从检查点提取基础参数并进行格式转换
    #     # --------------------------------------------------
        
    #     # means -> _xyz
    #     xyz = ckpt_data.get("means")
    #     if xyz is not None:
    #         xyz = xyz.to(device)
        
    #     # scales -> _scaling
    #     scaling = ckpt_data.get("scales")
    #     if scaling is not None:
    #         scaling = scaling.to(device)
        
    #     # quats -> _rotation
    #     rotation = ckpt_data.get("quats")
    #     if rotation is not None:
    #         rotation = rotation.to(device)
        
    #     # features_dc 需要转换形状从 (N, 3) 到 (N, 1, 3)
    #     features_dc = ckpt_data.get("features_dc")
    #     if features_dc is not None:
    #         features_dc = features_dc.to(device)# 如果形状是 (N, 3)，转换为 (N, 1, 3)
    #         if features_dc.dim() == 2:
    #             features_dc = features_dc.unsqueeze(1)
        
    #     # features_rest 需要转换形状从 (N, 15, 3) 到 (N, 15, 3)（已经是正确形状）
    #     features_rest = ckpt_data.get("features_rest")
    #     if features_rest is not None:
    #         features_rest = features_rest.to(device)
        
    #     # opacities -> _opacity
    #     opacity = ckpt_data.get("opacities")
    #     if opacity is not None:
    #         opacity = opacity.to(device)
        
    #     # mobilities -> _mobility
    #     mobility = ckpt_data.get("mobilities")
    #     if mobility is not None:
    #         mobility = mobility.to(device)
        
    #     # --------------------------------------------------
    #     # 2. 初始化基础参数为 nn.Parameter
    #     # --------------------------------------------------
    #     if xyz is not None:
    #         self._xyz = nn.Parameter(xyz.requires_grad_(True))
        
    #     if features_dc is not None:
    #         self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        
    #     if features_rest is not None:
    #         self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        
    #     if scaling is not None:
    #         self._scaling = nn.Parameter(scaling.requires_grad_(True))
        
    #     if rotation is not None:
    #         self._rotation = nn.Parameter(rotation.requires_grad_(True))
        
    #     if opacity is not None:
    #         self._opacity = nn.Parameter(opacity.requires_grad_(True))
        
    #     # --------------------------------------------------
    #     # 3. 初始化语义特征（如果检查点中没有，则创建零张量）
    #     # --------------------------------------------------
    #     if "semantic_feature" in ckpt_data and ckpt_data["semantic_feature"] is not None:
    #         semantic_feature = ckpt_data["semantic_feature"].to(device)
    #         self._semantic_feature = nn.Parameter(semantic_feature.requires_grad_(True))
    #     else:
    #         N = xyz.shape[0] if xyz is not None else 0
    #         self._semantic_feature = nn.Parameter(
    #             torch.zeros((N, semantic_feature_size), device=device, dtype=torch.float32).requires_grad_(True)
    #         )
        
    #     # --------------------------------------------------
    #     # 4. 初始化对象 ID
    #     # --------------------------------------------------
    #     if obj_ids is not None:
    #         if isinstance(obj_ids, (int, float)):
    #             # 如果 obj_ids 是一个数字，创建全部是这个数字的 tensor
    #             N = xyz.shape[0] if xyz is not None else 0
    #             obj_ids = torch.full((N,), int(obj_ids), dtype=torch.int32, device=device)
    #         elif isinstance(obj_ids, np.ndarray):
    #             obj_ids = torch.tensor(obj_ids, dtype=torch.int32, device=device)
    #         elif isinstance(obj_ids, torch.Tensor):
    #             obj_ids = obj_ids.to(device)
    #         self._obj_ids = obj_ids
    #     else:
    #         # 如果没有提供 obj_ids，创建默认的（全为 0）
    #         N = xyz.shape[0] if xyz is not None else 0
    #         self._obj_ids = torch.zeros(N, dtype=torch.int32, device=device)
    #     # --------------------------------------------------
    #     # 5. 初始化移动性参数
    #     # --------------------------------------------------
    #     if mobility is not None:
    #         self._mobility = nn.Parameter(mobility.requires_grad_(True))
    #     else:
    #         # 如果检查点中没有 mobility，初始化为 0（静态）
    #         N = xyz.shape[0] if xyz is not None else 0
    #         mobility_init = torch.zeros((N, 1), device=device, dtype=torch.float32)
    #         mobility_raw = self.inverse_mobility_activation(mobility_init.clamp(1e-4, 1 - 1e-4))
    #         self._mobility = nn.Parameter(mobility_raw.requires_grad_(True))
        
    #     # --------------------------------------------------
    #     # 6. 初始化 change_ids
    #     # --------------------------------------------------
    #     if change_ids is not None:
    #         if not torch.is_tensor(change_ids):
    #             change_ids = torch.tensor(change_ids, device=device)
    #         self.change_ids = change_ids
    #     else:
    #         # 如果没有提供 change_ids，使用所有唯一的 obj_ids
    #         self.change_ids = torch.unique(self._obj_ids)
        
    #     # --------------------------------------------------
    #     # 7. 初始化其他必要的属性
    #     # --------------------------------------------------
    #     N = xyz.shape[0] if xyz is not None else 0
    #     self.max_radii2D = torch.zeros(N, device=device)
    #     self.spatial_lr_scale = spatial_lr_scale
        
    #     # 初始化梯度相关的属性（在 training_setup 中会重新初始化）
    #     self.xyz_gradient_accum = torch.zeros((N, 1), device=device)
    #     self.denom = torch.zeros((N, 1), device=device)
        
    #     # 重建 mob_mask
    #     self._mob_mask = None  # 会在 get_mob_mask 属性中动态计算
        
    #     # 初始化内部掩码（所有点初始化为非内部点）
    #     self._internal_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
    #     print(
    #         f"[Create from Ckpt] 成功加载 {N} 个高斯核\n"
    #         f"  ✓ xyz 形状: {self._xyz.shape}\n"
    #         f"  ✓ features_dc 形状: {self._features_dc.shape}\n"
    #         f"  ✓ features_rest 形状: {self._features_rest.shape}\n"
    #         f"  ✓ scaling 形状: {self._scaling.shape}\n"
    #         f"  ✓ rotation 形状: {self._rotation.shape}\n"
    #         f"  ✓ opacity 形状: {self._opacity.shape}\n"
    #         f"  ✓ mobility 形状: {self._mobility.shape}\n"
    #         f"  ✓ semantic_feature 形状: {self._semantic_feature.shape}\n"
    #         f"  ✓ change_ids: {self.change_ids.tolist() if torch.is_tensor(self.change_ids) else self.change_ids}"
    #     )
    def load_ply(self, path, og_number_points=-1):
        """
        Load PLY file for CARGaussianModel.
        
        If PLY contains 'mobility' field, load it directly.
        Otherwise, call parent's load_ply and initialize mobility to negative (static).
        
        Args:
            path (str): Path to PLY file
            og_number_points (int): Original number of points (for tracking)
        """
        # 先检查 PLY 文件中是否包含 mobility 字段
        plydata = PlyData.read(path)
        vertex = plydata.elements[0].data
        
        has_mobility = "mobility" in vertex.dtype.names
        
        if not has_mobility:
            # 如果没有 mobility，调用父类的 load_ply
            print("[load_ply] No mobility found in PLY, calling parent's load_ply...")
            super().load_ply(path, og_number_points)
            
            # 初始化 mobility 为负数（表示静态）
            N = self._xyz.shape[0]
            # 使用负数表示静态（例如 -1），这样 sigmoid 后会接近 0
            mobility_raw = torch.full((N, 1), -10.0, dtype=torch.float32, device="cuda")
            self._mobility = nn.Parameter(mobility_raw, requires_grad=True)
            
            # 初始化 CARGaussianModel 特定的属性
            self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
            self.denom = torch.zeros((N, 1), device="cuda")
            self._mob_mask = None
            self._internal_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
            
            print(f"[load_ply] Mobility initialized to -10.0 (static) for {N} Gaussians")
            return
        
        # 如果有 mobility，执行自己的加载逻辑
        print("[load_ply] Found mobility in PLY, loading with CARGaussianModel logic...")
        
        self.og_number_points = og_number_points

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

        # === obj_id ===
        if "obj_id" in vertex.dtype.names:
            obj_ids = vertex["obj_id"].astype(np.int32)
            self._obj_ids = torch.tensor(obj_ids, dtype=torch.int32, device="cuda")
        else:
            print("[load_ply] No obj_id found in PLY.")
            self._obj_ids = None

        # === mobility ===
        mobility = vertex["mobility"][..., np.newaxis]  # (N, 1)

        # ===== Convert to torch.nn.Parameters =====
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda")
                                        .transpose(1, 2).contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(), requires_grad=True)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda"), requires_grad=True)
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float32, device="cuda"), requires_grad=True)
        
        # === mobility parameter ===
        self._mobility = nn.Parameter(torch.tensor(mobility, dtype=torch.float32, device="cuda"), requires_grad=True)

        # === initialize runtime buffers ===
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        
        # === initialize CARGaussianModel specific buffers ===
        N = xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self._mob_mask = None
        self._internal_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
        
        print(f"[load_ply] CARGaussianModel loaded with {N} Gaussians (with mobility)")

    def save_ply(self, path):
        """
        Save PLY file for CARGaussianModel, including mobility parameter.
        
        Args:
            path (str): Path to save PLY file
        """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().cpu().numpy()

        # obj_ids (must be int32 for PLY)
        if hasattr(self, "_obj_ids"):
            obj_ids = self._obj_ids
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.detach().cpu().numpy()
            obj_ids = obj_ids.astype(np.int32).reshape(-1, 1)
        else:
            obj_ids = None

        # mobility (CARGaussianModel specific)
        if hasattr(self, "_mobility"):
            mobility = self._mobility.detach().cpu().numpy()
        else:
            mobility = None

        # original dtype list
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # add obj_id dtype (int32)
        if obj_ids is not None:
            dtype_full.append(("obj_id", "i4"))

        # add mobility dtype (float32)
        if mobility is not None:
            dtype_full.append(("mobility", "f4"))

        # base attributes
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature),
            axis=1
        )

        # append obj_ids into attribute matrix
        if obj_ids is not None:
            attributes = np.concatenate((attributes, obj_ids), axis=1)

        # append mobility into attribute matrix
        if mobility is not None:
            attributes = np.concatenate((attributes, mobility), axis=1)

        # write to ply
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        print(f"[save_ply] CARGaussianModel saved to {path} with {xyz.shape[0]} Gaussians")