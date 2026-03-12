import torch
import numpy as np
from torch import nn
from utils.general_utils import inverse_sigmoid
from scene.gaussian_model_sc import GaussianModelSC

class CARGaussianModelSC(GaussianModelSC):
    """
    Extend the original GaussianModel to support ScrewSplat-style articulation.
    """
    def __init__(self, sh_degree, n_screws=10):
        super().__init__(sh_degree)

        self.n_screws = n_screws
        self.change_ids = None
        self._internal_mask = None 
        self.initialized_articulation = False
        self.grad_hooks = [] 

    def init_articulation(self, screw_init):
        """
        screw_init:
        {
            "change_ids": [29, 31, 45],
            "screws": {
                "29": { "axis_point": [...], "axis_dir": [...], "top_ids": [...] },
                "31": { "axis_point": [...], "axis_dir": [...], "top_ids": [...] },
                ...
            }
        }

        self.get_obj_ids: [N] tensor for each gaussian
        """

        if self.initialized_articulation:
            return

        device = self._xyz.device
        N = self._xyz.shape[0]
        self._internal_mask = torch.zeros(N, dtype=torch.bool, device=device)
        obj_ids = self.get_obj_ids

        change_ids = screw_init["ids"]
        screws_dict = screw_init["screw_init"]
        self.change_ids = change_ids

        n_screws = len(change_ids)
        self.n_screws = n_screws

        print(f"[Init Articulation] N={N} Gaussians, screws={n_screws}")

        # -----------------------------------------
        # 1) 初始化 part indices (N × (1 + n_screws))
        # -----------------------------------------
        part = torch.zeros((N, n_screws + 1), device=device)

        # 默认所有都属于 root
        part[:, 0] = 1.0

        # change_ids 映射到 screw index
        # e.g., {29: 0, 31: 1, 45: 2}
        oid_to_sid = {oid: sid for sid, oid in enumerate(change_ids)}

        # 给 changed objects 提升 part weight
        for i in range(N):
            oid = int(obj_ids[i])
            if oid not in oid_to_sid:
                continue  # 静态对象 → root remains 1.0

            sid = oid_to_sid[oid]
            info = screws_dict[str(oid)]
            top_ids = info.get("top_global_ids", [])

            # --------------------------
            # top_global_ids 高亮
            # --------------------------
            if i in top_ids:
                part[i, 0] = 0.1        # root 降低
                part[i, sid + 1] = 1.0  # 主轴强依赖
            else:
                part[i, 0] = 1.0
                part[i, sid + 1] = 0.3 # 普通点均匀依赖

            # # root 稍微降低
            # part[i, 0] = 0.1
            # # 属于对应的 screw
            # part[i, sid + 1] = 1.0

        # 归一化
        # part = part / (part.sum(dim=1, keepdim=True) + 1e-9)

        self._part_indices = nn.Parameter(part, requires_grad=True)

        # -----------------------------------------
        # 2) 初始化 screw 参数 (ω, v)
        # -----------------------------------------
        screws = torch.zeros((n_screws, 6), device=device)

        for sid, oid in enumerate(change_ids):
            info = screws_dict[str(oid)]

            p = torch.tensor(info["axis_point"], device=device, dtype=torch.float32)
            w = torch.tensor(info["axis_dir"], device=device, dtype=torch.float32)

            w_init = w / (w.norm() + 1e-9)
            q_init = p 

            screws[sid, :3] = w_init          # w_raw（未归一化）
            screws[sid, 3:] = q_init 

        self._screws = nn.Parameter(screws, requires_grad=True)

        # -----------------------------------------
        # 3) 初始化 screw confidence γ_j
        # -----------------------------------------
        gamma = torch.ones(n_screws, device=device) * 0.8
        self._screw_confs = nn.Parameter(gamma, requires_grad=True)

        # -----------------------------------------
        # 4) 初始化 joint angles θ_j
        # -----------------------------------------

        desired_angles = torch.tensor([0.0, torch.pi/4], device=device)  # [2]

        joint_angles = []
        for theta in desired_angles:
            # 将真实角度 theta 映射到内部的 x
            x = self.angles_to_vectors(theta).repeat(n_screws)
            joint_angles.append(x)

        # 转为 Parameter
        self._joint_angles = nn.ParameterList([
            nn.Parameter(ja.clone().detach().requires_grad_(True)) for ja in joint_angles
        ])
        self.initialized_articulation = True

        print("[Init Articulation] Done.")

    def init_articulation_xyzrand(self, screw_init):
        """
        对每个物体生成 3 根 screw 轴：
            w0 = 原始轴
            w1 = 任意垂直方向
            w2 = w0 × w1
        """

        if self.initialized_articulation:
            return

        device = self._xyz.device
        N = self._xyz.shape[0]
        obj_ids = self.get_obj_ids

        change_ids = screw_init["ids"]
        screws_dict = screw_init["screw_init"]
        self.change_ids = change_ids

        original_screws = len(change_ids)
        screws_per_obj = 3                          # 每个物体 3 根轴
        n_screws = original_screws * screws_per_obj
        self.n_screws = n_screws

        print(f"[Init Articulation] N={N} Gaussians, screws={n_screws} (3 per object)")

        # ====================================================
        # 1) part_indices 初始化
        #     part[i] ∈ ℝ^(1 + n_screws)
        # ====================================================
        part = torch.zeros((N, n_screws + 1), device=device)
        part[:, 0] = 1.0  # root part

        # 原始：{29:0, 31:1 ...}
        oid_to_sid = {oid: sid for sid, oid in enumerate(change_ids)}

        # for i in range(N):
        #     oid = int(obj_ids[i])
        #     if oid not in oid_to_sid:
        #         continue

        #     base_sid = oid_to_sid[oid] * screws_per_obj  # 该物体的 3 条轴起始 index

        #     # root 降低
        #     part[i, 0] = 1.0

        #     # 属于 3 条轴（等权）
        #     part[i, base_sid + 1 : base_sid + 1 + screws_per_obj] = 1.0
        for i in range(N):
            oid = int(obj_ids[i])
            if oid not in oid_to_sid:
                continue

            base_sid = oid_to_sid[oid] * screws_per_obj
            info = screws_dict[str(oid)]
            top_ids = info.get("top_global_ids", [])

            # 是否在 top_global_ids 内
            is_top = (i in top_ids)

            # root 权重
            part[i, 0] = 0.1 if is_top else 0.1

            if is_top:
                # top 高亮点 → 让其强烈依赖主轴 w0
                part[i, base_sid + 1] = 3.0   # w0
                part[i, base_sid + 2] = 1.0   # w1
                part[i, base_sid + 3] = 1.0   # w2
            else:
                # 普通点 → 仍然均匀分布
                part[i, base_sid + 1 : base_sid + 1 + screws_per_obj] = 1.0

        # normalize
        part = part / (part.sum(dim=1, keepdim=True) + 1e-9)
        self._part_indices = nn.Parameter(part, requires_grad=True)

        # ====================================================
        # 2) screw 参数初始化 (ω, q)
        # ====================================================
        screws = torch.zeros((n_screws, 6), device=device)

        for obj_i, oid in enumerate(change_ids):
            info = screws_dict[str(oid)]

            p = torch.tensor(info["axis_point"], device=device, dtype=torch.float32)
            w0 = torch.tensor(info["axis_dir"], device=device, dtype=torch.float32)
            w0 = w0 / (w0.norm() + 1e-9)

            # ----------- 构造 w1（与 w0 正交）------------
            # 任选一个不平行的向量
            tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
            if torch.abs(torch.dot(tmp, w0)) > 0.9:
                tmp = torch.tensor([0.0, 1.0, 0.0], device=device)

            w1 = tmp - (tmp @ w0) * w0
            w1 = w1 / (w1.norm() + 1e-9)

            # ----------- 构造 w2 = w0 × w1 ------------
            w2 = torch.cross(w0, w1)
            w2 = w2 / (w2.norm() + 1e-9)

            base = obj_i * screws_per_obj
            screws[base + 0, :3] = w0
            screws[base + 1, :3] = w1
            screws[base + 2, :3] = w2

            screws[base + 0, 3:] = p
            screws[base + 1, 3:] = p
            screws[base + 2, 3:] = p

        self._screws = nn.Parameter(screws, requires_grad=True)

        # ====================================================
        # 3) screw confidence 初始化
        # ====================================================
        gamma = torch.ones(n_screws, device=device) * 0.6
        self._screw_confs = nn.Parameter(gamma, requires_grad=True)

        # ====================================================
        # 4) joint angles 初始化（仍然两组角度）
        # ====================================================
        desired_angles = torch.tensor([0.0, -torch.pi/4], device=device)

        joint_angles = []
        for theta in desired_angles:
            x = self.angles_to_vectors(theta).repeat(n_screws)
            joint_angles.append(x)

        self._joint_angles = nn.ParameterList([
            nn.Parameter(ja.clone().detach().requires_grad_(True)) for ja in joint_angles
        ])

        self.initialized_articulation = True
        print("[Init Articulation] Done (3 axes per object).")

    def screw_activation(self, raw_screws):
        """
        输入 raw_screws: [N, 6] = [w_raw(3), p_raw(3)]
        输出 screw: [N, 6] = [ω(3), v(3)]，可用于 exp_se3
        """
        # 1) 提取 w_raw 和 p_raw
        w_raw = raw_screws[:, :3]          
        p_raw = raw_screws[:, 3:]          

        # 2) 旋转方向必须是单位向量
        w = w_raw / (w_raw.norm(dim=1, keepdim=True) + 1e-9)
        # 3) 构造合法的 screw axis
        v = -torch.cross(w, p_raw, dim=1)

        # 4) 输出 twist 坐标 (ω, v)
        return torch.cat([w, v], dim=1)
    
    def vectors_to_angles(self, x):
        return torch.pi * torch.sigmoid(x) - torch.pi/2

    def angles_to_vectors(self, theta):
        theta = torch.clip(theta, -torch.pi/2, torch.pi/2)
        return inverse_sigmoid(theta / torch.pi + 0.5)

    # =======  Getter functions  =======
    @property
    def get_change_ids(self):
        return self.change_ids
    
    @property
    def get_internal_mask(self):
        return self._internal_mask
    
    @property
    def get_part_indices_hard(self):
        hard_ids = torch.argmax(self._part_indices, dim=1)  # shape [N]
        return hard_ids
    
    @property
    def get_part_indices(self):
        return torch.softmax(self._part_indices, dim=1) # m_{ij}

    @property
    def get_screws(self):
        return self.screw_activation(self._screws)

    @property
    def get_screw_confs(self):
        return torch.sigmoid(self._screw_confs)  # γ_j
    
    @property
    def get_joint_angles(self):
        return [
            self.vectors_to_angles(joint_angle) for joint_angle in self._joint_angles]
    
    def training_setup(self, training_args):
        """
        Extend GaussianModel.training_setup() to add articulation optimization.
        """

        super().training_setup(training_args)

        # -----------------------------
        # 针对 point-level 优化器的额外参数（part indices）
        # -----------------------------
        # 将 part_indices 加到 self.optimizer
        self.optimizer.add_param_group({
            'params': [self._part_indices],
            'lr': training_args.part_index_lr,
            'name': "part_indices"
        })

        self.screw_optimizer = torch.optim.Adam(
            [{'params': [self._screws], 'lr': training_args.screw_lr, 'name': 'screw'}],
            lr=0.0,
            eps=1e-15
        )

        self.screw_conf_optimizer = torch.optim.Adam(
            [{'params': [self._screw_confs], 'lr': training_args.screw_confidence_lr, 'name': 'screw_conf'}],
            lr=0.0,
            eps=1e-15
        )

        # self.joint_angle_optimizers = []
        # for j in range(len(self._joint_angles)):
        #     opt = torch.optim.Adam(
        #         [{'params': [self._joint_angles[j]], 'lr': training_args.joint_angle_lr,
        #         'name': f'joint_angle_{j}'}],
        #         lr=0.0,
        #         eps=1e-15
        #     )
        #     self.joint_angle_optimizers.append(opt)
        self.joint_angle_optimizer = [
            torch.optim.Adam(
                [{'params': [joint_angle], 'lr': training_args.joint_angle_lr, "name": f"joint_angle_{i}"}], lr=0.0, eps=1e-15) 
            for i, joint_angle in enumerate(self._joint_angles)]

        print("[Training Setup] Articulation optimizers initialized.")

    # Extend capture() so checkpoints include articulation
    def capture(self):
        parent_data = super().capture()
        return parent_data + (
            self._part_indices,
            self._screws,
            self._screw_confs,
            self._joint_angles,
            self.change_ids,
            self._internal_mask
        )
    def restore(self, model_args, training_args):
        """
        Overwrite GaussianModel.restore() to additionally restore articulation params.
        """

        # -----------------------------
        # 1. 拆分前 N 个字段（GaussianModel 的）
        # -----------------------------
        base_len = 14    # GaussianModel.capture() 返回 14 个字段
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
        ) = model_args[:base_len]

        extra_args = model_args[base_len:]

        if len(extra_args) == 6:
            # ckpt 中已经包含 articulation 参数
            self._part_indices, self._screws, self._screw_confs, self._joint_angles, self.change_ids, self._internal_mask = extra_args
            if not isinstance(self._joint_angles, nn.ParameterList):
                self._joint_angles = nn.ParameterList(list(self._joint_angles))

            # self._part_indices = nn.Parameter(part_indices.to(self._xyz.device), requires_grad=True)
            # self._screws       = nn.Parameter(screws.to(self._xyz.device), requires_grad=True)
            # self._screw_confs  = nn.Parameter(screw_confs.to(self._xyz.device), requires_grad=True)
            # self._joint_angles = nn.Parameter(joint_angles.to(self._xyz.device), requires_grad=True)

            self.n_screws = self._screws.shape[0]
            # self.change_ids = None
            self.initialized_articulation = True

            print(f"[Restore] Articulation restored: {self.n_screws} screws.")

        else:
            print("[Restore] No articulation params in ckpt → call init_articulation() later.")
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def densification_postfix(
        self,
        new_xyz, new_features_dc, new_features_rest,
        new_opacities, new_scaling, new_rotation, new_semantic_feature,
        new_part_indices=None,
    ):
        if hasattr(self, "_part_indices") and self._part_indices is not None:
            if new_part_indices is None:
                # 默认：新点全 root
                num_new = new_xyz.shape[0]
                K = self.get_part_indices.shape[1]
                new_part_indices = torch.zeros((num_new, K), device=self._part_indices.device, dtype=self._part_indices.dtype)
                new_part_indices[:, 0] = 1.0
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "semantic_feature": new_semantic_feature,
            "part_indices": new_part_indices
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        self._part_indices = optimizable_tensors["part_indices"]
        # buffers reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def all_parameters(self):
        """
        返回一个有序 dict:
        {
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "semantic_feature": self._semantic_feature (如果存在),
            "part_indices": self._part_indices,
            "screws": self._screws,
            "screw_confs": self._screw_confs,
            "joint_angle_0": self._joint_angles[0],
            "joint_angle_1": self._joint_angles[1],
            ...
        }
        """
        params = {}

        # Gaussian base parameters
        params["xyz"] = self._xyz
        params["features_dc"] = self._features_dc
        params["features_rest"] = self._features_rest
        params["scaling"] = self._scaling
        params["rotation"] = self._rotation
        params["opacity"] = self._opacity

        # semantic feature (if exists)
        if hasattr(self, "_semantic_feature"):
            params["semantic_feature"] = self._semantic_feature

        # Articulation parameters
        if hasattr(self, "_part_indices"):
            params["part_indices"] = self._part_indices

        if hasattr(self, "_screws"):
            params["screws"] = self._screws

        if hasattr(self, "_screw_confs"):
            params["screw_confs"] = self._screw_confs

        # Joint angles (list)
        if hasattr(self, "_joint_angles"):
            for i, ja in enumerate(self._joint_angles):
                params[f"joint_angle_{i}"] = ja

        return params
    
    def get_deform_mask(self):
        """
        True  : 原始 change_id 高斯（参与 deform）
        False : 静态点 + 内部新增点
        """

        assert self.change_ids is not None, "call init_deform() first"
        assert self._internal_mask is not None, "internal_mask not initialized"

        obj_ids = self.get_obj_ids
        device = obj_ids.device
        
        change_ids = torch.tensor(
            self.change_ids, device=device, dtype=obj_ids.dtype
        )

        obj_mask = torch.isin(obj_ids, change_ids)
        deform_mask = obj_mask & (~self._internal_mask)

        return deform_mask
    
    def freeze_gaussians(self):
        for p in [
            self._xyz, self._scaling, self._rotation,
            self._opacity, self._features_dc, self._features_rest,
            self._semantic_feature
        ]:
            p.requires_grad_(False)
    
    def add_internal_gaussians_and_freeze_others(
        self,
        ratio,
        training_args,
        min_per_object=1000,
        xyz_noise_scale=0.01,
        init_opacity=0.1,
        internal_obj_id=-1,
    ):
        """
        在 change_ids 对应物体内部添加高斯
        - 冻结旧高斯
        - 仅训练新增高斯
        """

        assert self.change_ids is not None, \
            "change_ids is None, call init_deform() first."

        device = self._xyz.device
        dtype = self._xyz.dtype

        obj_ids = self.get_obj_ids
        change_ids = self.change_ids

        # 记录旧点数量
        n_old = self.get_xyz.shape[0]

        new_xyz_all = []
        new_fdc_all = []
        new_frest_all = []
        new_opacity_all = []
        new_scaling_all = []
        new_rotation_all = []
        new_semantic_all = []
        new_obj_id_all = []
        new_part_indices_all = []

        for oid in change_ids:
            mask = (obj_ids == oid)
            num_exist = mask.sum().item()
            if num_exist == 0:
                continue

            num_new = max(int(num_exist * ratio), min_per_object)

            xyz_obj = self._xyz[mask]

            # -------- 稳健中心 + 立方体采样 --------
            center = xyz_obj.mean(dim=0)
            scales_obj = self.get_scaling[mask]
            avg_scale = scales_obj.mean().item()

            cube_half = 7.0 * avg_scale

            new_xyz = center + (torch.rand((num_new, 3), device=device, dtype=dtype) - 0.5) \
                                * 2.0 * cube_half

            # -------- scaling / rotation --------
            # new_scaling = torch.full((num_new, 3), -4.0, device=device, dtype=dtype)
            # internal latent scale（只 1 维）
            new_scaling = torch.full((num_new, 1), -4.0, device=device, dtype=dtype)

            new_rotation = torch.zeros((num_new, 4), device=device, dtype=dtype)
            new_rotation[:, 0] = 1.0

            # -------- opacity --------
            new_opacity = inverse_sigmoid(
                torch.full((num_new, 1), init_opacity, device=device, dtype=dtype)
            )

            # -------- SH --------
            new_fdc = torch.zeros((num_new, 1, 3), device=device, dtype=dtype)
            new_frest = torch.zeros(
                (num_new, (self.active_sh_degree + 1) ** 2 - 1, 3),
                device=device,
                dtype=dtype
            )

            # -------- semantic --------
            new_sem = torch.zeros(
                (num_new, self._semantic_feature.shape[1]),
                device=device,
                dtype=dtype
            )

            # -------- obj_id --------
            new_obj_id = torch.full(
                (num_new,), oid, device=device, dtype=torch.int32
            )

            # -------- part_indices --------
            n_parts = self.get_part_indices.shape[1]
            new_part = torch.zeros((num_new, n_parts), device=device, dtype=dtype)
            new_part[:, 0] = 1.0   # root = 1.0

            new_xyz_all.append(new_xyz)
            new_fdc_all.append(new_fdc)
            new_frest_all.append(new_frest)
            new_opacity_all.append(new_opacity)
            new_scaling_all.append(new_scaling)
            new_rotation_all.append(new_rotation)
            new_semantic_all.append(new_sem)
            new_obj_id_all.append(new_obj_id)
            new_part_indices_all.append(new_part)

            print(f"[Add Internal GS] oid={oid}: {num_exist} → +{num_new}")

        # -------- 拼接 --------
        new_xyz = torch.cat(new_xyz_all, 0)
        new_fdc = torch.cat(new_fdc_all, 0)
        new_frest = torch.cat(new_frest_all, 0)
        new_opacity = torch.cat(new_opacity_all, 0)
        new_scaling = torch.cat(new_scaling_all, 0)
        new_rotation = torch.cat(new_rotation_all, 0)
        new_semantic = torch.cat(new_semantic_all, 0)
        new_obj_ids = torch.cat(new_obj_id_all, 0)
        new_part_indices = torch.cat(new_part_indices_all, 0)

        # -------- 官方增点流程 --------
        self.densification_postfix(
            new_xyz,
            new_fdc,
            new_frest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_semantic,
            new_part_indices
        )

        self._obj_ids = torch.cat([self._obj_ids, new_obj_ids], dim=0)
        # ===== internal_mask 扩展 =====
        new_internal_mask = torch.ones(
            new_xyz.shape[0], dtype=torch.bool, device=device
        )
        self._internal_mask = torch.cat(
            [self._internal_mask, new_internal_mask], dim=0
        )
        # -------- 冻结旧点 --------
        # with torch.no_grad():
        #     for p in [
        #         self._xyz,
        #         self._features_dc,
        #         self._features_rest,
        #         self._opacity,
        #         self._scaling,
        #         self._rotation,
        #         self._semantic_feature,
        #     ]:
        #         p[:n_old].requires_grad_(False)
        #         p[n_old:].requires_grad_(True)
        self._freeze_old_params_via_hook(n_old)

        # self.optimizer.add_param_group({
        #     "params": [self._internal_log_scale],
        #     "lr": training_args.scaling_lr if hasattr(training_args, "scaling_lr") else 1e-3,
        #     "name": "internal_log_scale",
        # })
        print(f"[Add Internal GS] Done. Total points: {self.get_xyz.shape[0]}")

    def _freeze_old_params_via_hook(self, n_old):
        """
        通过注册 hook，在反向传播时强行将前 n_old 个点的梯度置为 0。
        这是在不拆分 Tensor 的情况下冻结部分点的唯一标准做法。
        """
        # 定义钩子函数
        def get_zero_grad_hook(n):
            def hook(grad):
                # 克隆梯度（避免 inplace 错误），并将前 n 个设为 0
                g = grad.clone()
                g[:n] = 0
                return g
            return hook

        # 清除旧的 hooks (如果有)
        for h in self.grad_hooks:
            h.remove()
        self.grad_hooks = []

        # 需要冻结的参数列表
        params_to_freeze = [
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._opacity,
            self._scaling, 
            self._rotation,
            self._semantic_feature,
            self._part_indices
        ]
        
        # 注册新 hooks
        print(f"[Freeze] Locking gradients for the first {n_old} points.")
        for p in params_to_freeze:
            if p.requires_grad:
                h = p.register_hook(get_zero_grad_hook(n_old))
                self.grad_hooks.append(h)