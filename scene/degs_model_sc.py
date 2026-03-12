import torch
from torch import nn
from utils.general_utils import inverse_sigmoid
from scene.gaussian_model_sc import GaussianModelSC


class DeGSModelSC(GaussianModelSC):
    """
    Deform-only Gaussian Model
    - supports change_ids
    - supports internal Gaussian insertion
    - supports deform mask
    - NO articulation / screw / joint parameters
    """

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        self.change_ids = None
        self._internal_mask = None 
        # self._internal_log_scale = None
        self.initialized_deform = False
        self.grad_hooks = [] 

    # -------------------------------------------------
    # change_ids 初始化（对齐 CARGaussianModel）
    # -------------------------------------------------
    def init_deform(self, screw_init: dict):
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
        if self.initialized_deform:
            return
        device = self._xyz.device
        N = self._xyz.shape[0]
        self._internal_mask = torch.zeros(N, dtype=torch.bool, device=device)

        change_ids = screw_init["ids"]
        self.change_ids = change_ids

        self.initialized_deform = True
        print(f"[Init DeGSModel] change_ids = {self.change_ids}")

    # =======  Getter functions  =======
    @property
    def get_change_ids(self):
        return self.change_ids
    
    @property
    def get_internal_mask(self):
        return self._internal_mask
    
    def capture(self):
        parent_data = super().capture()
        return parent_data + (self.change_ids, self._internal_mask)

    def restore(self, model_args, training_args):
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
        if len(extra_args) == 2:
            self.change_ids = extra_args[0]
            self._internal_mask = extra_args[1]
            # self._internal_log_scale = extra_args[2] <--- 删除这行
            print(f"[Restore] DeformModel restored.")

        else:
            print("[Restore] No internal mask params in ckpt.")
            self._internal_mask = None
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # print(opt_dict)
        self.optimizer.load_state_dict(opt_dict)

    # -------------------------------------------------
    # deform mask
    # -------------------------------------------------
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

        return deform_mask, obj_mask  # Bool[N]

    # -------------------------------------------------
    # 内部高斯添加
    # -------------------------------------------------
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
            new_log_scale_1d = torch.full(
                (num_new, 1), -4.0, device=device, dtype=dtype
            )
            # new_scaling = new_log_scale_1d.repeat(1, 3)
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

            new_xyz_all.append(new_xyz)
            new_fdc_all.append(new_fdc)
            new_frest_all.append(new_frest)
            new_opacity_all.append(new_opacity)
            new_scaling_all.append(new_scaling)
            new_rotation_all.append(new_rotation)
            new_semantic_all.append(new_sem)
            new_obj_id_all.append(new_obj_id)

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

        # -------- 官方增点流程 --------
        self.densification_postfix(
            new_xyz,
            new_fdc,
            new_frest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_semantic,
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
        #         # self._scaling,
        #         self._rotation,
        #         self._semantic_feature,
        #     ]:
        #         p[:n_old].requires_grad_(False)
        #         p[n_old:].requires_grad_(True)
        #     self._scaling.requires_grad_(False)
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
            self._semantic_feature
        ]
        
        # 注册新 hooks
        print(f"[Freeze] Locking gradients for the first {n_old} points.")
        for p in params_to_freeze:
            if p.requires_grad:
                h = p.register_hook(get_zero_grad_hook(n_old))
                self.grad_hooks.append(h) 