import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cargs_model import CARGaussianModel
import math
from utils.sh_utils import eval_sh
# from utils.dual_quaternion import quaternion_mul, quaternion_to_matrix, matrix_to_quaternion
# import seaborn as sns
import numpy as np
import math
import torch.nn.functional as F
from utils.arti_utils import exp_se3
from articulation.dual_quaternion_utils import quaternion_mul

def _chk(name, x):
    assert x.is_cuda, f"{name} not cuda"
    assert x.is_contiguous(), f"{name} not contiguous, stride={x.stride()}"
    assert x.dtype in (torch.float16, torch.float32), f"{name} bad dtype {x.dtype}"
    assert torch.isfinite(x).all(), f"{name} has NaN/Inf"
    
def render_gs(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, opt, scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None, override=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    base_xyz = pc.get_xyz if override is None else override["xyz"]

    screenspace_points = torch.zeros_like(
        base_xyz,
        dtype=base_xyz.dtype,
        requires_grad=True,
        device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=opt.include_feature,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if override is None:
        xyz = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        sh_features = pc.get_features
        semantic_feature_precomp = pc.get_semantic_feature
    else:
        xyz = override["xyz"]
        opacity = override["opacity"]
        scales = override["scales"]
        rotations = override["rotations"]
        sh_features = override["features"]
        semantic_feature_precomp = override["semantic_feature"]

    means3D = xyz
    means2D = screenspace_points 
    # if scale_const is not None:
    #     opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(d_rot, scaling_modifier)
    # else:
    #     rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = sh_features
        colors_precomp = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # if opt.include_feature:
    # semantic_feature_precomp = pc.get_semantic_feature
    
    # Reshape semantic feature to match CUDA expectations: [P, 1, 3] -> [P, 3]
    # semantic_feature_precomp = semantic_feature_precomp.squeeze(1)  # Remove the middle dimension
    # semantic_feature_precomp = semantic_feature_precomp/ (semantic_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)

    # --- enforce expected shapes ---
    means3D = means3D.contiguous()
    means2D = means2D.contiguous()
    opacity = opacity.contiguous()
    scales = scales.contiguous()
    rotations = rotations.contiguous()
   
    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        semantic_feature_precomp = semantic_feature_precomp[vis_mask] if semantic_feature_precomp is not None else None
        
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None

    rendered_image, semantic_feature, radii, depth, alpha = rasterizer(
        means3D=means3D,    # [N,3]
        means2D=means2D,    # [N,3]
        opacities=opacity,  # [N,1]
        shs=shs,            # [N,16,3]
        colors_precomp=colors_precomp,  # None
        semantic_feature_precomp=semantic_feature_precomp,  # [N,3]
        scales=scales,  # [N,3]
        rotations=rotations,    # [N,4]
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "semantic_feature": semantic_feature,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}

def render_gs_with_mobility(
    viewpoint_camera,
    gauss_cur,          # gaussian of current state
    gauss_oth,          # gaussian of other state
    mob_mask_oth,       # BoolTensor
    pipe,
    bg_color,
    opt,
):
    """
    Cross-Static Rendering with mobility (Stage 2(b)).

    Returns:
        render_gs output dict
    """
    m_oth_full = gauss_oth.get_mobility.squeeze()
    # --------------------------------------------------
    # static part
    #   - environment (current state, ~mob)
    #   - static part of other state's mob object
    # --------------------------------------------------
    xyz_tgtcs = torch.cat([
        gauss_cur.get_xyz,
        gauss_oth.get_xyz[mob_mask_oth],
    ], dim=0)

    opacity_tgtcs = torch.cat([
        gauss_cur.get_opacity,
        gauss_oth.get_opacity[mob_mask_oth] * (1.0 - m_oth_full[mob_mask_oth]).unsqueeze(-1),
    ], dim=0)

    scales_tgtcs = torch.cat([
        gauss_cur.get_scaling,
        gauss_oth.get_scaling[mob_mask_oth],
    ], dim=0)

    rotations_tgtcs = torch.cat([
        gauss_cur.get_rotation,
        gauss_oth.get_rotation[mob_mask_oth],
    ], dim=0)

    features_tgtcs = torch.cat([
        gauss_cur.get_features,
        gauss_oth.get_features[mob_mask_oth],
    ], dim=0)

    semantic_tgtcs = torch.cat([
        gauss_cur.get_semantic_feature,
        gauss_oth.get_semantic_feature[mob_mask_oth],
    ], dim=0)

    override = {
        "xyz": xyz_tgtcs,
        "opacity": opacity_tgtcs,   # [N,3]
        "scales": scales_tgtcs, # [N,1]
        "rotations": rotations_tgtcs,   # [N,4]
        "features": features_tgtcs, # [N, 16, 3]
        "semantic_feature": semantic_tgtcs, # [N,3]
    }
    # --------------------------------------------------
    # render
    # --------------------------------------------------
    return render_gs(
        viewpoint_camera=viewpoint_camera,
        pc=gauss_cur,      # pc still passed, but overridden
        pipe=pipe,
        bg_color=bg_color,
        opt=opt,
        override=override,
    )

def render_gs_with_dqa(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    opt,
    dqamodel,
    mask,
    scaling_modifier=1.0,
    random_bg_color=False,
    scale_const=None,
    vis_mask=None,
):
    """
    Render GS with mobility-aware dual-quaternion articulation
    by constructing an override and calling render_gs.
    """

    # ------------------------------------------------------------
    # 0. 基础 Gaussian 参数
    # ------------------------------------------------------------
    xyz = pc.get_xyz                     # [N,3]
    opacity = pc.get_opacity             # [N,1]
    scales = pc.get_scaling              # [N,3]
    rotations = pc.get_rotation          # [N,4]
    features = pc.get_features           # [N,*,3]
    semantic = pc.get_semantic_feature   # [N,3] or None

    mobility = pc.get_mobility.squeeze() # [N], in [0,1]

    if mask is None:
        raise ValueError("mask must be provided for render_gs_dqa")

    mask = mask.bool()
    inv_mask = ~mask

    # ------------------------------------------------------------
    # 1. 静态高斯（mask 外）
    # ------------------------------------------------------------
    xyz_s   = xyz[inv_mask]
    op_s    = opacity[inv_mask]
    sc_s    = scales[inv_mask]
    rot_s   = rotations[inv_mask]
    feat_s  = features[inv_mask]
    sem_s   = semantic[inv_mask] if semantic is not None else None

    # ------------------------------------------------------------
    # 2. mask 内高斯：复制两份
    # ------------------------------------------------------------
    xyz_m  = xyz[mask]
    op_m   = opacity[mask]
    sc_m   = scales[mask]
    rot_m  = rotations[mask]
    feat_m = features[mask]
    sem_m  = semantic[mask] if semantic is not None else None
    mob_m  = mobility[mask].unsqueeze(-1)   # [M,1]

    # ------------------------------------------------------------
    # 2.1 准静态（不做 dq 变换）
    # ------------------------------------------------------------
    xyz_qs = xyz_m
    op_qs  = op_m * (1.0 - mob_m)

    # ------------------------------------------------------------
    # 2.2 准动态（dq 变换）
    # ------------------------------------------------------------
    dqa_out = dqamodel(xyz_m, state=1)
    xyz_qd  = dqa_out["xt"]
    op_qd   = op_m * mob_m

    # ------------------------------------------------------------
    # 3. 合并三类高斯
    # ------------------------------------------------------------
    xyz_all = torch.cat([xyz_s, xyz_qs, xyz_qd], dim=0)
    op_all  = torch.cat([op_s,  op_qs,  op_qd],  dim=0)
    sc_all  = torch.cat([sc_s,  sc_m,   sc_m],   dim=0)
    rot_all = torch.cat([rot_s, rot_m,  rot_m],  dim=0)
    feat_all = torch.cat([feat_s, feat_m, feat_m], dim=0)

    if semantic is not None:
        sem_all = torch.cat([sem_s, sem_m, sem_m], dim=0)
    else:
        sem_all = None

    # ------------------------------------------------------------
    # 4. 构造 override
    # ------------------------------------------------------------
    override = {
        "xyz": xyz_all,
        "opacity": op_all,
        "scales": sc_all,
        "rotations": rot_all,
        "features": feat_all,
        "semantic_feature": sem_all,
    }

    # ------------------------------------------------------------
    # 5. 直接调用原 render_gs
    # ------------------------------------------------------------
    return render_gs(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        opt=opt,
        scaling_modifier=scaling_modifier,
        random_bg_color=random_bg_color,
        scale_const=scale_const,
        vis_mask=vis_mask,
        override=override,
    ), torch.cat([xyz_qs, xyz_qd], dim=0), torch.cat([op_qs,  op_qd],  dim=0)

def render_gs_with_dqa_all(
    viewpoint_cam,
    pc,
    pipe,
    opt,
    bg_color : torch.Tensor,
    dqa_model,
    scaling_modifier = 1.0,
    mobility_threshold = 0.5,
    progress = 1.0,
    device = "cuda"
):
    """
    基于 Mobility 筛选动静态点并进行铰链变换渲染。
    逻辑：
    1. 筛选 mobility > threshold 的动态点。
    2. 仅将动态点的 means 和 obj_ids 送入 dqa_model 变换。
    3. 将变换后的动态点与原始静态点拼接。
    4. 送入渲染器。
    """
    # 1. 获取所有基础属性
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    semantic = pc.get_semantic_feature
    obj_ids = pc.get_obj_ids      # [N, 1]
    mobility = pc.get_mobility    # [N, 1]
    
    # 2. 筛选动静态点掩码
    # move_mask: [N], True 代表动态点
    move_mask = (mobility > mobility_threshold).squeeze(-1)
    
    # 3. 提取动态点并执行变换
    # 初始化最终的位置和旋转（默认克隆原始值，即静态点保持不变）
    means3D_final = means3D.clone()
    rotations_final = rotations.clone()
    
    if move_mask.any():
        means3D_dyn = means3D[move_mask]
        obj_ids_dyn = obj_ids[move_mask]
        
        # 仅对动态点调用 dqa_model
        dqa_outputs = dqa_model(means3D_dyn, obj_ids=obj_ids_dyn, progress=progress)
        
        # 更新动态点的位置
        means3D_final[move_mask] = dqa_outputs["xt"]
        
        # 处理动态点的旋转叠加 (如果 DQA 输出旋转部 point_qr)
        if "point_qr" in dqa_outputs:
            
            point_qr_dyn = dqa_outputs["point_qr"] # [N_dyn, 4]
            rotations_dyn = rotations[move_mask]   # [N_dyn, 4]
            
            # 旋转叠加: q_new = q_delta * q_old
            rotations_dyn_transformed = quaternion_mul(point_qr_dyn, rotations_dyn)
            rotations_final[move_mask] = F.normalize(rotations_dyn_transformed, dim=-1)
    override = {
        "xyz": means3D_final,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations_final,
        "features": shs,
        "semantic_feature": semantic,
    }
    return render_gs(
        viewpoint_camera=viewpoint_cam,
        pc=pc,
        pipe=pipe,
        opt=opt,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        override = override,
    )

def render_gs_with_deform(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, opt, new_xyz, new_rots, scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=opt.include_feature,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if new_xyz is not None:
        xyz = new_xyz.clone()
    else:
        xyz = pc.get_xyz

    if new_rots is not None:
        rotations = new_rots.clone()
    else:
        rotations = pc.get_rotation

    opacity = pc.get_opacity
    scales = pc.get_scaling
    sh_features = pc.get_features

    means3D = xyz
    means2D = screenspace_points 
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(d_rot, scaling_modifier)
    # else:
    #     rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = sh_features
        colors_precomp = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # if opt.include_feature:
    semantic_feature_precomp = pc.get_semantic_feature
    
    # Reshape semantic feature to match CUDA expectations: [P, 1, 3] -> [P, 3]
    # semantic_feature_precomp = semantic_feature_precomp.squeeze(1)  # Remove the middle dimension
    # semantic_feature_precomp = semantic_feature_precomp/ (semantic_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)

    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        semantic_feature_precomp = semantic_feature_precomp[vis_mask] if semantic_feature_precomp is not None else None
        
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None

    rendered_image, semantic_feature, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature_precomp=semantic_feature_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "semantic_feature": semantic_feature,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}

def render_gs_with_objid(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, opt, scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=opt.include_feature,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    sh_features = pc.get_features

    means3D = xyz
    means2D = screenspace_points 
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(d_rot, scaling_modifier)
    # else:
    #     rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = sh_features
        colors_precomp = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # if opt.include_feature:
    # semantic_feature_precomp = pc.get_semantic_feature
    semantic_feature_precomp = pc.get_obj_probs #[N,num_objects]
    
    # Reshape semantic feature to match CUDA expectations: [P, 1, 3] -> [P, 3]
    # semantic_feature_precomp = semantic_feature_precomp.squeeze(1)  # Remove the middle dimension
    # semantic_feature_precomp = semantic_feature_precomp/ (semantic_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)

    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        semantic_feature_precomp = semantic_feature_precomp[vis_mask] if semantic_feature_precomp is not None else None
        
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None

    rendered_image, semantic_feature, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature_precomp=semantic_feature_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "semantic_feature": semantic_feature,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}

def render_by_dict(viewpoint_camera, pc, bg_color: torch.Tensor, include_feature=True, scaling_modifier=1.0, random_bg_color=False):
    xyz = pc["xyz"]
    opacity = pc["opacity"]
    scales = pc["scales"]
    rotations = pc["rotations"]
    sh_features = pc["features"]
    semantic_feature_precomp = pc["semantic_feature"]
    active_sh_degree = pc["active_sh_degree"]

    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=include_feature,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points

    cov3D_precomp = None
    shs = sh_features
    colors_precomp = None

    # if vis_mask is not None:
    #     means3D = means3D[vis_mask]
    #     means2D = means2D[vis_mask]
    #     shs = shs[vis_mask] if shs is not None else None
    #     colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
    #     semantic_feature_precomp = semantic_feature_precomp[vis_mask] if semantic_feature_precomp is not None else None
    #     opacity = opacity[vis_mask]
    #     scales = scales[vis_mask]
    #     rotations = rotations[vis_mask]
    #     cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None
    rendered_image, semantic_feature, radii, depth, alpha = rasterizer(
        means3D=means3D,    # [N,3]
        means2D=means2D,    # [N,3]
        opacities=opacity,  # [N,1]
        shs=shs,            # [N,1,3]
        colors_precomp=colors_precomp,
        semantic_feature_precomp=semantic_feature_precomp,  # [N,3,1]
        scales=scales,      # [N,3] 
        rotations=rotations,# [N,4]
        cov3D_precomp=cov3D_precomp
    )

    return {
        "render": rendered_image,
        "semantic_feature": semantic_feature,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": alpha,
        "bg_color": bg
    }
