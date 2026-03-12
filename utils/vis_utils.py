import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from utils.sh_utils import eval_sh, SH2RGB    # 你提供的文件名假设为 sh_utils.py

# ---------------------
# 普通 PLY 读取
# ---------------------
def load_normal_ply(ply):
    vertex = ply.elements[0].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    # 尝试读取颜色
    rgb = None
    if "red" in vertex.dtype.names:
        rgb = np.stack([
            vertex["red"], vertex["green"], vertex["blue"]
        ], axis=1) / 255.0
    elif "r" in vertex.dtype.names:
        rgb = np.stack([
            vertex["r"], vertex["g"], vertex["b"]
        ], axis=1) / 255.0
    else:
        print("⚠️ No color found, using gray.")
        rgb = np.ones_like(xyz) * 0.5

    return xyz, rgb, None  # features=None


# ---------------------
# Gaussian PLY 读取
# ---------------------
def load_gaussian_ply(ply, sh_degree=3):
    vertex = ply.elements[0].data
    N = len(vertex)

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # --- DC coefficients ---
    f_dc = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=1)  # (N,3)

    # --- Rest coefficients ---
    num_rest = 3 * ((sh_degree + 1)**2 - 1)
    f_rest = np.zeros((N, num_rest), dtype=np.float32)

    for i in range(num_rest):
        f_rest[:, i] = vertex[f"f_rest_{i}"]

    f_rest = f_rest.reshape(N, 3, -1)

    # ----------
    # Convert to torch for SH2RGB
    # ----------
    f_dc_torch = torch.tensor(f_dc, dtype=torch.float32)
    f_rest_torch = torch.tensor(f_rest, dtype=torch.float32)

    features = torch.cat([f_dc_torch.unsqueeze(-1), f_rest_torch], dim=-1)

    # Only DC gives approximate color
    rgb = SH2RGB(features[:, :, 0]).cpu().numpy().clip(0, 1)

    return xyz, rgb

def visualize_gs_pointcloud(pc, color_mode="sh", cam_dir=None):
    """
    用 Open3D 可视化 Gaussian Splatting 中的高斯中心点。
    颜色来源可以是：
    - 球谐函数 SH (color_mode='sh')
    - 对象 ID obj_id (color_mode='obj_id')
    
    Args:
        pc: GaussianModel 对象，含 xyz, sh feature, obj_ids
        color_mode: "sh" 或 "obj_id"
        cam_dir: 观察方向，用于 eval_sh（默认 [0,0,1]）
    """
    device = pc.get_xyz.device

    # --------------------------------------
    # 1. 获取点的位置
    # --------------------------------------
    xyz = pc.get_xyz.detach().cpu().numpy()   # [P, 3]

    # --------------------------------------
    # 2. SH 着色
    # --------------------------------------
    if color_mode == "sh":
        sh_feat = pc.get_features   # [P, 3, (deg+1)^2]
        P, C, F = sh_feat.shape

        # 当前 SH 阶数（例如 deg=3 → F=16）
        deg = int(np.sqrt(F) - 1)

        # 观察方向，默认从正 Z 看
        if cam_dir is None:
            cam_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        cam_dir = cam_dir / cam_dir.norm()

        dirs = cam_dir[None, :].repeat(P, 1)  # [P,3]

        # 计算 SH 颜色 [P,3]
        sh_colors = eval_sh(deg, sh_feat.permute(0, 2, 1), dirs)  # 转成 [..., C, F]
        # 映射回 [0,1]
        rgb = SH2RGB(sh_colors).clamp(0, 1)
        colors = rgb.detach().cpu().numpy()

    # --------------------------------------
    # 3. obj_id 着色
    # --------------------------------------
    elif color_mode == "id":
        assert hasattr(pc, "get_obj_ids"), "pc 中没有 obj_id 数据"

        obj_ids = pc.get_obj_ids.detach().cpu().numpy()  # [P]
        max_id = int(obj_ids.max())

        np.random.seed(42)
        color_map = np.random.rand(max_id + 1, 3)  # [id → RGB]

        colors = color_map[obj_ids]

    else:
        raise ValueError("color_mode must be 'sh' or 'id'")

    # --------------------------------------
    # 4. 构建 Open3D 点云
    # --------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[Visualizer] #Points = {xyz.shape[0]}, ColorMode = {color_mode}")
    o3d.visualization.draw_geometries([pcd])


def vis_change_gs(pc, change_ids, cam_dir=None, highlight_color=[1.0, 0.0, 0.0]):
    """
    使用 Open3D 可视化 GS 点云，并高亮 change_ids 中的物体。

    Args:
        pc: GaussianModel / ScrewGaussianModel
        change_ids: list[int] 需要高亮的 object id
        cam_dir: SH 可视化的观察方向
        highlight_color: 高亮颜色（变化物体的颜色）
    """

    device = pc.get_xyz.device

    # --------------------------------------
    # 1. 获取点的位置
    # --------------------------------------
    xyz = pc.get_xyz.detach().cpu().numpy()   # [P, 3]

    # --------------------------------------
    # 2. 基础颜色：用 SH 或 ID 上色（与 visualize_gs_pointcloud 一致）
    # --------------------------------------
    try:
        # 尝试 SH 着色
        sh_feat = pc.get_features  # [P, C, F]
        P, C, F = sh_feat.shape
        deg = int(np.sqrt(F) - 1)

        if cam_dir is None:
            cam_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        cam_dir = cam_dir / cam_dir.norm()

        dirs = cam_dir[None, :].repeat(P, 1)  # [P,3]

        sh_colors = eval_sh(deg, sh_feat.permute(0, 2, 1), dirs)  
        rgb = SH2RGB(sh_colors).clamp(0, 1)
        colors = rgb.detach().cpu().numpy()

    except Exception:
        # 若没有 SH，则使用 obj_id 随机着色
        assert hasattr(pc, "get_obj_ids"), "pc 中没有 obj_id 数据"
        obj_ids = pc.get_obj_ids.detach().cpu().numpy()
        max_id = int(obj_ids.max())

        np.random.seed(42)
        color_map = np.random.rand(max_id + 1, 3)
        colors = color_map[obj_ids]

    # --------------------------------------
    # 3. 获取每个点所属的 obj_id
    # --------------------------------------
    if hasattr(pc, "get_obj_ids"):
        obj_ids = pc.get_obj_ids.detach().cpu().numpy()
    else:
        raise ValueError("pc.get_obj_ids 不存在，无法高亮变化物体")

    change_ids = set([int(i) for i in change_ids])

    # --------------------------------------
    # 4. 高亮变化物体
    # --------------------------------------
    highlight_color = np.array(highlight_color, dtype=np.float32)
    highlight_mask = np.isin(obj_ids, list(change_ids))

    # 将变化物体点设为 highlight_color
    colors[highlight_mask] = highlight_color

    print(f"[vis_change_gs] #points={xyz.shape[0]}, changed_points={highlight_mask.sum()}, "
          f"change_ids={list(change_ids)}")

    # --------------------------------------
    # 5. Open3D 点云可视化
    # --------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


# ========================
# Visualize axis
# ========================
def align_vector_to_vector(a, b):
    """
    返回一个旋转矩阵，将向量 a 旋转到向量 b
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.999999:
        # 180 度反向，选任意垂直轴旋转
        ax = np.array([1, 0, 0])
        if abs(a[0]) > 0.9: ax = np.array([0, 1, 0])
        v = np.cross(a, ax)
        v = v / np.linalg.norm(v)
        H = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = -np.eye(3) + 2 * np.outer(v, v)
        return R

    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)  # same direction
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R

def vis_screw_axis(axis_point, axis_dir, base_cloud=None,
                   axis_len=0.5, radius=0.01, cone_radius=0.02, cone_height=0.06):
    """
    axis_point: (3,)
    axis_dir:   (3,) 需要是单位向量
    base_cloud: 一个 point cloud (可选)
    axis_len:   轴线的一半长度
    """

    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)

    start = axis_point - axis_dir * axis_len
    end   = axis_point + axis_dir * axis_len

    # ---------- create cylinder ----------
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=2 * axis_len)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color([1, 0, 0])  # red

    # align cylinder with axis_dir
    cylinder_frame = cylinder.get_center()
    T = np.eye(4)
    T[:3, :3] = align_vector_to_vector(np.array([0, 0, 1]), axis_dir)
    T[:3, 3] = start + axis_dir * axis_len  # cylinder center
    cylinder.transform(T)

    # ---------- create cone arrow ----------
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.compute_vertex_normals()
    cone.paint_uniform_color([1, 0.6, 0.1])  # orange

    # align cone direction +Z → axis_dir
    T_cone = np.eye(4)
    T_cone[:3, :3] = align_vector_to_vector(np.array([0, 0, 1]), axis_dir)
    T_cone[:3, 3] = end  # tip of axis
    cone.transform(T_cone)

    geoms = [cylinder, cone]

    if base_cloud is not None:
        geoms = [base_cloud] + geoms

    o3d.visualization.draw_geometries(geoms)

def vis_point_cloud(cloud, color=None):
    """
    cloud: numpy array [N,3]
    color: None or [N,3] (values in 0~1)
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    if color is None:
        # random color
        colors = np.random.rand(cloud.shape[0], 3)
    else:
        colors = color
        if colors.max() > 1.0:
            colors = colors / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def vis_two_point_clouds(cloudA, cloudB,
                         colorA=[0.6, 0.6, 0.6],
                         colorB=[1.0, 0.0, 0.0]):
    """
    cloudA: [N,3] numpy array  (base / canonical)
    cloudB: [M,3] numpy array  (changed / new)
    colorA: RGB for cloudA (default: gray)
    colorB: RGB for cloudB (default: red)
    """

    # --- cloud A (灰色) ---
    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(cloudA)
    colorsA = np.tile(np.array(colorA).reshape(1,3), (cloudA.shape[0], 1))
    pcdA.colors = o3d.utility.Vector3dVector(colorsA)

    # --- cloud B (高亮) ---
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(cloudB)
    colorsB = np.tile(np.array(colorB).reshape(1,3), (cloudB.shape[0], 1))
    pcdB.colors = o3d.utility.Vector3dVector(colorsB)

    # visualize together
    o3d.visualization.draw_geometries([pcdA, pcdB])

def vis_point_cloud_with_changeid(cloud, obj_ids, change_ids):
    """
    cloud: (N, 3) float
    obj_ids: (N,) int
    change_ids: list of object IDs to highlight
    """
    cloud = np.asarray(cloud)
    obj_ids = np.asarray(obj_ids).reshape(-1)
    change_ids = list(change_ids)

    assert cloud.shape[0] == obj_ids.shape[0], "cloud and obj_ids length mismatch"

    # ------------------------
    # Base color = gray
    # ------------------------
    colors = np.ones((cloud.shape[0], 3), dtype=np.float32) * 0.65

    # ------------------------
    # 12 bright colors
    # ------------------------
    vivid_colors = np.array([
        [1.00, 0.10, 0.10],  # vivid red
        [0.10, 0.30, 1.00],  # vivid blue
        [0.00, 0.80, 0.20],  # vivid green
        [1.00, 0.60, 0.00],  # orange
        [0.60, 0.00, 1.00],  # purple
        [0.00, 0.90, 0.90],  # cyan
        [1.00, 1.00, 0.00],  # yellow
        [1.00, 0.20, 0.70],  # pink
        [0.00, 0.70, 0.70],  # teal
        [0.80, 0.40, 0.00],  # brown-orange
        [0.30, 1.00, 0.30],  # bright green
        [1.00, 0.00, 1.00],  # magenta
    ], dtype=np.float32)

    # ------------------------
    # Assign vivid colors to each changed object
    # ------------------------
    for idx, oid in enumerate(change_ids):
        mask = (obj_ids == oid)
        if mask.sum() == 0:
            continue
        colors[mask] = vivid_colors[idx % len(vivid_colors)]

    # ------------------------
    # Build Open3D cloud
    # ------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd],
                                      window_name="Change-Highlighted Point Cloud")

def vis_camera_with_pcd(c2w, cloud, cam_scale=0.2, cloud_color=[0.7,0.7,0.7]):
    """
    c2w: [4,4] camera-to-world extrinsic matrix
    cloud: [N,3] point cloud in world coordinates
    cam_scale: size of the camera axes
    cloud_color: default grey color
    """

    # 1) Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    colors = np.tile(np.array(cloud_color).reshape(1,3), (cloud.shape[0],1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 2) Create axes to represent camera pose
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cam_scale
    )
    camera_frame.transform(c2w)    # put camera in world coordinates

    # 3) Visualize both
    o3d.visualization.draw_geometries([pcd, camera_frame])

def vis_point_cloud_with_id_axis(
    xyz,
    obj_ids,
    change_ids=None,
    axis_point=None,
    axis_dir=None,
    axis_len=0.5, radius=0.01, cone_radius=0.02, cone_height=0.06
):
    """
    xyz: (N,3)
    obj_ids: (N,)
    change_ids: list[int] 需要高亮的物体ID
    axis_point, axis_dir 可选，若无则只显示点云
    """

    xyz = np.asarray(xyz)
    obj_ids = np.asarray(obj_ids)

    # 所有 obj 的颜色
    unique_ids = np.unique(obj_ids)
    cmap = {}
    np.random.seed(42)
    for oid in unique_ids:
        cmap[oid] = np.random.rand(3)*0.5

    # 普通颜色
    colors = np.array([cmap[i] for i in obj_ids])

    # 对 change_ids 增亮颜色
    if change_ids is not None:
        change_ids = set(change_ids)
        highlight_color = np.array([0.1, 1.0, 0.1])   # 亮红
        mask = np.isin(obj_ids, list(change_ids))
        colors[mask] = highlight_color

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 整体几何体列表
    geoms = [pcd]

    # 若给了 axis，则加入轴
    if axis_point is not None and axis_dir is not None:
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)

        start = axis_point - axis_dir * axis_len
        end   = axis_point + axis_dir * axis_len

        # ---------- create cylinder ----------
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=2 * axis_len)
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color([1, 0, 0])  # red

        # align cylinder with axis_dir
        cylinder_frame = cylinder.get_center()
        T = np.eye(4)
        T[:3, :3] = align_vector_to_vector(np.array([0, 0, 1]), axis_dir)
        T[:3, 3] = start + axis_dir * axis_len  # cylinder center
        cylinder.transform(T)

        # ---------- create cone arrow ----------
        cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
        cone.compute_vertex_normals()
        cone.paint_uniform_color([1, 0.6, 0.1])  # orange

        # align cone direction +Z → axis_dir
        T_cone = np.eye(4)
        T_cone[:3, :3] = align_vector_to_vector(np.array([0, 0, 1]), axis_dir)
        T_cone[:3, 3] = end  # tip of axis
        cone.transform(T_cone)

        axis_geoms = [cylinder, cone]

        # axis_geoms = vis_screw_axis(axis_point, axis_dir, base_cloud=None)
        geoms.extend(axis_geoms)

    # 显示
    o3d.visualization.draw_geometries(geoms)

def vis_point_cloud_with_id_axes(
    xyz,
    colors,
    screw_dict,       # 你的 JSON["screw_init"]，包含 axis_point / axis_dir / top_global_ids
    change_ids=None,  # JSON["ids"]
    axis_len=0.5,
    radius=0.01,
    cone_radius=0.02,
    cone_height=0.06,
):
    """
    xyz: (N,3)
    colors: (N,3)
    screw_dict: dict, JSON["screw_init"][oid] = {
        "axis_point": [...],
        "axis_dir": [...],
        "top_global_ids": [...]
    }
    change_ids: list[int], e.g. [36]
    """

    xyz = np.asarray(xyz)
    colors = np.asarray(colors).copy()

    geoms = []

    # ============================================================
    # 1. 高亮 top_global_ids (自动从 screw_dict 中读取)
    # ============================================================
    if change_ids is not None:
        for oid in change_ids:
            oid = str(oid)

            if oid not in screw_dict:
                continue

            top_ids = screw_dict[oid].get("top_global_ids", [])

            if top_ids:
                top_ids = np.asarray(top_ids, dtype=int)
                top_ids = top_ids[(top_ids >= 0) & (top_ids < xyz.shape[0])]
                print(f"[vis] Highligting {len(top_ids)} Gaussian points for object {oid}")
                colors[top_ids] = np.array([1.0, 0.0, 0.0])   # 红色高亮

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geoms.append(pcd)

    # ============================================================
    # 2. 绘制每个 change_id 的 screw axis
    # ============================================================
    if change_ids is not None:
        np.random.seed(0)
        axis_colors = np.random.rand(len(change_ids), 3)

        for i, oid in enumerate(change_ids):
            oid = str(oid)
            if oid not in screw_dict:
                continue

            ax_info = screw_dict[oid]
            axis_point = np.asarray(ax_info["axis_point"], float)
            axis_dir   = np.asarray(ax_info["axis_dir"], float)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)

            # build cylinder
            start = axis_point - axis_dir * axis_len
            end   = axis_point + axis_dir * axis_len

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=2*axis_len)
            cylinder.compute_vertex_normals()
            cylinder.paint_uniform_color(axis_colors[i])

            T = np.eye(4)
            T[:3, :3] = align_vector_to_vector(np.array([0,0,1]), axis_dir)
            T[:3, 3] = start + axis_dir * axis_len
            cylinder.transform(T)

            # cone arrow
            cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
            cone.compute_vertex_normals()
            cone.paint_uniform_color(axis_colors[i] * 0.8)

            T2 = np.eye(4)
            T2[:3, :3] = align_vector_to_vector(np.array([0,0,1]), axis_dir)
            T2[:3, 3] = end
            cone.transform(T2)

            geoms += [cylinder, cone]

    # ============================================================
    # 显示
    # ============================================================
    o3d.visualization.draw_geometries(geoms)

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def align_vector_to_vector(a, b):
    """返回一个旋转矩阵，使得 a → b"""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.999999:
        # 180度，随便选一个垂直轴
        ax = np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])
        v = normalize(np.cross(a, ax))
        H = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        return -np.eye(3) + 2*np.outer(v, v)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return R

def make_cylinder(axis_point, axis_dir, length=0.5, radius=0.01, color=[255,0,0]):
    """
    输出 cylinder 的 vertices 和 colors
    我们把圆柱离散成 20 段
    """
    axis_dir = normalize(axis_dir)
    start = axis_point - axis_dir * length
    end   = axis_point + axis_dir * length

    # 生成局部圆柱（z 方向）
    n = 20
    theta = np.linspace(0, 2*np.pi, n)
    circle = np.stack([np.cos(theta)*radius, np.sin(theta)*radius, np.zeros_like(theta)], axis=1)

    # 顶 + 底
    bottom = circle + np.array([0,0,0])
    top    = circle + np.array([0,0,2*length])

    # 合并
    verts = np.vstack([bottom, top])    # [40,3]

    # 将 cylinder 对齐到 axis
    R = align_vector_to_vector(np.array([0,0,1]), axis_dir)
    verts = verts @ R.T   # rotate
    verts += start        # translate

    cols = np.tile(np.array(color), (verts.shape[0],1))
    return verts, cols


def make_cone(axis_point, axis_dir, height=0.07, radius=0.03, color=[255,128,0]):
    """
    构建 arrow 头部（圆锥）
    """
    axis_dir = normalize(axis_dir)
    tip = axis_point + axis_dir * height*0.8
    base_center = axis_point + axis_dir * (-height*0.2)

    n = 20
    theta = np.linspace(0, 2*np.pi, n)
    circle = np.stack([np.cos(theta)*radius,
                       np.sin(theta)*radius,
                       np.zeros_like(theta)], axis=1)

    # rotate circle
    R = align_vector_to_vector(np.array([0,0,1]), axis_dir)
    circle = circle @ R.T
    circle += base_center

    verts = np.vstack([circle, tip[None]])  # [21,3]
    cols  = np.tile(np.array(color), (verts.shape[0],1))
    return verts, cols

def visualize_pcd_matching(
    xyz0, xyz1,
    idx0, idx1,
    max_lines=2000,
    color0=(0.2, 0.8, 0.2),
    color1=(0.2, 0.2, 0.9),
    match_color=(1.0, 0.0, 0.0),
    point_size=3.0,
):
    assert len(idx0) == len(idx1), "idx0 / idx1 length mismatch"

    xyz0 = np.asarray(xyz0)
    xyz1 = np.asarray(xyz1)

    M = len(idx0)

    # ---------- 下采样匹配对（防止太密） ----------
    if M > max_lines:
        sel = np.random.choice(M, max_lines, replace=False)
        idx0 = idx0[sel]
        idx1 = idx1[sel]
        M = max_lines

    # ---------- 原始点云 ----------
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(xyz0)
    pcd0.colors = o3d.utility.Vector3dVector(
        np.tile(color0, (xyz0.shape[0], 1))
    )

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    pcd1.colors = o3d.utility.Vector3dVector(
        np.tile(color1, (xyz1.shape[0], 1))
    )

    # ---------- 高亮匹配点 ----------
    match_pts0 = xyz0[idx0]
    match_pts1 = xyz1[idx1]

    pcd0_match = o3d.geometry.PointCloud()
    pcd0_match.points = o3d.utility.Vector3dVector(match_pts0)
    pcd0_match.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 1.0, 0.0], (M, 1))
    )

    pcd1_match = o3d.geometry.PointCloud()
    pcd1_match.points = o3d.utility.Vector3dVector(match_pts1)
    pcd1_match.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.6, 0.0], (M, 1))
    )

    # ---------- 连线 ----------
    line_points = []
    line_indices = []
    line_colors = []

    for i in range(M):
        line_points.append(match_pts0[i])
        line_points.append(match_pts1[i])
        line_indices.append([2 * i, 2 * i + 1])
        line_colors.append(match_color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(match_color, (len(line_indices), 1))
    )

    # ---------- 可视化 ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Matching", width=1600, height=900)
    vis.add_geometry(pcd0)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd0_match)
    vis.add_geometry(pcd1_match)
    vis.add_geometry(line_set)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()

def vis_mask_with_top_ids(
    xyz,
    mask,
    global_top_ids,
    base_color=(0.7, 0.7, 0.7),
    top_color=(1.0, 0.1, 0.1),
    max_points=200000,
):
    """
    Args:
        xyz             : (N,3) numpy array, 完整点云
        mask            : (N,) bool, 需要可视化的子集
        global_top_ids  : list / np.ndarray, mask 内需要高亮的全局索引
        base_color      : mask 点的颜色
        top_color       : top 点的颜色
        max_points      : 防止点太多（可选下采样）
    """

    assert xyz.shape[0] == mask.shape[0]

    xyz = np.asarray(xyz)
    mask = np.asarray(mask).astype(bool)
    global_top_ids = np.asarray(global_top_ids, dtype=np.int64)

    # ---------- 只取 mask 内的点 ----------
    idx_mask = np.where(mask)[0]

    if len(idx_mask) == 0:
        print("[vis] Empty mask, nothing to visualize.")
        return

    # 可选下采样（防止 open3d 卡死）
    if len(idx_mask) > max_points:
        idx_mask = np.random.choice(idx_mask, max_points, replace=False)

    pts = xyz[idx_mask]

    # ---------- 颜色初始化 ----------
    colors = np.tile(base_color, (len(idx_mask), 1))

    # ---------- 高亮 top ids ----------
    # global id → mask 内的局部索引
    id2local = {gid: i for i, gid in enumerate(idx_mask)}
    for gid in global_top_ids:
        if gid in id2local:
            colors[id2local[gid]] = top_color

    # ---------- Open3D ----------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Mask points (highlighted top ids)",
        point_show_normal=False,
    )


def visualize_mobility_open3d(
    gaussian,
    show_only_mob=False,
    window_name="Mobility Visualization"
):
    """
    Visualize mobility directly in an Open3D window.

    Args:
        gaussian: CARGaussianModelSC
        show_only_mob: if True, only visualize mob_mask points
        window_name: window title
    """

    # --------------------------------------------------
    # fetch data
    # --------------------------------------------------
    xyz = gaussian.get_xyz.detach().cpu().numpy()           # (N, 3)
    mob = gaussian.get_mobility.squeeze().detach().cpu().numpy()  # (N,)
    mask = gaussian.get_mob_mask.detach().cpu().numpy()     # (N,)

    # --------------------------------------------------
    # color mapping
    # --------------------------------------------------
    colors = np.zeros_like(xyz)

    # non-target objects: gray
    colors[:] = np.array([0.5, 0.5, 0.5])

    # target object: blue -> red
    colors[mask, 0] = mob[mask]          # R
    colors[mask, 1] = 0.0                # G
    colors[mask, 2] = 1.0 - mob[mask]    # B

    if show_only_mob:
        xyz = xyz[mask]
        colors = colors[mask]

    # --------------------------------------------------
    # Open3D point cloud
    # --------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --------------------------------------------------
    # visualization
    # --------------------------------------------------
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=1024,
        height=768,
        point_show_normal=False,
    )