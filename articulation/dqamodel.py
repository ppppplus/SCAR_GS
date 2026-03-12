import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import re
from utils.general_utils import get_expon_lr_func
from articulation.dual_quaternion_utils import (
    quaternion_mul,
    normalize_dualquaternion,
    quaternion_to_matrix,
    dual_quaternion_apply,
    dual_quaternion_inverse,
    quaternion_slerp,
    dual_quaternion_slerp,
    scale_dual_quaternion

)

class DQArtiModel(nn.Module):
    """
    Dual-quaternion articulation module with Automatic Type Discovery.
    
    Phase 1 (Warm-up): Allows general rigid motion (Screw motion) for all joints.
    Phase 2 (Locked): Snaps joints to Pure Revolute or Pure Prismatic based on analysis.
    """

    def __init__(self, num_joints=1, joint_types=None):
        """
        num_joints: K
        joint_types: list of ['s', 'r', 'p']. 
                     If None, defaults to ['r'].
                     NOTE: These types are ignored until `analyze_and_lock_joints` is called,
                     unless you manually set self.types_locked = True immediately.
        """
        super().__init__()
        self.device = "cuda"
        self.num_joints = num_joints
        
        # 初始默认都标记为 'r'，但在 locked=False 时，行为表现为通用关节
        self.joint_types = joint_types or ['r'] * num_joints
        
        # 锁定标志：False=自由探索阶段，True=强制约束阶段
        self.types_locked = False 

        # joint params: [qr(4), t(3)]
        # initialized with identity rotation and zero translation (plus noise)
        joints = torch.zeros(num_joints, 7).to(self.device)
        joints[:, 0] = 1.0 
        joints += torch.randn_like(joints) * 1e-5
        self.joints = nn.Parameter(joints)
        self.register_buffer('joint_obj_ids', torch.full((num_joints,), -1, dtype=torch.long))

        self.optimizer = None
        self.lr_scheduler_args = None

        # buffers for identity
        self.register_buffer('qr_id', torch.tensor([1., 0., 0., 0.]))
        self.register_buffer('qd_id', torch.tensor([0., 0., 0., 0.]))

    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(
            [{"params": self.joints, "lr": 0.0, "name": "dq_joints"}],
            eps=1e-15,
        )
        self.lr_scheduler_args = get_expon_lr_func(
            lr_init=training_args.dq_lr_init,
            lr_final=training_args.dq_lr_init * 0.01,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=training_args.dq_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        if self.optimizer is None or self.lr_scheduler_args is None:
            return
        lr = self.lr_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def init_from_axis_json(self, json_path, angle=math.pi / 4):
        with open(json_path, 'r') as f:
            data = json.load(f)
        ids = data["ids"]
        screw_init = data["screw_init"]
        
        assert len(ids) <= self.num_joints

        for k, id_ in enumerate(ids):
            info = screw_init[str(id_)]
            self.joint_obj_ids[k] = int(id_)
            p = torch.tensor(info["axis_point"], device=self.device)
            d = torch.tensor(info["axis_dir"], device=self.device)
            d = F.normalize(d, dim=0)

            theta = angle
            qr = torch.zeros(4, device=p.device)
            qr[0] = math.cos(theta / 2)
            qr[1:] = d * math.sin(theta / 2)
            qr = F.normalize(qr, dim=0)

            R = quaternion_to_matrix(qr)
            t = p - R @ p

            self.joints[k, :4].copy_(qr)
            self.joints[k, 4:7].copy_(t)
        
        print(f"[DQArticulation] Initialized {len(ids)} joints from axis json.")

    # ==================================================
    # Core Logic: Get Dual Quaternions
    # ==================================================
    # def get_slot_dual_quaternions(self):
    #     qrs, qds = [], []

    #     for k in range(self.num_joints):
    #         # 获取原始参数
    #         joint = self.joints[k]
    #         qr_raw = joint[:4]
    #         t = joint[4:7]
    #         t0 = torch.cat([torch.zeros(1, device=t.device), t])
            
    #         curr_type = self.joint_types[k]

    #         # 静态关节永远保持单位DQ
    #         if curr_type == 's':
    #             qr, qd = self.qr_id, self.qd_id
    #         else:
    #             # 判决逻辑：
    #             # 如果还没锁定(types_locked=False)，或者锁定了但是是旋转关节('r')
    #             # 则允许 qr 从参数中学习（即允许旋转）
    #             allow_rotation = (not self.types_locked) or (curr_type == 'r')

    #             if allow_rotation:
    #                 qr = F.normalize(qr_raw, dim=-1)
    #                 # 一般刚体变换/旋转关节：qd 包含由 t 和 R 共同决定的平移部
    #                 qd = 0.5 * quaternion_mul(t0, qr)
    #             else:
    #                 # 锁定且为平移关节 ('p')
    #                 # 强制旋转为 Identity，只允许 t 生效
    #                 qr = self.qr_id
    #                 # 纯平移：qd = 0.5 * t * 1
    #                 qd = 0.5 * quaternion_mul(t0, qr)

    #         qrs.append(qr)
    #         qds.append(qd)

    #     qrs = torch.stack(qrs)
    #     qds = torch.stack(qds)
    #     return normalize_dualquaternion(qrs, qds)
    def get_slot_dual_quaternions(self, progress=1.0):
        """
        获取双四元数。
        
        Args:
            progress: float in [0, 1] - 控制旋转和平移的程度
        """
        qrs, qds = [], []

        for k in range(self.num_joints):
            joint = self.joints[k]
            qr_raw = joint[:4]
            t = joint[4:7]
            # print(t)
            t0 = torch.cat([torch.zeros(1, device=t.device), t])
            curr_type = self.joint_types[k]
            qr = F.normalize(qr_raw, dim=-1)
            if curr_type == 's':
                qr, qd = self.qr_id, self.qd_id
            else:
                allow_rotation = (not self.types_locked) or (curr_type == 'r')
                
                if allow_rotation:
                    # 应用 progress 插值
                    # if progress < 1.0:
                    #     qr = quaternion_slerp(self.qr_id, qr, progress)
                    
                    # 根据（可能插值后的）旋转计算 qd
                    # t_scaled = t * progress
                    # qd = 0.5 * quaternion_mul(t, qr)
                    qd = 0.5 * quaternion_mul(t0, qr)
                    # dq_target = (qr, qd_target)
                    # dq_id = (self.qr_id, self.qd_id) # (1,0,0,0) 和 (0,0,0,0)
                    
                    # 2. 直接在双四元数空间插值
                    # qr, qd = dual_quaternion_slerp(dq_id, dq_target, progress)
                    qr, qd = scale_dual_quaternion(qr, qd, progress)
                else:
                    # 锁定且为平移关节 ('p')
                    qr = self.qr_id
                    # 根据 progress 缩放平移向量
                    t_scaled = t * progress
                    # print(t_scaled)
                    t0_scaled = torch.cat([torch.zeros(1, device=t_scaled.device), t_scaled])
                    # print(t0_scaled, qr)
                    qd = 0.5 * quaternion_mul(t0_scaled, qr)

            qrs.append(qr)
            qds.append(qd)

        qrs = torch.stack(qrs)
        qds = torch.stack(qds)
        return normalize_dualquaternion(qrs, qds)

    # ==================================================
    # 【NEW】Type Discovery and Locking
    # ==================================================
    @torch.no_grad()
    def analyze_and_lock_joints(self, rotation_threshold_deg=1.0):
        """
        Call this method after N warm-up steps.
        It classifies joints as 'p' or 'r' based on the magnitude of rotation learned so far.
        """
        if self.types_locked:
            return

        print(f"\n[DQArticulation] Analyzing {self.num_joints} joints for locking (Thres: {rotation_threshold_deg} deg)...")
        new_types = []

        for k in range(self.num_joints):
            # Static joints remain static
            if self.joint_types[k] == 's':
                new_types.append('s')
                continue

            # Calculate rotation angle: theta = 2 * acos(w)
            qr = F.normalize(self.joints[k, :4], dim=0)
            w =  torch.clamp(torch.abs(qr[0]), 0.0, 1.0)
            angle_deg = math.degrees(2 * math.acos(w))

            # Classification
            if angle_deg < rotation_threshold_deg:
                print(f"  Joint {k}: Rot {angle_deg:.4f}° -> Classified as PRISMATIC ('p')")
                new_types.append('p')
                # 【重要】Reset rotation params to identity to remove artifacts
                self.joints[k, :4].copy_(self.qr_id)
            else:
                print(f"  Joint {k}: Rot {angle_deg:.4f}° -> Classified as REVOLUTE ('r')")
                new_types.append('r')

        self.joint_types = new_types
        self.types_locked = True
        print(f"[DQArticulation] Locking enabled. New types: {self.joint_types}\n")

    def blend_dual_quaternions(self, slot_qr, slot_qd, weights):
        qr = torch.einsum('nk,kc->nc', weights, slot_qr)
        qd = torch.einsum('nk,kc->nc', weights, slot_qd)
        return normalize_dualquaternion(qr, qd)

    def forward(self, x, obj_ids=None, state=1, progress=1.0):
        """
        x: [N,3]
        obj_ids: [N] 或 [N,1] - 每个高斯点所属的物体 ID
        state: 0 (inverse) or 1 (forward)
        progress: float in [0, 1] - 控制旋转和平移的程度
        """
        # 直接调用 get_slot_dual_quaternions，传入 progress 参数
        # 该函数内部已经处理了插值
        slot_qr, slot_qd = self.get_slot_dual_quaternions(progress=progress)

        if state == 0:
            slot_qr, slot_qd = dual_quaternion_inverse((slot_qr, slot_qd))

        N = x.shape[0]
        point_qr = self.qr_id.view(1, 4).repeat(N, 1)
        point_qd = self.qd_id.view(1, 4).repeat(N, 1)

        if obj_ids is not None:
            if obj_ids.dim() > 1:
                obj_ids = obj_ids.squeeze(-1)
            
            for k in range(self.num_joints):
                target_id = self.joint_obj_ids[k]
                if target_id < 0:
                    continue
                
                mask = (obj_ids == target_id)
                if mask.any():
                    point_qr[mask] = slot_qr[k]
                    point_qd[mask] = slot_qd[k]
                    xt = dual_quaternion_apply((point_qr, point_qd), x)
        else:
            xt = dual_quaternion_apply((slot_qr, slot_qd), x)
            point_qr = slot_qr[0:1].repeat(N, 1) if slot_qr.dim() > 1 else slot_qr.repeat(N, 1)

        return {
            "xt": xt,
            "d_xyz": xt - x,
            "point_qr": point_qr,
        }

    def capture(self):
        return {
            "joints": self.joints.detach().cpu(),
            "joint_obj_ids": self.joint_obj_ids.detach().cpu(), 
            "joint_types": self.joint_types, # Save types too
            "types_locked": self.types_locked
        }

    def restore(self, weights):
        device = self.device or self.joints.device
        
        if "joint_types" in weights:
            self.joint_types = weights["joint_types"]
        if "types_locked" in weights:
            self.types_locked = weights["types_locked"]
        if "joint_obj_ids" in weights:
            self.joint_obj_ids.copy_(weights["joint_obj_ids"].to(device))

        joints = weights["joints"].to(device)
        assert joints.shape == self.joints.shape
        with torch.no_grad():
            self.joints.copy_(joints)
    
    @torch.no_grad()
    def init_from_articulation_json(self, json_path):
        """
        从导出的 articulation_params.json 初始化铰链模型。
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 获取检查点数据部分
        ckpt_data = data.get("checkpoints", data)
        
        # 按名称排序以确保关节索引的一致性 (checkpoint_0, checkpoint_1...)
        ckpt_names = sorted(ckpt_data.keys(), key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else x)
        
        num_to_init = len(ckpt_names)
        assert num_to_init <= self.num_joints, f"JSON中有 {num_to_init} 个关节，但模型只有 {self.num_joints} 个"

        new_types = list(self.joint_types)

        for k, name in enumerate(ckpt_names):
            info = ckpt_data[name]
            obj_id = info.get("obj_id", -1)
            self.joint_obj_ids[k] = int(obj_id)
            # 1. 提取基础参数
            p = torch.tensor(info["articulation_pivot"], device=self.device, dtype=torch.float32)
            d = torch.tensor(info["articulation_axis"], device=self.device, dtype=torch.float32)
            d = F.normalize(d, dim=0)
            
            theta = info.get("articulation_angle", 0.0)
            dist = info.get("articulation_dist", 0.0)
            
            # 2. 确定关节类型 (1: Revolute, 2: Prismatic)
            arti_type = info.get("articulation_articulation_type", 1)
            if isinstance(arti_type, list): arti_type = arti_type[0]
            
            if arti_type == 1:  # 旋转关节 (Revolute)
                new_types[k] = 'r'
                # 计算旋转四元数 qr
                qr = torch.zeros(4, device=self.device)
                qr[0] = math.cos(theta / 2)
                qr[1:] = d * math.sin(theta / 2)
                qr = F.normalize(qr, dim=0)
                
                # 计算平移部 t，使得旋转中心 p 保持不动: t = p - R@p
                R = quaternion_to_matrix(qr)
                t = p - R @ p
                
            elif arti_type == 2:  # 平移关节 (Prismatic)
                new_types[k] = 'p'
                # 旋转部为单位阵
                qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                # 平移部为沿轴向的位移: t = dist * d
                t = dist * d
            
            # 3. 写入参数
            self.joints[k, :4].copy_(qr)
            self.joints[k, 4:7].copy_(t)
            
            print(f"  Joint {k} ({name}): Type={new_types[k]}, Pivot={p.tolist()}, Axis={d.tolist()}")

        self.joint_types = new_types
        # 如果你想立即锁定类型，可以取消下面注释
        self.types_locked = True
        
        print(f"[DQArticulation] 成功从 JSON 初始化了 {num_to_init} 个关节。")

    @torch.no_grad()
    def init_from_articulations(self, articulations):
        """
        从导出的 ckpt的articulations 初始化铰链模型。
        """
        new_types = list(self.joint_types)
        for k, item in enumerate(articulations):
            j_id = item['id']
            self.joint_obj_ids[k] = int(j_id)
            pivot = item['pivot'].to(self.device)
            axis = item['axis'].to(self.device)
            axis = F.normalize(axis, dim=0)
            arti_type = item.get("articulation_type", 1)
            theta = item.get("angle", 0.0)
            dist = item.get("dist", 0.0)

            if arti_type == 1:  # 旋转关节 (Revolute)
                new_types[k] = 'r'
                # 计算旋转四元数 qr
                qr = torch.zeros(4, device=self.device)
                qr[0] = math.cos(theta / 2)
                qr[1:] = axis * math.sin(theta / 2)
                qr = F.normalize(qr, dim=0)
                
                # 计算平移部 t，使得旋转中心 p 保持不动: t = p - R@p
                R = quaternion_to_matrix(qr)
                t = pivot - R @ pivot
                
            elif arti_type == 2:  # 平移关节 (Prismatic)
                new_types[k] = 'p'
                # 旋转部为单位阵
                qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                # 平移部为沿轴向的位移: t = dist * d
                t = dist * axis
            
            # 3. 写入参数
            self.joints[k, :4].copy_(qr)
            self.joints[k, 4:7].copy_(t)
            
            print(f"  Joint {k}: Type={new_types[k]}, Pivot={pivot.tolist()}, Axis={axis.tolist()}")

        self.joint_types = new_types
        # 如果你想立即锁定类型，可以取消下面注释
        self.types_locked = True
        
        print(f"[DQArticulation] 成功从 JSON 初始化了 {len(new_types)} 个关节。")
