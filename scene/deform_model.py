"""this file is partly copied from the work *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction* CVPR 2024

We use the results as the initial guess, while we only output the dx and dr of each gs, no ds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.time_utils import DeformNetwork, PointNet
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.W = 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.W)
        self.gaussian_warp = nn.Linear(self.W, 3)
        self.gaussian_rotation = nn.Linear(self.W, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, gaussians, mask=None):
        x = gaussians.get_xyz.detach()  # Nx3
        if mask is not None:
            x = x[mask]
        # h = x.view(1, -1, 3)
        h = x.unsqueeze(0)
        B, N, C = h.shape
        h = torch.permute(h, dims=(0, 2, 1))

        # h = h.view(B, C, N)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        # h = h.transpose(1, 2).contiguous()

        h = F.max_pool1d(h, kernel_size=1).squeeze(2)
        h = torch.permute(h, dims=(0, 2, 1))
        h = h.squeeze(0)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        d_xyz = self.gaussian_warp(h)
        d_rotation = self.gaussian_rotation(h)

        # return (
        #     x + d_xyz,
        #     gaussians.get_rotation + d_rotation,
        #     (d_xyz, d_rotation),
        # )
        return d_xyz, d_rotation


class DeformGS:
    def __init__(self, training_args):
        # self.deform = DeformNetwork(is_blender=is_blender).cuda()
        self.deform = PointNet().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.train_setting(training_args)

    def step(self, gaussians, mask):
        """
        返回完整 N 个 gaussians 的 new_xyz / new_rot
        """
        device = gaussians.get_xyz.device

        # 1. 原始值
        xyz_full = gaussians.get_xyz
        rot_full = gaussians.get_rotation

        # 2. 预测 deform 点的 delta
        d_xyz, d_rot = self.deform(gaussians, mask)

        # 3. clone + scatter
        new_xyz = xyz_full.clone()
        new_rot = rot_full.clone()

        new_xyz[mask] = xyz_full[mask] + d_xyz
        new_rot[mask] = rot_full[mask] + d_rot
        
        return new_xyz, new_rot, (d_xyz, d_rot)
        
    def train_setting(self, training_args):
        l = [
            {
                "params": list(self.deform.parameters()),
                "lr": training_args["position_lr_init"] * self.spatial_lr_scale,
                "name": "deform",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args["position_lr_init"] * self.spatial_lr_scale,
            lr_final=training_args["position_lr_final"],
            lr_delay_mult=training_args["position_lr_delay_mult"],
            max_steps=training_args["deform_lr_max_steps"],
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(
            model_path, "deform/iteration_{}".format(iteration)
        )
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            self.deform.state_dict(), os.path.join(out_weights_path, "deform.pth")
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "deform/iteration_{}/deform.pth".format(loaded_iter)
        )
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
