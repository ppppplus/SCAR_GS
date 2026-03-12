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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs(network_output - gt).mean()

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(network_output.device).float()
    if mask.shape[0] != network_output.shape[0]:
        mask = mask.expand_as(network_output)

    masked_diff = torch.abs(network_output - gt) * mask
    valid_elements = mask.sum()

    if valid_elements > 0:
        return masked_diff.sum() / (valid_elements + 1e-6)
    else:
        return torch.zeros(1, device=network_output.device)


def log_l1_depth_loss(pred_depth, gt_depth, mask):
    valid_pixel = mask.sum()
    if valid_pixel == 0:
        return torch.zeros(1, device=pred_depth.device)

    diff = torch.log(1 + torch.abs(pred_depth - gt_depth))
    return (diff * mask).sum() / valid_pixel

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True, val_range=None):
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    (_, channel, height, width) = img1.size()

    if val_range is None:
        L = 1 if torch.max(img1) <= 1 else 255
    else:
        L = val_range

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.to(img1.device).float()
        valid = mask.sum()
        return (ssim_map * mask).sum() / valid if valid > 0 else torch.zeros(1, device=img1.device)

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def arap_loss(new_xyz, neighbor_indices, neighbor_dist, neighbor_weight):
    neighbor_indices = torch.tensor(neighbor_indices, requires_grad=False).long()
    neighbor_pts = new_xyz[neighbor_indices]
    curr_offset = neighbor_pts - new_xyz[:, None]
    curr_offset_mag = torch.sqrt((curr_offset.pow(2)).sum(-1) + 1e-20)
    return torch.sqrt(
        (curr_offset_mag - neighbor_dist).pow(2) * neighbor_weight + 1e-20
    ).mean()
