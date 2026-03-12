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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render, render_edit 
# from gaussian_renderer import gsplat_render as render, render_edit
from gaussian_renderer import render

import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from scene import Scene, GaussianModel
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
# from utils.clip_utils import CLIPEditor
import yaml
# from models.networks import CNN_decoder, MLP_encoder


def feature_visualize_saving(feature):
    assert feature.ndim == 3 and feature.shape[0] == 3, "[3, H, W]"
    min_val = feature.amin(dim=(1, 2), keepdim=True)
    max_val = feature.amax(dim=(1, 2), keepdim=True)
    feature_norm = (feature - min_val) / (max_val - min_val + 1e-5)
    vis_feature = feature_norm.permute(1, 2, 0).clamp(0, 1).cpu()

    return vis_feature

def render_set(model_path, name, iteration, views, gaussians, pipeline, optim, background, decode):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
    
    # if decode:
    #     gt_feature_map = views[0].semantic_feature.cuda()
    #     feature_out_dim = gt_feature_map.shape[0]
    #     feature_in_dim = int(feature_out_dim/2)
    #     cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
    #     cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
    print("mkdirs:", render_path, gts_path, feature_map_path, gt_feature_map_path, saved_feature_path, depth_path)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(gt_feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True) ###

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, optim) 
        # dict_keys(['render', 'semantic_feature', 'viewspace_points', 'visibility_filter', 'radii', 'depth', 'alpha', 'bg_color'])
        gt = view.original_image[0:3, :, :]
        gt_feature_map = view.semantic_feature.cuda() 
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        ### depth ###
        depth = render_pkg["depth"]
        scale_nor = depth.max().item()
        depth_nor = depth / scale_nor
        depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        colormap = plt.get_cmap('jet')
        depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        depth_colored_rgb = depth_colored[:, :, :3]
        depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
        depth_image.save(output_path)
        ##############

        # visualize feature map
        feature_map = render_pkg["semantic_feature"]
        # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        # if decode:
        #     feature_map = cnn_decoder(feature_map)

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
        Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        # feature_map = feature_map.cpu().numpy().astype(np.float16)
        # torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)


def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, optim: OptimizationParams , skip_train : bool, skip_test : bool): 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, optim, background, False)

        if not skip_test:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, optim, background, False)

        # if novel_view:
        #     render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
        #                     edit_config, dataset.speedup, multi_interpolate, num_views)

        # if video:
        #     render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config)

        # if novel_video:
        #     render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--edit_config', default="no editing", type=str)

    args = get_combined_args(parser)
    args.dataset_path = os.path.join(args.dataset_path, args.scene_name)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), optim.extract(args), args.skip_train, args.skip_test) ###