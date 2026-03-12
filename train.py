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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.render import render_gs
from gaussian_renderer import network_gui
import sys, os
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state, get_linear_noise_func
import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from utils.metrics import *
from utils.log_utils import prepare_output_and_logger
from utils.net_utils import render_net_image
# from pytorch_lightning.loggers import TensorBoardLogger
# from torchviz import make_dot
import yaml
from argparse import Namespace
import argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Trainer:
    def __init__(self, args, dataset, opt, pipe):
        self.args = args
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe

        self.tb_writer = prepare_output_and_logger(dataset)

        self.gaussian = GaussianModel(dataset.sh_degree)

        self.scene = Scene(dataset, self.gaussian, load_iteration=None, shuffle=True)
        
        # if args.init_from_pcd:
        #     print('Init Gaussians with pcd from depth.')
        #     self.gaussian.create_from_pcd(fetchPly(f'{args.source_path}/point_cloud.ply'))
        
        self.gaussian.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        
        self.checkpoint_iterations = args.checkpoint_iterations
        self.start_checkpoint = args.start_checkpoint
        self.save_iterations = args.save_iterations

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.metric_depth_loss_weight = opt.metric_depth_loss_weight
        self.metric_sem_loss_weight = opt.metric_sem_loss_weight

        # self.reg_weight = self.args.opacity_reg_weight

        if self.start_checkpoint:
            checkpoint_path = self.scene.model_path + "/chkpnt" + str(self.start_checkpoint) + ".pth"
            (model_params, self.iteration) = torch.load(checkpoint_path)
            self.gaussian.restore(model_params, self.opt)
            self.iteration += 1
        else:
            self.iteration = 1
        self.progress_bar = tqdm.tqdm(range(self.iteration, opt.iterations+1), desc="Training progress")

    # no gui mode
    def train(self, iters=5000):
        while self.iteration <= iters:
            self.train_step()
    
    def train_step(self):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussian.oneupSHdegree()
        
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

        # Render
        # random_bg = (not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
        # bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        render_pkg = render_gs(viewpoint_cam, self.gaussian, self.pipe, self.background, self.opt)
        image, semantic_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["semantic_feature"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # RGB Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        # if random_bg:
        #     gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        # elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
        #     gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # opacity = self.gaussian.get_opacity
        # reg_loss = F.binary_cross_entropy(opacity, (opacity.detach() > 0.5) * 1.0)
        # loss = loss + reg_loss * self.reg_weight
        
        # Semantic Loss
        if self.opt.include_feature:
            gt_semantic_feature =  viewpoint_cam.semantic_feature
            target_mask = viewpoint_cam.target_mask
            sem_loss = l1_loss(semantic_feature, gt_semantic_feature, target_mask)            
        else:
            sem_loss = torch.tensor(0., device="cuda", requires_grad=True)

        # Depth Loss
        # depth_loss = torch.tensor([0.])
        # depth_loss = torch.tensor(0., device='cuda', requires_grad=True) 
        if self.metric_depth_loss_weight > 0:
            depth = render_pkg['depth']
            gt_depth = viewpoint_cam.depth.cuda()
            invalid_mask = (gt_depth < 0.1)
            valid_mask = ~invalid_mask
            n_valid_pixel = valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
                # loss = loss + depth_loss * self.metric_depth_loss_weight
        
        if self.opt.include_feature:
            loss = rgb_loss + sem_loss*self.metric_sem_loss_weight + depth_loss*self.metric_depth_loss_weight
        else:
            loss = rgb_loss #+ depth_loss*self.metric_depth_loss_weight

        loss.backward()

        if (self.iteration in self.checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
            chkpnt_path = os.path.join(self.scene.model_path, "chkpnt")
            os.makedirs(chkpnt_path, exist_ok=True)
            torch.save((self.gaussian.capture(), self.iteration), os.path.join(chkpnt_path, f"iteration_{self.iteration}.pth"))
            # self.scene.save_2gs(iter, self.args.num_slots, self.args.vis_cano, self.args.vis_center)
            point_cloud_path = os.path.join(self.scene.model_path, "point_cloud/iteration_{}".format(self.iteration))
            self.gaussian.save_ply(os.path.join(point_cloud_path, f"point_cloud.ply"))

        self.iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations+1:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussian.max_radii2D.shape[0] == 0:
                self.gaussian.max_radii2D = torch.zeros_like(radii)
            self.gaussian.max_radii2D[visibility_filter] = torch.max(self.gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussian.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussian.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussian.optimizer.step()
                self.gaussian.update_learning_rate(self.iteration)
                self.gaussian.optimizer.zero_grad(set_to_none=True)
            
            if (self.iteration in self.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)                
                torch.save((self.gaussian.capture(), self.iteration), self.scene.model_path + "/chkpnt" + str(self.iteration) + ".pth")

            # tb_writer
            if self.iteration % self.opt.report_interval == 0:
                self.training_report(viewpoint_cam.image_name, image, gt_image, semantic_feature, gt_semantic_feature, depth, gt_depth, rgb_loss, sem_loss, depth_loss, loss, self.iter_start.elapsed_time(self.iter_end)) 
        
        self.iteration += 1

        ## network gui
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(self.dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render_gs(custom_cam, self.gaussian, self.pipe, self.background, self.opt)
                        net_image = render_net_image(render_pkg, self.dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": self.gaussian.get_opacity.shape[0],
                        "loss": self.ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, self.dataset.dataset_path, metrics_dict)
                    if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
    
    def norm_depth(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm

    def training_report(self, name, rgb, gt_rgb, sem, gt_sem, depth, gt_depth, rgb_loss, sem_loss, depth_loss, total_loss, elapsed):
        if self.tb_writer:
            self.tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), self.iteration)
            self.tb_writer.add_scalar('train_loss_patches/sem_loss', sem_loss.item(), self.iteration) 
            self.tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), self.iteration)
            self.tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), self.iteration)

            self.tb_writer.add_scalar('iter_time', elapsed, self.iteration)
            self.tb_writer.add_images("train_view_{}/rgb".format(name), rgb[None], global_step=self.iteration)
            self.tb_writer.add_images("train_view_{}/gt_rgb".format(name), gt_rgb[None], global_step=self.iteration)
            self.tb_writer.add_images("train_view_{}/sem".format(name), sem[None], global_step=self.iteration)
            self.tb_writer.add_images("train_view_{}/gt_sem".format(name), gt_sem[None], global_step=self.iteration)
            self.tb_writer.add_images("train_view_{}/depth".format(name), self.norm_depth(depth)[None], global_step=self.iteration)
            self.tb_writer.add_images("train_view_{}/gt_depth".format(name), self.norm_depth(gt_depth)[None], global_step=self.iteration)

            self.tb_writer.add_histogram("scene/opacity_histogram", self.scene.gaussians.get_opacity, self.iteration)
            self.tb_writer.add_scalar('total_points', self.scene.gaussians.get_xyz.shape[0], self.iteration)
        # Report test and samples of training set
        # config = {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}

        # for idx, viewpoint in enumerate(config['cameras']):
        #     image = torch.clamp(renderFunc(viewpoint, scene.gaussian, *renderArgs)["render"], 0.0, 1.0)
        #     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        #     if self.tb_writer and (idx < 5):
        #         self.tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=self.iteration)
        #         self.tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=self.iteration)
        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def parse_config(config_path, override_dataset_path=None, override_model_path=None):
    # 读取 yaml
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # 命令行覆盖（关键：在 extract 之前改）
    if override_dataset_path is not None:
        cfg_dict["dataset_path"] = override_dataset_path
    if override_model_path is not None:
        cfg_dict["model_path"] = override_model_path

    # 创建虚拟 parser，让 ParamGroup 加载默认值
    from argparse import ArgumentParser
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # 生成 Namespace，用于 ParamGroup.extract
    ns = Namespace(**cfg_dict)

    # 替换 ParamGroup 中的默认参数
    model_params = lp.extract(ns)
    optim_params = op.extract(ns)
    pipeline_params = pp.extract(ns)

    return Namespace(**cfg_dict), model_params, optim_params, pipeline_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file.")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    cfg, lp, op, pp = parse_config(
        args.config,
        override_dataset_path=args.dataset_path,
        override_model_path=args.model_path,
    )

    cfg.save_iterations.append(cfg.iterations)
    # cfg.checkpoint_iterations.append(cfg.iterations)

    print("Optimizing " + cfg.model_path)
    safe_state(cfg.quiet)
    seed_everything(cfg.seed)
    network_gui.init(cfg.ip, cfg.port)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    trainer = Trainer(args=cfg, dataset=lp, opt=op, pipe=pp)
    trainer.train(cfg.iterations)
    print("\nTraining complete.")
