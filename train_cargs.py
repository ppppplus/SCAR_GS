import torch
from random import randint
import sys, os
import yaml
from argparse import Namespace
import argparse
import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from pytorch3d.loss import chamfer_distance
import random

from scene import CARGSScene
from scene.cargs_model import CARGaussianModel
from scene.dataset_readers import fetchPly

from gaussian_renderer.render import render_gs, render_gs_with_screw, render_gs_with_screw_wo_branch_mixing, render_gs_with_mobility, render_gs_with_dqa
from gaussian_renderer import network_gui

from utils.general_utils import safe_state, get_linear_noise_func
from utils.loss_utils import l1_loss, ssim
from utils.metrics import *
from utils.log_utils import prepare_output_and_logger, list_parameters
from utils.net_utils import render_net_image
from utils.axis_init_utils import screw_to_axis
from utils.vis_utils import visualize_mobility_open3d
from articulation.dqamodel import DQArtiModel
def weighted_chamfer(x1, w1, x2, w2, tau, device):
    if x1.shape[0] == 0 or (w2 > tau).sum() == 0:
        return torch.zeros([], device=device)

    d12 = chamfer_distance(
        x1[None],
        x2[w2 > tau][None],
        batch_reduction=None,
        point_reduction=None,
        single_directional=True,
    )[0]

    d21 = chamfer_distance(
        x2[None],
        x1[w1 > tau][None],
        batch_reduction=None,
        point_reduction=None,
        single_directional=True,
    )[0]
    wd12 = (d12 * w1) / (w1.sum() + 1e-6)
    wd21 = (d21 * w2) / (w2.sum() + 1e-6)
    # return (
    #     (d12 * w1).sum() / (w1.sum() + 1e-6)
    #     + (d21 * w2).sum() / (w2.sum() + 1e-6)
    # ), wd12, wd21
    return wd12.sum() + wd21.sum(), d12, d21

class Trainer:
    def __init__(self, args, dataset, opt, pipe):
        self.args = args
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe

        self.tb_writer = prepare_output_and_logger(dataset)
        screw_init_path = os.path.join(os.path.dirname(args.base_model_path1), "catch", "ours_" + str(args.start_iterations), "screw_init.json")
        self.gaussian0 = CARGaussianModel(dataset.sh_degree)
        self.gaussian1 = CARGaussianModel(dataset.sh_degree)
        self.dqamodel = DQArtiModel()

        self.dqamodel.init_from_axis_json(
            screw_init_path,
            angle=-math.pi / 4
        )

        self.scene0 = CARGSScene(dataset, self.gaussian0, dataset_path=args.dataset_path0, base_model_path=args.base_model_path0, screw_init_path=screw_init_path, load_iteration=args.start_iterations, state=0, shuffle=True)
        self.scene1 = CARGSScene(dataset, self.gaussian1, dataset_path=args.dataset_path1, base_model_path=args.base_model_path1, screw_init_path=screw_init_path, load_iteration=args.start_iterations, state=1, shuffle=True)
        
        self.scene0.gaussians.training_setup(opt)
        self.scene1.gaussians.training_setup(opt)
        self.dqamodel.training_setup(opt)

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

        self.save_gt = True

        if self.start_checkpoint:
            chkpnt_path = os.path.join(self.scene0.model_path, "chkpnt", f"chkpnt_{self.start_checkpoint}.pth")
            # (model_params, self.iteration) = torch.load(checkpoint_path)
            ckpt = torch.load(chkpnt_path)
            self.gaussian0.restore(ckpt["gaussian0"], opt)
            self.gaussian1.restore(ckpt["gaussian1"], opt)
            self.dqamodel.restore(ckpt["dqamodel"])
            self.iteration = ckpt["iteration"] + 1
        else:
            self.iteration = 1

        self.stages = [
            ("mobility", self.args.iters[0]),
            ("mobility+render",  self.args.iters[1]),
            ("artidq",  self.args.iters[2]),
        ]
        self.stage_bar = tqdm.tqdm(
            total=len(self.stages),
            desc="Stage Progress",
            position=0,      # 固定在最上面
            leave=True,
        )

    def locate_stage(self):
        """
        Given a global iteration index, locate:
        - which stage it belongs to
        - the inner iteration index within that stage

        Returns:
            stage_idx (int)
            inner_iter (int)
        """
        acc = 0
        for stage_idx, (_, stage_iters) in enumerate(self.stages):
            if self.iteration < acc + stage_iters:
                return stage_idx, self.iteration - acc
            acc += stage_iters

        # iteration 超过所有 stage，总是落在最后一个 stage 的末尾
        last_stage_idx = len(self.stages) - 1
        last_stage_iters = self.stages[last_stage_idx][1]
        return last_stage_idx, last_stage_iters

    def train_all(
        self,
        iters
    ):
        start_stage_idx, inner_start_iter = self.locate_stage()

        print(
            f"[Resume] global_iter={self.iteration}, "
            f"start_stage={self.stages[start_stage_idx][0]}, "
            f"inner_iter={inner_start_iter}"
        )
        # add internal gaussians to gaussian 0
        new_xyz, internal_mask = self.gaussian0.add_random_points_inside_object(
            num_new_points=1000, expansion_factor=0.1, ratio=0.1
        )
        # register mobility grad hook, only update mobilities in mob_mask
        self.gaussian0._register_mobility_grad_hook()
        self.gaussian1._register_mobility_grad_hook()

        # for it in range(self.opt.iterations):
        for stage_idx, (stage_name, stage_iters) in enumerate(self.stages):
            if stage_idx < start_stage_idx:
                self.stage_bar.update(1)
                continue
            # 更新大 progress bar 的描述
            self.stage_bar.set_description(
                f"Stage Progress ({stage_idx + 1}/{len(self.stages)})"
            )

            # -----------------------------
            # 小 progress bar（当前 stage）
            # -----------------------------
            iter_bar = tqdm.tqdm(
                total=stage_iters,
                desc=stage_name,
                position=1,     # 永远在第二行
                leave=False,    # stage 结束就消失
                dynamic_ncols=True,
            )
            # -----------------------------
            # Stage dispatch
            # -----------------------------

            # ========= 实际训练 =========
            if stage_name == "mobility":
                self.train_mobility(iter_bar)
                iter_bar.close()
                self.iteration = iters[0] + 1

            elif stage_name == "mobility+render":
                self.train_mobility_with_render(iter_bar)
                iter_bar.close()
                self.iteration = iters[0] + iters[1] + 1

            else:
                self.save_gt = True
                self.train_artidq(iter_bar)
                iter_bar.close()
                self.iteration = iters[0] + iters[1] + iters[2] + 1

            self.stage_bar.update(1)
            
        self.stage_bar.close()

    def init_axis_by_chamfer(self, tau=0.01, device="cuda"):
        """
        Cross-static geometric consistency loss (Stage 2(a)),
        adapted from SplArt compute_geometric_loss(mode="cs").

        Args:
            gaussian0, gaussian1:
                Gaussian models for state 0 and state 1
            tau:
                opacity / weight culling threshold

        Returns:
            loss_geom: scalar tensor
        """

        # --------------------------------------------------
        # 1. Masks: only target object participates
        # --------------------------------------------------
        mask0 = self.gaussian0.get_mob_mask   # (N0,)
        mask1 = self.gaussian1.get_mob_mask   # (N1,)

        # --------------------------------------------------
        # 2. Geometry (anchors & targets)
        # --------------------------------------------------
        # anchor geometry (treated as static references)
        ref0 = self.gaussian0.get_xyz[mask0].detach()
        ref1 = self.gaussian1.get_xyz[mask1].detach()

        # target geometry (same points, but weighted by mobility)
        xyz0 = self.gaussian0.get_xyz[mask0].detach()
        xyz1 = self.gaussian1.get_xyz[mask1].detach()
        opacity0 = self.gaussian0.get_opacity[mask0].detach().squeeze(-1)
        opacity1 = self.gaussian1.get_opacity[mask1].detach().squeeze(-1)
        m0 = self.gaussian0.get_mobility.squeeze()[mask0]   # (K0,)
        m1 = self.gaussian1.get_mobility.squeeze()[mask1]   # (K1,)

        means_tgt0 = torch.cat([xyz0, xyz1], dim=0)
        weights_tgt0 = torch.cat(
            (opacity0, opacity1 * (1 - m1))
        )
        # weights_tgt0: 0的所有+1的静态
        loss0, d12, d21 = weighted_chamfer(ref0, opacity0, means_tgt0, weights_tgt0, tau=tau, device=device)


    def compute_cross_static_geometric_loss(
            self,
            it,
            device,
            tau=0.01,
        ):
        """
        Cross-static geometric consistency loss (Stage 2(a)),
        adapted from SplArt compute_geometric_loss(mode="cs").

        Args:
            gaussian0, gaussian1:
                Gaussian models for state 0 and state 1
            tau:
                opacity / weight culling threshold

        Returns:
            loss_geom: scalar tensor
        """

        # --------------------------------------------------
        # 1. Masks: only target object participates
        # --------------------------------------------------
        mask0 = self.gaussian0.get_mob_mask   # (N0,)
        mask1 = self.gaussian1.get_mob_mask   # (N1,)

        # --------------------------------------------------
        # 2. Geometry (anchors & targets)
        # --------------------------------------------------
        # anchor geometry (treated as static references)
        ref0 = self.gaussian0.get_xyz[mask0].detach()
        ref1 = self.gaussian1.get_xyz[mask1].detach()

        # target geometry (same points, but weighted by mobility)
        xyz0 = self.gaussian0.get_xyz[mask0].detach()
        xyz1 = self.gaussian1.get_xyz[mask1].detach()
        opacity0 = self.gaussian0.get_opacity[mask0].detach().squeeze(-1)
        opacity1 = self.gaussian1.get_opacity[mask1].detach().squeeze(-1)
        m0 = self.gaussian0.get_mobility.squeeze()[mask0]   # (K0,)
        m1 = self.gaussian1.get_mobility.squeeze()[mask1]   # (K1,)

        means_tgt0 = torch.cat([xyz0, xyz1], dim=0)
        weights_tgt0 = torch.cat(
            (opacity0, opacity1 * (1 - m1))
        )
        means_tgt1 = torch.cat([xyz1, xyz0], dim=0)
        weights_tgt1 = torch.cat(
            (opacity1, opacity0 * (1 - m0))
        )
        # --------------------------------------------------
        # 4. Weighted Chamfer Distance (same as SplArt)
        # --------------------------------------------------
        # --------------------------------------------------
        # 6. Final cross-static geometric loss
        # --------------------------------------------------
        # weights_tgt0: 0的所有+1的静态，weights_tgt1: 1的所有+0的静态
        loss0, d12, d21 = weighted_chamfer(ref0, opacity0, means_tgt0, weights_tgt0, tau=tau, device=device)
        loss1, _, _ = weighted_chamfer(ref1, opacity1, means_tgt1, weights_tgt1, tau=tau, device=device)
        # if it > 0 and it % 99 == 0:
        #     np.save("/home/ubuntu/TJH/Work/aff_ws/CAR_GS/my_scripts/data/gt_idmask/stage1/means_tgt0.npy", means_tgt0.detach().cpu().numpy())
        #     np.save("/home/ubuntu/TJH/Work/aff_ws/CAR_GS/my_scripts/data/gt_idmask/stage1/weights_tgt0.npy", weights_tgt0.detach().cpu().numpy())
        #     np.save("/home/ubuntu/TJH/Work/aff_ws/CAR_GS/my_scripts/data/gt_idmask/stage1/d21.npy", d21.detach().cpu().numpy())

        return loss0 + loss1

    def compute_cross_mobility_geometric_loss(
            self,
            dq_xyz,
            dq_opacity,
            device,
            tau=0.01,
        ):
        """
        Cross-static geometric consistency loss (Stage 2(a)),
        adapted from SplArt compute_geometric_loss(mode="cs").

        Args:
            gaussian0, gaussian1:
                Gaussian models for state 0 and state 1
            tau:
                opacity / weight culling threshold

        Returns:
            loss_geom: scalar tensor
        """

        # --------------------------------------------------
        # 1. Masks: only target object participates
        # --------------------------------------------------
        mask0 = self.gaussian0.get_mob_mask   # (N0,)
        mask1 = self.gaussian1.get_mob_mask   # (N1,)

        # --------------------------------------------------
        # 2. Geometry (anchors & targets)
        # --------------------------------------------------
        # anchor geometry (treated as static references)
        ref = self.gaussian1.get_xyz[mask1].detach()
        weights_ref = self.gaussian1.get_opacity[mask1].detach().squeeze(-1)
        tgt = dq_xyz
        weights_tgt = dq_opacity.squeeze(-1)

        # --------------------------------------------------
        # 4. Weighted Chamfer Distance (same as SplArt)
        # --------------------------------------------------
        # --------------------------------------------------
        # 6. Final cross-static geometric loss
        # --------------------------------------------------
        loss, wd12, wd21 = weighted_chamfer(ref, weights_ref, tgt, weights_tgt, tau=tau, device=device)
        return loss

    def train_mobility(self, iter_bar):
        """
        Train mobility using two Gaussian states (start / end).
        """

        device = torch.device("cuda")
        # self.gaussian0.freeze_except_mobility()
        # self.gaussian1.freeze_except_mobility()

        self.optimizer = torch.optim.Adam(
            [self.gaussian0._mobility, self.gaussian1._mobility],
            lr=self.opt.mobility_lr
        )

        # --------------------------------------------------
        # 4. Prepare static / target split  (CORRECT VERSION)
        # --------------------------------------------------
        mask0 = self.gaussian0.get_mob_mask   # BoolTensor (N0,)
        mask1 = self.gaussian1.get_mob_mask   # BoolTensor (N1,)

        # --------------------------------------------------
        # Training loop (mobility-only)
        # --------------------------------------------------
        for it in range(iter_bar.total): 
            self.optimizer.zero_grad()

            # ---- masked mobility (ONLY target object has mobility) ----
            m0_full = self.gaussian0.get_mobility.squeeze()
            m1_full = self.gaussian1.get_mobility.squeeze()

            # defensive masking (non-target always 0)
            m0 = torch.zeros_like(m0_full)
            m1 = torch.zeros_like(m1_full)
            m0[mask0] = m0_full[mask0]
            m1[mask1] = m1_full[mask1]

            # # --------------------------------------------------
            # # mobility regularization
            # # --------------------------------------------------
            # # only regularize target object
            reg0 = (m0[mask0]).mean()
            reg1 = (m1[mask1]).mean()
            reg_loss = reg0 + reg1
            loss_geom = self.compute_cross_static_geometric_loss(
                it=it,
                device=device,
                tau=0.01,
            )

            loss = loss_geom + 0.001*reg_loss
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if it > 0 and it % 50 == 0:
                    iter_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                    iter_bar.update(50)
                if (it + self.iteration) in self.checkpoint_iterations:
                    iteration = it + self.iteration
                    print(f"\n[ITER {iteration}] Saving mobility Gaussians")
                    save_path = os.path.join(self.scene0.model_path, "chkpnt")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        {
                            "iteration": iteration,
                            "gaussian0": self.gaussian0.capture(),
                            "gaussian1": self.gaussian1.capture(),
                            "dqamodel": self.dqamodel.capture(),
                        },
                        f"{save_path}/chkpnt_{iteration}.pth"
                    )
        # --------------------------------------------------
        # 7. Save mobility
        # --------------------------------------------------
        # mobility0 = self.gaussian0.get_mobility.detach().cpu().numpy()
        # mobility1 = self.gaussian1.get_mobility.detach().cpu().numpy()

        # np.save(self.args.dataset_path0 + "/mobility.npy", mobility0)
        # np.save(self.args.dataset_path1 + "/mobility.npy", mobility1)
        # print("Mobility training finished and saved.")  


    def sample_viewpoint(self):
        """
        Randomly sample a training camera from state0 or state1.

        Returns:
            camera: GS Camera
            gt_image: torch.Tensor [3, H, W] in [0,1]
            state: int (0 or 1)
        """

        # ------------------------------------
        # 1. randomly choose state
        # ------------------------------------
        state = random.randint(0, 1)

        if state == 0:
            cameras = self.scene0.getTrainCameras()
        else:
            cameras = self.scene1.getTrainCameras()

        # ------------------------------------
        # 2. randomly choose camera
        # ------------------------------------
        cam = random.choice(cameras)

        # ------------------------------------
        # 3. fetch GT image
        # ------------------------------------
        # GS convention: camera.original_image
        gt_image = cam.original_image.to("cuda")

        return cam, gt_image, state
    
    def train_mobility_with_render(self, iter_bar):
        """
        Stage 2(b): Cross-Static Rendering (CSR) mobility refinement
        Assumes:
        - gaussian0 / gaussian1 already loaded
        - mobility already initialized (from Stage 2(a))
        - only mob_mask points are trainable
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.opt.mobility_lr_render
        device = torch.device("cuda")
        # self.progress_bar = tqdm.tqdm(range(self.iteration, self.args.stage2_iterations+1), desc="Training progress")
        # --------------------------------------------------
        # 1. Freeze everything except mobility
        # # --------------------------------------------------
        # self.gaussian0.freeze_except_mobility()
        # self.gaussian1.freeze_except_mobility()
        self.gaussian0.prune_internal_points() 
        self.gaussian0.apply_mob_mask_hook()
        self.gaussian1.apply_mob_mask_hook()
        self.gaussian0.update_learning_rate(self.iteration)
        self.gaussian1.update_learning_rate(self.iteration)

        # --------------------------------------------------
        # 2. Prepare masks
        # --------------------------------------------------
        mask0 = self.gaussian0.get_mob_mask
        mask1 = self.gaussian1.get_mob_mask

        # --------------------------------------------------
        # 3. Training loop
        # --------------------------------------------------
        # self.progress_bar.set_description("Stage2(b) Mobility")
        
        for it in range(iter_bar.total): 
            self.iter_start.record()
            self.optimizer.zero_grad()
            self.gaussian0.optimizer.zero_grad()
            self.gaussian1.optimizer.zero_grad()


            # --------------------------------------------------
            # 3.1 Sample a training view
            # --------------------------------------------------
            viewpoint_cam, gt_image, state = self.sample_viewpoint()
            # state_weight = 5.0 if state == 1 else 5.0
            # state ∈ {0,1}
            if state == 0:
                gauss_cur = self.gaussian0
                gauss_oth = self.gaussian1
                mask_cur = mask0
                mask_oth = mask1
            else:
                gauss_cur = self.gaussian1
                gauss_oth = self.gaussian0
                mask_cur = mask1
                mask_oth = mask0

            m_oth_full = gauss_oth.get_mobility.squeeze()
            # --------------------------------------------------
            # 3.4 Render CSR image
            # --------------------------------------------------
            render_pkg = render_gs_with_mobility(
                viewpoint_camera=viewpoint_cam,
                gauss_cur=gauss_cur,
                gauss_oth=gauss_oth,
                mob_mask_oth=mask_oth,
                pipe=self.pipe,
                bg_color=self.background,
                opt=self.opt,
            )

            image, semantic_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["semantic_feature"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth = render_pkg["depth"]
            # --------------------------------------------------
            # 3.5 Photometric loss
            # --------------------------------------------------
            # rgb_loss = torch.mean(torch.abs(image - gt_image))
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)

            rgb_loss = (
                (1.0 - self.opt.lambda_dssim) * Ll1
                + self.opt.lambda_dssim * (1.0 - ssim_value)
            )

            gt_depth = viewpoint_cam.depth.cuda()
            invalid_mask = (gt_depth < 0.1)
            valid_mask = ~invalid_mask
            n_valid_pixel = valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
            # --------------------------------------------------
            # 3.6 Mobility regularization (prevent all-mobile)
            # --------------------------------------------------
            reg_loss = m_oth_full[mask_oth].mean()

            loss = rgb_loss + depth_loss + 0.01 * reg_loss

            loss.backward()
            # self.optimizer.step()
            self.gaussian0.optimizer.step()
            self.gaussian1.optimizer.step()
            self.gaussian0.update_learning_rate(self.iteration)
            self.gaussian1.update_learning_rate(self.iteration)
            self.iter_end.record()
            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if it > 0 and it % 50 == 0:
                    iter_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                    iter_bar.update(50)
                if (it + self.iteration) in self.checkpoint_iterations:
                    iteration = it + self.iteration
                    print(f"\n[ITER {iteration}] Saving mobility Gaussians")
                    save_path = os.path.join(self.scene0.model_path, "chkpnt")  # 或你统一的 model_path
                    torch.save(
                        {
                            "iteration": iteration,
                            "gaussian0": self.gaussian0.capture(),
                            "gaussian1": self.gaussian1.capture(),
                            "dqamodel": self.dqamodel.capture(),
                        },
                        f"{save_path}/chkpnt_{iteration}.pth"
                    )
                # tb_writer
                if it % self.opt.report_interval == 0:
                    self.training_report(iter=it+self.iteration, rgb_loss=rgb_loss, depth_loss=depth_loss, reg_loss=reg_loss, total_loss=loss, elapsed=self.iter_start.elapsed_time(self.iter_end), render_debug=True) 
            # self.iteration += 1

    def train_artidq(self, iter_bar):
        mask1 = self.gaussian1.get_mob_mask
        # --------------------------------------------------
        # 1. densify gaussian0 by mobility
        # # --------------------------------------------------
        # self.gaussian0.freeze_except_mobility()
        # self.gaussian1.freeze_except_mobility()
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = self.opt.mobility_lr_render*5
        # self.gaussian0.prune_internal_points() # 删除训练mobs添加的内部点
        # self.gaussian0.prune_points()
        with torch.no_grad():
            mob_thresh = 0.1
            mob1 = self.gaussian1.get_mobility.squeeze()   # [N1]
            select_mask = mask1 & (mob1 < mob_thresh)            # bool mask
        data = self.gaussian1.extract_gaussians(select_mask, detach=False, device=self.gaussian0.get_xyz.device)
        self.gaussian0.add_gaussians_from_data(data)    # 将gaussian1的静态点加入到gaussian0中
        # self.gaussian0.apply_internal_mask_hook()   # 只训练gaussian0的内部点的高斯参数
        mask0 = self.gaussian0.get_mob_mask
        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        # self.progress_bar.set_description("Stage3 Dual quaternion")
        for it in range(iter_bar.total): 
            self.iter_start.record()
            # --------------------------------------------------
            # 3.1 Sample a training view
            # --------------------------------------------------
            # Pick a random Camera
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene1.getTrainCameras().copy()
            viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

            # state = 0 
            gauss_cur = self.gaussian0
            mask_cur = mask0
            # --------------------------------------------------
            # 3.2 Masked mobility
            # --------------------------------------------------
            m_cur_full = gauss_cur.get_mobility.squeeze()

            m_cur = torch.zeros_like(m_cur_full)

            m_cur[mask_cur] = m_cur_full[mask_cur]

            # --------------------------------------------------
            # 3.4 Render CSR image
            # --------------------------------------------------
            render_pkg, dq_xyz, dq_opacity = render_gs_with_dqa(
                viewpoint_camera=viewpoint_cam,
                pc=gauss_cur,
                pipe=self.pipe,
                bg_color=self.background,
                opt=self.opt,
                dqamodel=self.dqamodel,
                mask=mask_cur,
            )

            image, semantic_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["semantic_feature"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth = render_pkg["depth"]
            # --------------------------------------------------
            # Photometric loss
            # --------------------------------------------------
            # rgb_loss = torch.mean(torch.abs(image - gt_image))
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)

            rgb_loss = (
                (1.0 - self.opt.lambda_dssim) * Ll1
                + self.opt.lambda_dssim * (1.0 - ssim_value)
            )
            # --------------------------------------------------
            # Depth loss
            # --------------------------------------------------
            gt_depth = viewpoint_cam.depth.cuda()
            invalid_mask = (gt_depth < 0.1)
            valid_mask = ~invalid_mask
            n_valid_pixel = valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
            # --------------------------------------------------
            # geometric loss
            # --------------------------------------------------
            geo_loss = self.compute_cross_mobility_geometric_loss(dq_xyz, dq_opacity, device=self.gaussian0.get_xyz.device)
            # --------------------------------------------------
            # Mobility regularization (prevent all-mobile)
            # --------------------------------------------------
            reg_loss = m_cur[mask_cur].mean()
            
            # optimizer
            if it < 15000:
                # loss = rgb_loss + depth_loss + geo_loss + 0.001*reg_loss
                if it == 5000:
                    self.dqamodel.analyze_and_lock_joints()
                loss = geo_loss
                loss.backward()
                self.dqamodel.optimizer.step()
                self.dqamodel.update_learning_rate(it)
                self.dqamodel.optimizer.zero_grad()
            else:
                loss = (rgb_loss + depth_loss + geo_loss + 0.001*reg_loss)*5
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # self.gaussian0.optimizer.step()
            # self.gaussian0.update_learning_rate(it)
            # self.gaussian0.optimizer.zero_grad()
            self.iter_end.record()
            
            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if it>0 and it % 50 == 0:
                    iter_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                    iter_bar.update(50)
                if (it + self.iteration) in self.checkpoint_iterations:
                    iteration = it + self.iteration
                    print(f"\n[ITER {iteration}] Saving mobility Gaussians")
                    save_path = os.path.join(self.scene0.model_path, "chkpnt")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        {
                            "iteration": iteration,
                            "gaussian0": self.gaussian0.capture(),
                            "gaussian1": self.gaussian1.capture(),
                            "dqamodel": self.dqamodel.capture(),
                        },
                        f"{save_path}/chkpnt_{iteration}.pth"
                    )
                # tb_writer
                if it % self.opt.report_interval == 0:
                    self.training_report(iter=it+self.iteration, rgb_loss=rgb_loss, depth_loss=depth_loss, reg_loss=reg_loss, total_loss=loss, elapsed=self.iter_start.elapsed_time(self.iter_end), render_debug=True, dqa=True, geo_loss=geo_loss) 

    def norm_depth(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm
    
    def training_report(
            self,
            iter,
            rgb_loss,
            depth_loss,
            reg_loss,
            total_loss,
            elapsed,
            render_debug=False,
            dqa=False,
            geo_loss=None,
        ):
            """
            TensorBoard report for Stage 2(b) mobility training
            """
            if self.tb_writer is None:
                return

            # --------------------------------------------------
            # 1. scalar losses
            # --------------------------------------------------
            self.tb_writer.add_scalar(
                "loss/rgb_loss",
                rgb_loss.item(),
                iter,
            )

            self.tb_writer.add_scalar(
                "loss/depth_loss",
                depth_loss.item(),
                iter,
            )

            self.tb_writer.add_scalar(
                "loss/mobility_reg_loss",
                reg_loss.item(),
                iter,
            )

            self.tb_writer.add_scalar(
                "loss/total_loss",
                total_loss.item(),
                iter,
            )

            self.tb_writer.add_scalar(
                "iter_time",
                elapsed,
                iter,
            )

            # -------------------------------------------------
            # 2. optional CSR rendering debug
            # --------------------------------------------------
            if render_debug:
                if dqa:
                    if geo_loss is not None:
                        self.tb_writer.add_scalar(
                            "loss/geo_loss",
                            geo_loss.item(),
                            iter,
                        )
                    for viewpoint_cam in self.scene1.getTrainCameras().copy():
                        name = viewpoint_cam.image_name
                        mask0 = self.gaussian0.get_mob_mask
                        gauss_cur = self.gaussian0
                        mask_cur = mask0
                        m_cur_full = gauss_cur.get_mobility.squeeze()
                        m_cur = torch.zeros_like(m_cur_full)
                        m_cur[mask_cur] = m_cur_full[mask_cur]
                        render_pkg, _, _ = render_gs_with_dqa(
                            viewpoint_camera=viewpoint_cam,
                            pc=gauss_cur,
                            pipe=self.pipe,
                            bg_color=self.background,
                            opt=self.opt,
                            dqamodel=self.dqamodel,
                            mask=mask_cur,
                        )
                        image, semantic_feature, depth = render_pkg["render"], render_pkg["semantic_feature"], render_pkg["depth"]
                        if self.save_gt:
                            gt_image = viewpoint_cam.original_image
                            gt_semantic_feature = viewpoint_cam.semantic_feature
                            gt_depth = viewpoint_cam.depth
                            self.tb_writer.add_images("train_view/{}/gt_rgb".format(name), gt_image[None], global_step=iter)
                            # self.tb_writer.add_images("train_view/{}/gt_sem".format(name), gt_semantic_feature[None], global_step=iter)
                            self.tb_writer.add_images("train_view/{}/gt_depth".format(name), self.norm_depth(gt_depth)[None], global_step=iter)
                            
                        self.tb_writer.add_images("train_view/{}/rgb".format(name), image[None], global_step=iter)
                        # self.tb_writer.add_images("train_view/{}/sem".format(name), semantic_feature[None], global_step=iter)
                        self.tb_writer.add_images("train_view/{}/depth".format(name), self.norm_depth(depth)[None], global_step=iter)
                else:
                        # only render one camera per state to avoid heavy logging
                        cams0 = self.scene0.getTrainCameras()
                        cams1 = self.scene1.getTrainCameras()
                        m0 = self.gaussian0.get_mobility.squeeze()   # (N0,)
                        m1 = self.gaussian1.get_mobility.squeeze()   # (N1,)
                        mask0 = self.gaussian0.get_mob_mask
                        mask1 = self.gaussian1.get_mob_mask

                        m0_tgt = m0[mask0]
                        m1_tgt = m1[mask1]

                        # --------------------------------------
                        # Histogram (MOST IMPORTANT)
                        # --------------------------------------
                        self.tb_writer.add_histogram(
                            "mobility/state0/target",
                            m0_tgt.detach(),
                            iter
                        )
                        self.tb_writer.add_histogram(
                            "mobility/state1/target",
                            m1_tgt.detach(),
                            iter
                        )

                        debug_pairs = [
                            (0, self.gaussian0, self.gaussian1, cams0),
                            (1, self.gaussian1, self.gaussian0, cams1),
                        ]

                        for sid, gauss_cur, gauss_oth, cams in debug_pairs:
                            if len(cams) == 0:
                                continue

                            # for cam in cams:
                            cam = cams[3]
                            name = cam.image_name

                            render_pkg = render_gs_with_mobility(
                                viewpoint_camera=cam,
                                gauss_cur=gauss_cur,
                                gauss_oth=gauss_oth,
                                mob_mask_oth=gauss_oth.get_mob_mask,
                                pipe=self.pipe,
                                bg_color=self.background,
                                opt=self.opt,
                            )

                            image = render_pkg["render"]
                            depth = render_pkg["depth"]

                            self.tb_writer.add_images(
                                f"train_view/state{sid}/{name}/rgb",
                                image[None],
                                global_step=iter,
                            )

                            self.tb_writer.add_images(
                                f"train_view/state{sid}/{name}/depth",
                                self.norm_depth(depth)[None],
                                global_step=iter,
                            )

                            if self.save_gt:
                                gt_image = cam.original_image
                                self.tb_writer.add_images(
                                    f"train_view/state{sid}/{name}/gt_rgb",
                                    gt_image[None],
                                    global_step=iter,
                                )

                self.save_gt = False

            # --------------------------------------------------
            # 4. scene statistics
            # --------------------------------------------------
            self.tb_writer.add_scalar(
                "scene/total_points_state0",
                self.gaussian0.get_xyz.shape[0],
                iter,
            )

            self.tb_writer.add_scalar(
                "scene/total_points_state1",
                self.gaussian1.get_xyz.shape[0],
                iter,
            )


def parse_config(
    config_path,
    override_dataset_path0=None,
    override_dataset_path1=None,
    override_base_model_path0=None,
    override_base_model_path1=None,
    override_model_path=None,
):
    # 1. 读取 yaml
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # 2. dataset_path*
    if override_dataset_path0 is not None:
        cfg_dict["dataset_path0"] = override_dataset_path0

    if override_dataset_path1 is not None:
        cfg_dict["dataset_path1"] = override_dataset_path1

    # 3. base_model_path*
    if override_base_model_path0 is not None:
        cfg_dict["base_model_path0"] = override_base_model_path0

    if override_base_model_path1 is not None:
        cfg_dict["base_model_path1"] = override_base_model_path1

    # 4. model_path（主输出目录）
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
    parser.add_argument("--base_model_path0", type=str, default=None)
    parser.add_argument("--base_model_path1", type=str, default=None)
    parser.add_argument("--dataset_path0", type=str, default=None)
    parser.add_argument("--dataset_path1", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    cfg, lp, op, pp = parse_config(
        args.config,
        override_dataset_path0=args.dataset_path0,
        override_dataset_path1=args.dataset_path1,
        override_base_model_path0=args.base_model_path0,
        override_base_model_path1=args.base_model_path1,
        override_model_path=args.model_path,
    )

    cfg.checkpoint_iterations.append(cfg.iters[0])
    cfg.checkpoint_iterations.append(cfg.iters[0]+cfg.iters[1])
    cfg.checkpoint_iterations.append(cfg.iterations)

    # cfg.checkpoint_iterations.append(cfg.iterations)

    print("Optimizing " + cfg.model_path)
    safe_state(cfg.quiet)
    seed_everything(cfg.seed)
    network_gui.init(cfg.ip, cfg.port)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    trainer = Trainer(args=cfg, dataset=lp, opt=op, pipe=pp)
    trainer.train_all(cfg.iters)
    # trainer.train_mobility_with_render()
    print("\nTraining complete.")
