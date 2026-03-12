import os
import random
from tqdm import tqdm
from datetime import datetime   
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from quantizer import VectorQuantizer
# from quantizer_vqae import VectorQuantizerAE
from quantizer_ae import VectorAE
# from quantizerae import VQAutoEncoder1D
from dino.dino_dataloader import DinoDataset
from clip.clip_dataloader import PyramidMixtureDataset, DenseMixtureDataset, FlattenMixtureDataset
from semantic_feature_dataloader import SematicFeatureDataset, PyramidSematicFeatureDataset
from siglip2.sam_siglip2_dataloader import SAMSigLIP2Dataset
from siglip2.siglip2_dataloader import PyramidSigLIP2Dataset
import sys
sys.path.append('..')
from utils.lem_utils import index_to_rgb_images, generate_colors
import configargparse
import re
import torch.nn.functional as F
from openseg.openseg_dataloader import OpenSegDataset

# Configuration
random.seed(0)

# Global writer initialization
writer = None

class Trainer:
    def __init__(self, args):
        self.args = args
        self.tensorboard_step = 0
        self.writer = None
        self.prefix = self.name_prefix()
        self.initialize_writer()
        if self.args.ckpt_dir:
            self.ckpt_dir = self.args.ckpt_dir
        else:
            # self.ckpt_dir = os.path.join(self.args.image_dir, "../ckpts", self.args.feat_type)
            self.ckpt_dir = None
        self.save_dir = os.path.join(self.args.image_dir, "../ckpts/ae", self.args.feat_type)
        os.makedirs(self.save_dir, exist_ok=True)
        self.ckpt_iter = 0

    def initialize_writer(self):
        writer_dir_base = os.path.join("runs", self.args.dataset, self.args.image_dir.split("/")[-2], self.prefix)
        writer_dir = writer_dir_base
        if os.path.exists(writer_dir):
            counter = 0
            while os.path.exists(writer_dir):
                counter += 1
                writer_dir = f"{writer_dir_base}_{counter}"
        self.writer = SummaryWriter(log_dir=writer_dir)

    def name_prefix(self):
        dino_w_str = str(self.args.dino_weight).replace(".", "")
        kl_beta_str = str(self.args.kl_beta).replace(".", "")
        min_p_str = str(self.args.min_p).replace(".", "")
        max_p_str = str(self.args.max_p).replace(".", "")
        # Format the current timestamp. For example: "20240331-235959"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.args.feat_type}{dino_w_str}_{self.args.weight_mode}{min_p_str}{max_p_str}_{self.args.n_e}_{self.args.e_dim}_{kl_beta_str}_{timestamp}"

    def select_dataset(self):
        dataset_cls = {
            'dino': DinoDataset,
            'pyrclip': PyramidMixtureDataset,
            'mixclip': DenseMixtureDataset,
            'flattenclip': FlattenMixtureDataset,
            'clip_dino': SematicFeatureDataset,
            'pyrclip_dino': PyramidSematicFeatureDataset,
            'siglip2_sam':SAMSigLIP2Dataset,
            'pyrsiglip2': PyramidSigLIP2Dataset,
            'openseg': OpenSegDataset,
        }.get(self.args.feat_type, DinoDataset)
        return dataset_cls(self.args.image_dir)

    def train(self):
        data_loader = DataLoader(self.select_dataset(), batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        color_map = generate_colors(self.args.n_e)
        model, optimizer, scheduler = self.setup_training()

        model.train()

        for epoch in tqdm(range(self.ckpt_iter, self.args.epoch), dynamic_ncols=True):
            encoding_indices = []
            for feature in tqdm(data_loader, leave=False, dynamic_ncols=True):
                feature = feature.float().to("cuda")
                feature_hat, l2loss, cosloss, loss = model(feature)
                # print(perplexity)

                # if self.args.feat_type == "flattenclip" or self.args.feat_type == 'siglip2_sam':
                #     encoding_indices.append(min_encoding_indices)
                # else:   
                #     encoding_indices.append(min_encoding_indices.view(*feature.shape[:3], 1))
                
                # flattened_encoding_indices = min_encoding_indices.view(-1)
                # histogram = torch.histc(flattened_encoding_indices.float(), bins=args.n_e, min=0, max=args.n_e-1)
                # num_elements = histogram.sum()
                # frac = histogram / num_elements
                # flattened_encoding_indices_prob = encoding_indices_prob.view(-1, args.n_e)
                # load_balancing_loss = (frac * torch.mean(flattened_encoding_indices_prob, dim=0)).sum()
                
                # loss_d = -1 * torch.log2(d.mean() if d.mean() > 0 else torch.tensor(1e-10))
                
                # loss = loss_cos + args.load_balance_weight * load_balancing_loss + loss_kl*0.05 
                # l_rec = F.mse_loss(z_dec, z)
                # loss = 
                # loss = rec_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            # metric_loss = 1 - torch.mean(torch.cosine_similarity(feature[...,:512].to("cuda"), z_q[...,:512], dim = -1))
            # encoding_indices_tensor = torch.cat(encoding_indices, dim=0).to("cpu")
            self.write_tensorboard(l2loss=l2loss, cosloss=cosloss, loss=loss)
            # self.write_tensorboard(metric_loss, loss, loss_cos, loss_kl, load_balancing_loss, d, loss_d, perplexity)
            # if self.tensorboard_step % self.args.interv_n == 0:
            #     self.save_model(model, encoding_indices_tensor, color_map)
            self.tensorboard_step += 1
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch + 1, model, optimizer, scheduler)
        print(f"[TensorBoard] Logs are being saved to: {self.writer.log_dir}")
        self.save_checkpoint(self.args.epoch, model, optimizer, scheduler)

    def scan_ckpt(self):
        PATTERN = re.compile(r"^iteration_(\d+).pt$") 
        """返回目录中已有的轮数列表（按升序）。"""
        if not os.path.isdir(self.ckpt_dir):
            return []
        iters = []
        for fname in os.listdir(self.ckpt_dir):
            m = PATTERN.match(fname)
            if m:
                iters.append(int(m.group(1)))
        if not iters:
            return None
        iters = sorted(iters)
        if self.args.ckpt_num is not None:
            try:
                ckpt_num = int(self.args.ckpt_num)
                # 指定的存在就用它
                if ckpt_num in iters:
                    path = os.path.join(self.ckpt_dir, f"iteration_{ckpt_num}.pt")
                    return ckpt_num, path
                # 不存在则回退到最大
            except (TypeError, ValueError):
                # 非法值（比如 "latest"），忽略并回退到最大
                pass
        last = iters[-1]
        return last, os.path.join(self.ckpt_dir, f"iteration_{last}.pt")
    
    def setup_training(self):
        concat = self.args.feat_type in ['clip_dino', 'pyrclip_dino']
        model = VectorAE(self.args.feat_dim, self.args.e_dim, self.args.device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=1000)
        if self.ckpt_dir and os.path.exists(self.ckpt_dir):
            model, optimizer, scheduler = self.load_checkpoint(model, optimizer, scheduler)
        return model, optimizer, scheduler

    def save_checkpoint(self, epoch, model, optimizer, scheduler):
        # os.makedirs(self.ckpt_dir, exist_ok=True)
        # 文件名：ite{epoch}_codebook.pt 兼容你原来的命名
        fname = f"iteration_{epoch}.pt"
        path = os.path.join(self.save_dir, fname)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(self.args),
        }
        torch.save(ckpt, path)
        print(f"ckpt saved to {path}")
        return path

    def load_checkpoint(self,  model, optimizer, scheduler):
        res = self.scan_ckpt()  # 你已有的函数, 返回 (last_epoch, path) 或 None
        if res is None:
            print("No checkpoint found, starting from scratch.")
            self.ckpt_iter = 0
            return None
        else:
            latest, ckpt_path = res
            print(f"Find {ckpt_path}, starting from ite {latest}")

        ckpt = torch.load(ckpt_path, map_location=self.args.device)
        
        model.load_state_dict(ckpt["model_state"], strict=True)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        # 恢复步数/epoch
        self.tensorboard_step = ckpt.get("tensorboard_step", 0)
        self.ckpt_iter = ckpt.get("epoch", 0)

        return model, optimizer, scheduler
    
    def write_tensorboard(self, l2loss, cosloss, loss): 
        self.writer.add_scalar('train_loss/l2_loss', l2loss.item(), self.tensorboard_step)
        self.writer.add_scalar('train_loss/cos_loss', cosloss.item(), self.tensorboard_step)
        self.writer.add_scalar('train_loss/total_loss', loss.item(), self.tensorboard_step)
        # self.tb_writer.add_histogram("feat", outputs, self.tensorboard_step)

def parse_args():
    parser = configargparse.ArgParser(description="Training script parameters")
    parser.add_argument('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--output_codebook_dir', type=str, default=None)
    parser.add_argument('--base_codebook_path', type=str, default="")
    parser.add_argument('--feat_type', type=str, default='dino')
    parser.add_argument('--dino_weight', type=float, default=0.1)
    parser.add_argument('--load_balance_weight', type=float, default=1.0)
    parser.add_argument('--feat_dim', type=int, default=768) 
    parser.add_argument('--n_e', type=int, default=128)
    parser.add_argument('--e_dim', type=int, default=768) # 384, 512, 512 + 384 = 896
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--kl_beta', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_p', type=float, default=0.0)
    parser.add_argument('--weight_mode', type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
