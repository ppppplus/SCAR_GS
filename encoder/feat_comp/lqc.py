"""
Borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vq_model import VQModel
import math

def pad_to_multiple_4(x, mode="replicate"):
    """
    x: [B, C, H, W]
    返回:
      x_pad: 补到(⌈H/4⌉*4, ⌈W/4⌉*4)
      pad:   (left, right, top, bottom) 便于事后反裁
    """
    B, C, H, W = x.shape
    Ht = math.ceil(H / 4) * 4
    Wt = math.ceil(W / 4) * 4
    pad_h = Ht - H
    pad_w = Wt - W
    # F.pad 的顺序是 (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
    x_pad = F.pad(x, pad, mode=mode)  # mode: "replicate" / "reflect" / "constant"
    return x_pad, pad, (H, W)

def unpad_to_orig(y, orig_hw):
    """
    y:   [B, C, H_pad, W_pad]
    截回 orig_hw 尺寸
    """
    H, W = orig_hw
    return y[..., :H, :W]

class VectorQuantizerLQC(nn.Module):
    def __init__(self, fdim, n_e, e_dim, beta, device, concat = False):
        super().__init__()
        self.sem_ae = VQModel(
                    in_channels=fdim,
                    out_channels=fdim,
                    latent_channels=e_dim,
                    norm_num_groups=4,
                    block_out_channels=[256, 64, 16],
                    down_block_types=["DownEncoderBlock2D"] * 3,
                    up_block_types=["UpDecoderBlock2D"] * 3,
                    layers_per_block=1,
                    norm_type="group",
                    num_vq_embeddings=n_e,
                )
        self.sem_ae.to(device)
        self.fdim = fdim
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.device = device
    
    def forward(self, z):
        # z_flattened = z.view(-1, self.fdim)
        # z = z.permute(0,3,1,2)  
        # z_pad, pad, orig_hw = pad_to_multiple_4(z, mode="replicate")
        res = self.sem_ae(z)

        sample = res["sample"]  # [B, fdim, H, W]
        # z_hat = unpad_to_orig(sample, orig_hw)  
        commit_loss = res["commit_loss"]
        min_encoding_indices = self.sem_ae.info[2].unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1) 
        # print(sample.shape)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-6)))
        # print(perplexity)
        rec_loss = F.mse_loss(sample, z)
       
        return rec_loss, commit_loss, perplexity
    
    @torch.no_grad()
    def ende_vectors(
            self,
            x: torch.Tensor
        ):
        # z_flattened = z.view(-1, self.fdim).to(self.device)
        res = self.sem_ae(x)
        return res["sample"] 
    
    @torch.no_grad()
    def encode_vectors(
            self,
            x: torch.Tensor
        ):
        output = self.sem_ae.encode(x)
        return output["latents"]
    
    @torch.no_grad()
    def decode_vectors(
            self,
            h: torch.Tensor
        ):
        output = self.sem_ae.decode(h)
        return output["sample"]

    @torch.no_grad()
    def encode_quantize_vectors(
            self,
            x: torch.Tensor
        ):
        h, q = self.sem_ae.encode_quantize(x)
        return h, q

    

    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device, concat = False, dino_weight = 0):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim).to(device)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if device != None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.concat = concat
        self.dino_weight = dino_weight

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, height, width, channel)

        quantization pipeline:

            1. get encoder input (B,H,W,C)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        
        z_flattened = z.view(-1, self.e_dim)    # [N,512]
        
        assert not torch.isnan(z_flattened).any()
        
        cb_normalized = self._normalize_cb(self.embedding.weight)
        d = self._d(cb_normalized, z_flattened) # [N, n_e]
        
        assert not torch.isnan(cb_normalized).any()
        assert not torch.isnan(d).any()

        # find closest encodings
        min_encoding_indices = torch.argmax(d, dim=1).unsqueeze(1)  # [N,1] 找到N个特征最相近的码本索引
        encoding_indices_prob = torch.softmax(d, dim=1) # [N,n_e]
        
        assert not torch.isnan(min_encoding_indices).any()
        assert not torch.isnan(encoding_indices_prob).any()
        
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)  # [N,n_e]的one-hot矩阵
        
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, cb_normalized).view(z.shape)  # [B, H, W, dim]
        assert not torch.isnan(z_q).any()
        
        # compute loss for embedding
        e_mean = torch.mean(min_encodings, dim=0)
        loss_kl = - torch.sum(e_mean * torch.log(1 / self.n_e / (e_mean + 1e-6)))
        loss, constrative_loss = self._loss(cb_normalized, min_encoding_indices, z_q, z)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # perplexity
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-6)))

        return loss, constrative_loss, loss_kl, encoding_indices_prob, d, z_q, perplexity, min_encodings, min_encoding_indices

    @torch.no_grad()
    def encode_vectors(
        self,
        x: torch.Tensor,                  # [B, C]
        return_indices: bool = False,
        return_probs: bool = False,
    ):
        """
        量化一批向量（[B,C]）→ 返回量化后向量（[B,C]）。
        不做分块；若 B 很大且 n_e 大，注意显存。
        """
        if x.dim() != 2:
            raise ValueError(f"encode_vectors expects [B,C], got {tuple(x.shape)}")
        B, C = x.shape
        if C != self.e_dim:
            raise ValueError(f"channel dim mismatch: got {C}, expected {self.e_dim}")

        x = x.to(self.device)

        # 归一化后的码本 [n_e, C]
        cb_norm = self._normalize_cb(self.embedding.weight)

        # 余弦相似度 [B, n_e]
        d = self._d(cb_norm, x)

        # 选中码字索引 [B]
        idx = torch.argmax(d, dim=1)

        # one-hot 取量化向量 [B,C]
        one_hot = F.one_hot(idx, num_classes=self.n_e).to(cb_norm.dtype)
        z_q = one_hot @ cb_norm

        # 可选返回
        out = (z_q,)
        if return_indices:
            out += (idx[:, None],)                 # [B,1]
        if return_probs:
            probs = torch.softmax(d, dim=1)        # [B, n_e]
            out += (probs,)

        return out[0] if len(out) == 1 else out

    def _d(self, cb, z_flattened):
        if self.concat:
            d_clip = self._cosine_sim(cb[:, :512], z_flattened[:, :512])
            d_dino = self._cosine_sim(cb[:, 512:], z_flattened[:, 512:])
            d = d_clip + self.dino_weight * d_dino
        else:
            d = self._cosine_sim(cb, z_flattened)
        return d
    
    def _loss(self, cb, min_encoding_indices, z_q, z):
        loss = 0
        constrative_loss = 0

        if self.concat:
            z_q_clip = z_q[:, :, :, :512]
            z_q_dino = z_q[:, :, :, 512:]
            loss_cos_clip = (1 - torch.mean(torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim = -1))) \
                            +  self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_clip.detach(), z[:, :, :, :512], dim = -1)))
            loss_cos_dino = (1 - torch.mean(torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim = -1))) \
                            + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_dino.detach(), z[:, :, :, 512:], dim = -1)))
            loss += loss_cos_clip + self.dino_weight * loss_cos_dino
            # constrative loss
            cb_clip = cb[...,:512]
            cb_dino = cb[...,512:]
            cb_clip_cos = torch.cosine_similarity(cb_clip.unsqueeze(0), cb_clip.unsqueeze(1), dim = -1)
            cb_dino_cos = torch.cosine_similarity(cb_dino.unsqueeze(0), cb_dino.unsqueeze(1), dim = -1)
            cb_cos = cb_clip_cos + self.dino_weight * cb_dino_cos   # [n_e,n_e]
            # mean of (cosine simlarity of (every feature and the other features))
            cb_neg = (torch.sum(cb_cos, dim=1) - cb_cos[0][0]) / (cb_cos.shape[0] - 1)
            x = F.embedding(min_encoding_indices, cb_neg[...,None]).squeeze()   # [N]
            
            zq_clip_cos = torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim = -1).view(-1)
            zq_dino_cos = torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim = -1).view(-1)
            zq_cos = zq_clip_cos + self.dino_weight * zq_dino_cos
            
            constrative_loss += torch.mean(-1 * zq_cos + x)
            
        else:
            loss += (1 - torch.mean(torch.cosine_similarity(z_q, z.detach(), dim = -1))) \
                    + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q.detach(), z, dim = -1)))
            # constrative_loss += 0
            cb_cos = torch.cosine_similarity(cb.unsqueeze(0), cb.unsqueeze(1), dim = -1)
            cb_neg = (torch.sum(cb_cos, dim=1) - cb_cos[0][0]) / (cb_cos.shape[0] - 1)
            x = F.embedding(min_encoding_indices, cb_neg[...,None]).squeeze()   # [N]
            zq_cos = torch.cosine_similarity(z_q, z.detach(), dim = -1).view(-1)
            constrative_loss += torch.mean(-1 * zq_cos + x)
        
        return loss, constrative_loss
    
    def _mse(self, embedding, z_flattened):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding**2, dim=1) - 2 * \
        torch.matmul(z_flattened, embedding.t())
        return d
    
    def _cosine_sim(self, embedding, z_flattened):
        # embedding: [128, dim], z_flattened: [N, dim]
        embedding_norm = torch.norm(embedding, dim=-1)[None, :] # [1,n_e]
        z_flattened_norm = torch.norm(z_flattened, dim=-1)[:, None] # [N,1]

        assert not torch.isnan(embedding).any()
        assert not torch.isnan(z_flattened).any()
        assert not torch.isnan(embedding_norm).any()

        assert not torch.isnan(z_flattened_norm).any()

        d = torch.matmul(z_flattened, embedding.t()) / (torch.matmul(z_flattened_norm, embedding_norm) + 1e-6)
        assert not torch.isnan(torch.matmul(z_flattened, embedding.t())).any()
        assert not torch.isnan(torch.matmul(z_flattened_norm, embedding_norm)).any()
        assert not torch.isnan(d).any()

        return d
    
    def _normalize_cb(self, cb):
        cb_normalized = cb / torch.norm(cb, p=2, dim=-1, keepdim=True)
        return cb_normalized