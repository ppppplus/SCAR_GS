"""
Borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vq_model import VQModel
import math

class VectorEncoder(nn.Module):
    def __init__(self, in_dim=768, latent_dim=6, width=256, depth=2, dropout=0.0):
        super().__init__()

        layers = [nn.LayerNorm(in_dim)]
        d_in = in_dim

        for _ in range(depth):
            layers += [
                nn.Linear(d_in, width),
                nn.SiLU(),
                nn.LayerNorm(width), 
                nn.Dropout(dropout),
            ]
            d_in = width

        layers += [nn.Linear(d_in, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class VectorDecoder(nn.Module):
    def __init__(self, out_dim=768, latent_dim=6, width=256, depth=2, dropout=0.0):
        super().__init__()
        layers = [nn.LayerNorm(latent_dim)]
        d_in = latent_dim
        for _ in range(depth):
            layers += [
                nn.Linear(d_in, width),
                nn.SiLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout)
            ]
            d_in = width

        layers += [nn.Linear(d_in, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim).to(device)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        bound = 1 / math.sqrt(e_dim)
        self.embedding.weight.data.uniform_(-bound, bound)

        if device != None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        z_flattened = z.view(-1, self.e_dim)    # [N,fdim]
        
        assert not torch.isnan(z_flattened).any()
        
        # cb = self._normalize_cb(self.embedding.weight)

        # cb = self.embedding.weight
        cb = F.normalize(self.embedding.weight, dim=-1)
        d = self._d(cb, z_flattened) # [N, n_e]
        
        assert not torch.isnan(cb).any()
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
        z_q = torch.matmul(min_encodings, cb).view(z.shape)  # [B, H, W, dim]
        assert not torch.isnan(z_q).any()
        
        # commitment loss
        commit_loss = self._loss(cb, min_encoding_indices, z_q, z)

        # kl_loss
        e_mean = torch.mean(min_encodings, dim=0)                   # [n_e]
        p = encoding_indices_prob.mean(dim=0)
        kl_loss = torch.sum(p * (torch.log(p) - math.log(1.0 / self.n_e)))
        
        # load_balancing_loss
        flattened_encoding_indices_prob = encoding_indices_prob.view(-1, self.n_e)
        load_balancing_loss =  (e_mean * torch.mean(flattened_encoding_indices_prob, dim=0)).sum()

        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # perplexity
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-6)))

        return z_q, commit_loss, kl_loss, load_balancing_loss, cb, perplexity

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

        # 码本 [n_e, C]
        # cb_norm = self._normalize_cb(self.embedding.weight)
        cb = self.embedding.weight

        # 余弦相似度 [B, n_e]
        d = self._d(cb, x)

        # 选中码字索引 [B]
        idx = torch.argmax(d, dim=1)

        # one-hot 取量化向量 [B,C]
        one_hot = F.one_hot(idx, num_classes=self.n_e).to(cb.dtype)
        z_q = one_hot @ cb

        # 可选返回
        out = (z_q,)
        if return_indices:
            out += (idx[:, None],)                 # [B,1]
        if return_probs:
            probs = torch.softmax(d, dim=1)        # [B, n_e]
            out += (probs,)

        return out[0] if len(out) == 1 else out

    # def _d(self, cb, z_flattened):
    #     d = self._cosine_sim(cb, z_flattened)
    #     return d
    def _d(self, cb, z):
        # """
        # cb: codebook, [n_e, e_dim]
        # z:  input features, [N, e_dim]
        # return: pairwise L2 distance, [N, n_e]
        # """
        # # z^2
        # z_norm = (z ** 2).sum(dim=1, keepdim=True)       # [N, 1]
        # # cb^2
        # cb_norm = (cb ** 2).sum(dim=1)                   # [n_e]
        # # z·cb^T -> [N, n_e]
        # dot = z @ cb.t()
        # # ||z - c||^2 = |z|^2 + |c|^2 - 2 z·c
        # dist = z_norm + cb_norm - 2 * dot
        # return dist
        cb_norm = F.normalize(cb, dim=-1)
        z_norm  = F.normalize(z,  dim=-1)
        return torch.matmul(z_norm, cb_norm.t())
    
    def _loss(self, cb, min_encoding_indices, z_q, z):
        loss = 0
        loss += (1 - torch.mean(torch.cosine_similarity(z_q, z.detach(), dim = -1))) \
                + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q.detach(), z, dim = -1)))
        # loss += torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # constrative_loss += 0
        cb_cos = torch.cosine_similarity(cb.unsqueeze(0), cb.unsqueeze(1), dim = -1)
        cb_neg = (torch.sum(cb_cos, dim=1) - cb_cos[0][0]) / (cb_cos.shape[0] - 1)
        # x = F.embedding(min_encoding_indices, cb_neg[...,None]).squeeze()   # [N]
        # zq_cos = torch.cosine_similarity(z_q, z.detach(), dim = -1).view(-1)
        return loss
    
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
    

class VectorQuantizerAE(nn.Module):
    def __init__(self, fdim, n_e, e_dim, beta, device):
        super().__init__()
        self.enc = VectorEncoder(in_dim=fdim, latent_dim=e_dim).to(device)
        self.dec = VectorDecoder(out_dim=fdim, latent_dim=e_dim).to(device)
        self.vq = VectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, device=device).to(device)
        
        self.fdim = fdim
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.device = device
    
    def forward(self, z):

        z_flattened = z.view(-1, self.fdim).to(self.device)
        h = self.enc(z_flattened)
        # h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        h = F.normalize(h, dim=-1)
        h_q, commit_loss, kl_loss, load_balancing_loss, cb, perplexity = self.vq(h)
        z_hat = self.dec(h_q)
        # print(perplexity)
        # rec_loss = F.mse_loss(z_hat, z)
        rec_loss = self.cosine_recon(z_hat, z_flattened) + 0.001*F.mse_loss(z_hat, z_flattened)
        return z_hat, rec_loss, commit_loss, kl_loss, load_balancing_loss, cb, h, perplexity
    
    def cosine_recon(self, a, b):
        a = F.normalize(a, dim=1, eps=1e-6)
        b = F.normalize(b, dim=1, eps=1e-6)
        return 1.0 - (a * b).sum(dim=1).mean()
    
    @torch.no_grad()
    def ende_vectors(
        self,
        x: torch.Tensor
    ):
        if x.dim != 2:
            B, N, D = x.shape
            assert D == self.fdim, f"Expected feature dim {self.fdim}, got {D}"
            x_flattened = x.reshape(-1, D).to(self.device)
            h = self.enc(x_flattened)
            h_q, commit_loss, loss_kl, \
                encoding_indices_prob, min_encoding_indices, perplexity = self.vq(h)
            x_flattened_hat = self.dec(h_q)  # [B*N, fdim]
            x_hat = x_flattened_hat.view(B, N, D)
        else:
            h = self.enc(x)
            h_q, commit_loss, loss_kl, encoding_indices_prob, min_encoding_indices, perplexity = self.vq(h)
            x_hat = self.dec(h_q)
        return x_hat
    
    @torch.no_grad()
    def encode_vectors(
        self,
        x: torch.Tensor
    ):
        if x.dim != 2:
            B, N, D = x.shape
            assert D == self.fdim, f"Expected feature dim {self.fdim}, got {D}"
            x_flattened = x.reshape(-1, D).to(self.device)
            h = self.enc(x_flattened)
            h_q, commit_loss, loss_kl, \
                encoding_indices_prob, min_encoding_indices, perplexity = self.vq(h)
        else:
            h = self.enc(x)
            h_q, commit_loss, loss_kl, encoding_indices_prob, min_encoding_indices, perplexity = self.vq(h)
        return h, h_q