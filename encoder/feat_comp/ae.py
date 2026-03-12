"""
Borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, fdim):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(fdim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        for m in self.decoder:
            x = m(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

class VectorEncoder(nn.Module):
    def __init__(self, in_dim=768, latent_dim=6, width=256, depth=2, dropout=0.0):
        super().__init__()
        layers = [nn.LayerNorm(in_dim)]
        d_in = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(dropout)]
            d_in = width
        layers += [nn.Linear(d_in, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B,C]
        z = self.net(x)    # [B,D]
        return z

class VectorDecoder(nn.Module):
    def __init__(self, out_dim=768, latent_dim=6, width=256, depth=2, dropout=0.0):
        super().__init__()
        layers = [nn.LayerNorm(latent_dim)]
        d_in = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(dropout)]
            d_in = width
        layers += [nn.Linear(d_in, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):  # z: [B,D]
        x_hat = self.net(z)  # [B,C]
        return x_hat    

class VectorAE(nn.Module):
    def __init__(self, fdim, hdim, device, encoder_hidden_dims=[512, 256, 128, 64, 32, 3], decoder_hidden_dims=[16, 32, 64, 128, 256, 512, 768]):
        super().__init__()
        self.model = Autoencoder(encoder_hidden_dims=encoder_hidden_dims, decoder_hidden_dims=decoder_hidden_dims, fdim=fdim)
        self.fdim = fdim
        self.hdim = hdim
        self.device = device
        self.model = self.model.to(device)
    
    def forward(self, z):
        if z.dim() == 4 and z.shape[1] == self.fdim:  # [B, C, H, W]
            z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.fdim).to(self.device)
        outputs = self.model(z_flattened)
        # print(outputs.shape, data.shape)
        l2loss = l2_loss(outputs, z_flattened) 
        cosloss = cos_loss(outputs, z_flattened)
        loss = l2loss + cosloss * 0.001

        return outputs, l2loss, cosloss, loss
    
    def cosine_recon(self, a, b):
        a = F.normalize(a, dim=1, eps=1e-6)
        b = F.normalize(b, dim=1, eps=1e-6)
        return 1.0 - (a * b).sum(dim=1).mean()
    
    @torch.no_grad()
    def ende_vectors(
        self,
        x: torch.Tensor
    ):
        if x.dim() == 4:
            if x.shape[1] == self.fdim:  # [B, C, H, W]
                x = x.permute(0, 2, 3, 1).contiguous()
            b, h, w, _ = x.shape
            x_flattened = x.view(-1, self.fdim).to(self.device)
            outputs = self.model(x_flattened)
            outputs = outputs.view(b, h, w, -1)
        else:
            x_flattened = x.view(-1, self.fdim).to(self.device)
            outputs = self.model(x_flattened)
        return outputs
    
    @torch.no_grad()
    def encode_vectors(
        self,
        x: torch.Tensor
    ):
        if x.dim() == 4:
        # z_flattened = z.view(-1, self.fdim).to(self.device)
            if x.shape[1] == self.fdim:  # [B, C, H, W]
                x = x.permute(0, 2, 3, 1).contiguous()
            b, h, w, _ = x.shape
            x_flattened = x.view(-1, self.fdim).to(self.device)
            outputs = self.model.encode(x_flattened)
            outputs = outputs.view(b, h, w, -1)
        else:
            x_flattened = x.view(-1, self.fdim).to(self.device)
            outputs = self.model.encode(x_flattened)
        return outputs
