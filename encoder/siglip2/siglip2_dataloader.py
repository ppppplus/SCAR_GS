import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import open_clip
import json
from clip.clip_utils import to_tensor, get_image_paths
from clip.pyramid.feature_dataloader import FeatureDataloader
from siglip2.siglip2_utils import SIGLIP2NetWork
from clip.pyramid.mixture_embedding_dataloader import MixtureEmbeddingDataloader

class PyramidSigLIP2Dataset(Dataset):
    def __init__(self, path, cache_subdir: str = "siglip2_features", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = SIGLIP2NetWork(device=device)
        # self.model.eval()
        # self.model = self.model.to(self.device)

        # self.cache_dir = Path(path) / cache_subdir
        # self.cache_dir.mkdir(parents=True, exist_ok=True)

        # self.process = transforms.Compose(
        #         [
        #             transforms.Resize((224, 224)),
        #             transforms.Normalize(
        #                 mean=[0.48145466, 0.4578275, 0.40821073],
        #                 std=[0.26862954, 0.26130258, 0.27577711],
        #             ),
        #         ]
        #     )
        self.process = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.Normalize(
                        mean=[0.5,0.5,0.5],
                        std=[0.5,0.5,0.5],
                    ),
                ]
            )
        
        self.image_paths = get_image_paths(path)
        self.image_shape = to_tensor(Image.open(self.image_paths[0])).shape[1:3]

        self.cfg = {
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(self.image_shape),
                "model_name": "ViT-B/16",
            }
                
        self.mixture_path = Path(os.path.join(path, f"pyramid_siglip2_{self.cfg['tile_size_range'][0]}", "cache"))

        self._load_mixture()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def _get_images(self):
        image_list = []
        for image_path in tqdm(self.image_paths):
            image_list.append(to_tensor(Image.open(image_path)))
        return torch.stack(image_list)
    
    def _load_mixture(self):
        mixture_data = None
        if self.mixture_path.with_suffix(".npy").exists():
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
                embedding_size = 768
            )
        else:
            image_list = self._get_images()
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                image_list = image_list,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
                embedding_size = 768
            )
        self.data = mixture_data().to(self.device)

# class MixtureEmbeddingDataloader(FeatureDataloader):
#     def __init__(
#         self,
#         cfg: dict,
#         device: torch.device,
#         model,
#         image_list: torch.Tensor = None,
#         cache_path: str = None,
#     ):
#         assert "tile_size_range" in cfg
#         assert "tile_size_res" in cfg
#         assert "stride_scaler" in cfg
#         assert "image_shape" in cfg
#         assert "model_name" in cfg

#         self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
#         self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

#         self.model = model
#         # self.embed_size = self.model.embedding_dim
#         self.embed_size = 512
#         self.data_dict = {}
#         self.data = None
#         super().__init__(cfg, device, image_list, cache_path)

#     def __call__(self):
#         """
#             return patch level clip feature mixture.
#         """
#         return self.data

#     def _stride_scaler(self, tile_ratio, stride_scaler):
#         return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

#     def load(self):
#         # don't create anything, PatchEmbeddingDataloader will create itself
#         cache_info_path = self.cache_path.with_suffix(".info")
#         mix_cache_path = self.cache_path.with_suffix(".npy")

#         # check if cache exists
#         if not cache_info_path.exists():
#             raise FileNotFoundError

#         # if config is different, remove all cached content
#         with open(cache_info_path, "r") as f:
#             cfg = json.loads(f.read())
#         if cfg != self.cfg:
#             for f in os.listdir(self.cache_path):
#                 os.remove(os.path.join(self.cache_path, f))
#             raise ValueError("Config mismatch")

#         # load mixture
#         self.data = torch.from_numpy(np.load(mix_cache_path)).half()

#     def create(self, image_list):
#         os.makedirs(self.cache_path, exist_ok=True)
#         for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
#             stride_scaler = self.strider_scaler_list[i]
#             self.data_dict[i] = PatchEmbeddingDataloader(
#                 cfg={
#                     "tile_ratio": tr.item(),
#                     "stride_ratio": stride_scaler,
#                     "image_shape": self.cfg["image_shape"],
#                     "model_name": self.cfg["model_name"],
#                 },
#                 device=self.device,
#                 model=self.model,
#                 process=self.process,
#                 image_list=image_list,
#                 cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
#             )
#         # create mixture
#         self._create_mixture()


#     def save(self):
#         cache_info_path = self.cache_path.with_suffix(".info")
#         with open(cache_info_path, "w") as f:
#             f.write(json.dumps(self.cfg))
#         # don't save PatchEmbeddingDataloader, PatchEmbeddingDataloader will save itself
#         # save mixture
#         np.save(self.cache_path.with_suffix(".npy"), self.data)

#     def _create_mixture(self):
#         mix_feat = self.data_dict[0].data.detach().clone().permute(0, 3, 1, 2).float()  # [N, dim, hc, wc]
#         _, _, a, b = mix_feat.shape
#         for i in range(1, len(self.tile_sizes) - 1):
#             feat = self.data_dict[i].data.permute(0, 3, 1, 2).float()
#             feat_interp = F.interpolate(feat, size=(a, b), mode="nearest")
#             mix_feat += feat_interp
#         self.data = (mix_feat.permute(0, 2, 3, 1) / len(self.tile_sizes)).half()

# class PatchEmbeddingDataloader(FeatureDataloader):
#     def __init__(
#         self,
#         cfg: dict,
#         device: torch.device,
#         model,
#         image_list: torch.Tensor = None,
#         cache_path: str = None,
#     ):
#         assert "tile_ratio" in cfg
#         assert "stride_ratio" in cfg
#         assert "image_shape" in cfg
#         assert "model_name" in cfg

#         self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
#         self.stride = int(self.kernel_size * cfg["stride_ratio"])
#         self.padding = self.kernel_size // 2
#         self.center_x = (
#             (self.kernel_size - 1) / 2
#             - self.padding
#             + self.stride
#             * np.arange(
#                 np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
#             )
#         )
#         self.center_y = (
#             (self.kernel_size - 1) / 2
#             - self.padding
#             + self.stride
#             * np.arange(
#                 np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
#             )
#         )
#         self.center_x = torch.from_numpy(self.center_x).half()
#         self.center_y = torch.from_numpy(self.center_y).half()
#         self.start_x = self.center_x[0].float()
#         self.start_y = self.center_y[0].float()

#         self.model = model
#         # self.embed_size = self.model.embedding_dim
#         self.embed_size = 768
#         super().__init__(cfg, device, image_list, cache_path)

#     def load(self):
#         cache_info_path = self.cache_path.with_suffix( ".info")
#         if not cache_info_path.exists():
#             raise FileNotFoundError
#         with open(cache_info_path, "r") as f:
#             cfg = json.loads(f.read())
#         if cfg != self.cfg:
#             raise ValueError("Config mismatch")
#         self.data = torch.from_numpy(np.load(self.cache_path)).half()

#     def create(self, image_list):
#         assert self.model is not None, "model must be provided to generate features"
#         assert image_list is not None, "image_list must be provided to generate features"

#         unfold_func = torch.nn.Unfold(
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#         ).to(self.device)

#         img_embeds = []
#         for img in tqdm(image_list, desc="Embedding images", leave=False):
#             img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), unfold_func))
#         self.data = torch.from_numpy(np.stack(img_embeds)).half()

#     def __call__(self, img_points):
#         # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
#         # return: (B, 512)
#         img_points = img_points.cpu()
#         img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

#         x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
#         y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
#         return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

#     def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y):
#         img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
#         topleft = self.data[img_ind, x_ind, y_ind].to(self.device)
#         topright = self.data[img_ind, x_ind + 1, y_ind].to(self.device)
#         botleft = self.data[img_ind, x_ind, y_ind + 1].to(self.device)
#         botright = self.data[img_ind, x_ind + 1, y_ind + 1].to(self.device)

#         x_stride = self.stride
#         y_stride = self.stride
#         right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).to(self.device)  # .half()
#         top = torch.lerp(topleft, topright, right_w[:, None])
#         bot = torch.lerp(botleft, botright, right_w[:, None])

#         bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).to(self.device)  # .half()
#         return torch.lerp(top, bot, bot_w[:, None])

#     def _embed_siglip2_tiles(self, image, unfold_func):
#         # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
#         aug_imgs = torch.cat([image])

#         tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")
#         # [N,3,H,W]
#         with torch.no_grad():
#             clip_embeds = self.model.encode_image(self.process(tiles))
#         torch.cuda.empty_cache()
#         clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

#         clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
#         clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
#         clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
#         return clip_embeds.detach().cpu().numpy()