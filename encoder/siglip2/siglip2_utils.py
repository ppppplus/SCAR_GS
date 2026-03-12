import torch
from PIL import Image

from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
class SIGLIP2NetWork:
    def __init__(self, device="cuda",
                 local_model_path="local-dir:/home/ubuntu/Documents/TJH/model_zoo/siglip2/timm-ViT-B-16-SigLIP2-512"):
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained(model_name=local_model_path, device=device)
        # self.process = transforms.Compose(
        #         [
        #             transforms.Resize((512, 512)),
        #             transforms.Normalize(
        #                 mean=[0.5,0.5,0.5],
        #                 std=[0.5,0.5,0.5],
        #             ),
        #         ]
        #     )
        self.tokenizer = get_tokenizer(local_model_path)
        self.model.eval()

    @torch.no_grad()
    def encode_image_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor_chw_uint8: [3, H, W], uint8 CPU
        # pil = image_tensor_chw_uint8
        pil = Image.fromarray(image_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8"))
        img = self.preprocess(pil).unsqueeze(0).to(self.device) # [1,3,512,512]
        feat = self.model.encode_image(img, normalize=True)  # [1, C], already L2-normalized
        return feat.squeeze(0).detach().cpu()                # [C] CPU
    
    @torch.no_grad()
    def encode_image_patch(self, image_tensor):
        pil = Image.fromarray(image_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8"))
        img = self.preprocess(pil).unsqueeze(0).to(self.device)  # [1,3,512,512]
        feat_map = self.model.visual.trunk.forward_features(img)     # [1, 1024, 768]
        feat = self.model.visual.trunk.forward_head(feat_map, pre_logits=True)  # [1,768]
        return feat_map.squeeze(0).detach().cpu(), feat.squeeze(0).detach().cpu()

    @torch.no_grad()
    def encode_image(self,
                 input: torch.Tensor,
                 batch_size: int = 64,
                 use_amp: bool = True) -> torch.Tensor:
        """
        input: [3,H,W] 或 [N,3,H,W]，应已满足模型的预处理规范（大小/归一化）
        return: 若输入是单张 -> [C]；若输入是批量 -> [N,C]
        """
        # 统一成 [N,3,H,W]
        single = (input.dim() == 3)
        if single:
            input = input.unsqueeze(0)

        # 分批送 GPU 编码
        feats = []
        amp_dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16)
        for s in range(0, input.size(0), batch_size):
            batch = input[s:s + batch_size].to(self.device, non_blocking=True)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    f = self.model.encode_image(batch)      # [b, C]
            else:
                f = self.model.encode_image(batch)

            # L2 归一化（可选，但大多数 CLIP/SigLIP 下游都会这么做）
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)

            feats.append(f.float().cpu())
            del batch, f
            torch.cuda.empty_cache()

        out = torch.cat(feats, dim=0)  # [N, C]
        return out[0] if single else out

    @torch.no_grad()
    def extract_text_features(self, text: str) -> torch.Tensor:
        text = f"a {text}"
        text_input = self.tokenizer([text], context_length=self.model.context_length).to(self.device)
        text_feat = self.model.encode_text(text_input, normalize=True)
        return text_feat       

    @torch.no_grad()
    def match_features(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        text_prob = torch.sigmoid(img_feat.float() @ text_feat.T * self.model.logit_scale.exp() + self.model.logit_bias)
        return text_prob 