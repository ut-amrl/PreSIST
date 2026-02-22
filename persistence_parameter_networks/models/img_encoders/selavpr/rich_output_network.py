import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .backbone.vision_transformer import vit_large  # your DinoVisionTransformer ctor

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class Flatten(nn.Module):
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class LocalAdapt(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.upconv2(x)
        return x


def _infer_token_grid_hw(
    patch_tokens: torch.Tensor,
    img_hw: tuple[int, int] | None,
    patch_size: int | None,
) -> tuple[int, int]:
    """
    Infer (H, W) of the patch token grid.
    Prefer using image size & patch_size if available; otherwise fall back to sqrt(N).
    """
    B, N, C = patch_tokens.shape

    if img_hw is not None and patch_size is not None:
        H_img, W_img = img_hw
        H = H_img // patch_size
        W = W_img // patch_size
        if H * W == N:
            return H, W

    # Fallback: assume square grid
    H = int(math.isqrt(N))
    if H * H != N:
        raise ValueError(
            f"Cannot infer token grid: N={N} is not a perfect square and "
            f"(img_hw, patch_size) did not match. Provide img_hw and patch_size."
        )
    return H, H


def _extract_dino_tokens(backbone_out):
    """
    Robustly extract (cls_token, patch_tokens) from a DinoVisionTransformer output.

    Many DINO-style implementations return a dict containing:
      - "x_norm_clstoken": (B, C)
      - "x_norm_patchtokens": (B, N, C)

    Some return a single tensor of tokens (B, 1+N, C) where tokens[:,0] is CLS.
    """
    if isinstance(backbone_out, dict):
        patch = backbone_out.get("x_norm_patchtokens", None)
        cls = backbone_out.get("x_norm_clstoken", None)

        # Some forks expose raw tokens under different keys
        if patch is None:
            tok = backbone_out.get("x_norm_tokens", None) or backbone_out.get("tokens", None)
            if tok is not None and tok.dim() == 3 and tok.shape[1] >= 2:
                cls = tok[:, 0]
                patch = tok[:, 1:]
        if patch is None:
            raise KeyError(
                "Backbone dict output did not include x_norm_patchtokens (or compatible tokens). "
                f"Available keys: {list(backbone_out.keys())}"
            )
        return cls, patch

    if torch.is_tensor(backbone_out):
        # Assume (B, 1+N, C) tokens
        if backbone_out.dim() != 3 or backbone_out.shape[1] < 2:
            raise ValueError(f"Unexpected tensor output from backbone: {tuple(backbone_out.shape)}")
        cls = backbone_out[:, 0]
        patch = backbone_out[:, 1:]
        return cls, patch

    if isinstance(backbone_out, (tuple, list)) and len(backbone_out) > 0:
        # Some APIs return (tokens, ...) – try first element
        return _extract_dino_tokens(backbone_out[0])

    raise TypeError(f"Unsupported backbone output type: {type(backbone_out)}")


class GeoLocalizationNet(nn.Module):
    """
    DINO-compatible output for object-conditioned cross-attention over full-image tokens.

    Returns dict:
      - patch_tokens:     (B, N, C)   keys/values for cross-attention
      - patch_tokens_hw:  (B, H, W, C) grid view for mask->token alignment
      - patch_token_map:  (B, C, H, W) map view (optional convenience)
      - cls_token:        (B, C) or None
      - grid_hw:          (H, W)

      Optional (if use_local=True):
      - local_map:        (B, 128, Hl, Wl)
      - local_tokens_hw:  (B, Hl, Wl, 128)
      - local_hw:         (Hl, Wl)
    """

    def __init__(self, args, use_local: bool = True, return_global_only: bool = False):
        super().__init__()
        self.backbone = get_backbone(args)
        self.LocalAdapt = LocalAdapt() if use_local else None
        self.features_dim = int(getattr(args, "features_dim", 1024))

        # These help infer grid shape when N isn't square
        self.patch_size = getattr(self.backbone, "patch_size", None)
        # DINO impls vary: img_size might be int or tuple
        img_size = getattr(self.backbone, "img_size", None)
        if isinstance(img_size, int):
            self.img_hw = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.img_hw = (int(img_size[0]), int(img_size[1]))
        else:
            # fallback to args.image_size if present
            im = getattr(args, "image_size", None)
            if isinstance(im, int):
                self.img_hw = (im, im)
            elif isinstance(im, (tuple, list)) and len(im) == 2:
                self.img_hw = (int(im[0]), int(im[1]))
            else:
                self.img_hw = None

        self.return_global_only = return_global_only

        if self.return_global_only:
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(work_with_tokens=None),
                Flatten(),
            )


    def forward(self, x: torch.Tensor):
        backbone_out = self.backbone(x)

        cls_token, patch_tokens = _extract_dino_tokens(backbone_out)  # cls may be None
        # patch_tokens: (B, N, C)
        B, N, C = patch_tokens.shape

        H, W = _infer_token_grid_hw(patch_tokens, self.img_hw, self.patch_size)

        patch_tokens_hw = patch_tokens.view(B, H, W, C)         # (B, H, W, C)
        patch_token_map = patch_tokens_hw.permute(0, 3, 1, 2)   # (B, C, H, W)

        # Return global feature only
        if self.return_global_only:
            x1 = self.aggregation(patch_token_map)       # (B, C)
            global_feature = F.normalize(x1, p=2, dim=-1) # (B, C)
            return global_feature

        result = {
            "patch_tokens": patch_tokens,          # (B, N, C) -> K/V
            "patch_tokens_hw": patch_tokens_hw,    # (B, H, W, C) -> mask alignment
            "patch_token_map": patch_token_map,    # (B, C, H, W) -> pooling convenience
            "cls_token": cls_token,                # (B, C) or None
            "grid_hw": (H, W),
        }



        # Optional: local high-res map (useful later for ring/support pooling etc.)
        if self.LocalAdapt is not None:
            local_map = self.LocalAdapt(patch_token_map)          # (B, 128, Hl, Wl)
            local_map = F.normalize(local_map, p=2, dim=1)
            local_tokens_hw = local_map.permute(0, 2, 3, 1)       # (B, Hl, Wl, 128)

            result.update({
                "local_map": local_map,
                "local_tokens_hw": local_tokens_hw,
                "local_hw": (local_map.shape[-2], local_map.shape[-1]),
            })

        return result


def get_backbone(args):
    print("Get backbone: use adapter: ", args.use_adapter)
    if args.registers:
        backbone = vit_large(
            patch_size=14,
            img_size=args.image_size,
            init_values=1,
            block_chunks=0,
            num_register_tokens=4,
            use_adapter=args.use_adapter,
        )
    else:
        backbone = vit_large(
            patch_size=14,
            img_size=args.image_size,
            init_values=1,
            block_chunks=0,
            use_adapter=args.use_adapter,
        )

    assert not (args.foundation_model_path is None and args.resume is None), (
        "Please specify foundation model path."
    )
    if args.foundation_model_path:
        model_dict = backbone.state_dict()
        state_dict = torch.load(args.foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)

    args.features_dim = 1024
    return backbone
