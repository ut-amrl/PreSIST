"""Verification for Stage 2 of dense-prediction work.

Two checks:
  1. ``DenseRichOutputSelaEncoder`` exposes exactly the same trainable
     parameter set as the existing ``RichOutputSelaEncoder`` under the
     same args — i.e. SELA adapters are preserved as trainable and
     everything else stays frozen.
  2. ``DenseRichOutputSelaEncoder(x)`` returns 4 intermediate token maps
     of shape ``(B, 1024, 37, 37)`` for a 518x518 input on ViT-L/14.

Run inside the project container:
    PYTHONPATH=/workspace/code/semantic-persistence-filter \\
        python3 scripts/verify_dense_encoder.py
"""
from __future__ import annotations

import sys

import torch

from persistence_parameter_networks.models.img_encoders.selavpr.dense.dense_rich_img_encoder import (
    DenseRichOutputSelaEncoder,
)
from persistence_parameter_networks.models.img_encoders.selavpr.rich_img_encoder import (
    RichOutputSelaEncoder,
    RichOutputSelaEncoderArgs,
)


def _trainable_param_names(module) -> set[str]:
    return {n for n, p in module.named_parameters() if p.requires_grad}


def _strip_top_namespace(names: set[str]) -> set[str]:
    """The two encoders' parameter trees are identical from
    ``self.model.<...>`` onward, but DenseGeoLocalizationNet has no
    LocalAdapt submodule, so we ignore any param paths that only exist
    on one side. The fair comparison is the backbone params."""
    return {n for n in names if not n.startswith("model.LocalAdapt")}


def main() -> int:
    args = RichOutputSelaEncoderArgs(
        foundation_model_path="pretrained_models/dinov2_vitl14_pretrain.pth",
        registers=False,
        resume=False,
        image_size=518,
        freeze_dino=True,
        use_adapter=True,
        pretrained_adapter_path=None,
        freeze_pre_sela=False,
        use_local=False,
    )

    print("[verify] building RichOutputSelaEncoder (existing)...")
    orig = RichOutputSelaEncoder(args)
    print("[verify] building DenseRichOutputSelaEncoder (new)...")
    dense = DenseRichOutputSelaEncoder(args)

    orig_trainable = _strip_top_namespace(_trainable_param_names(orig))
    dense_trainable = _strip_top_namespace(_trainable_param_names(dense))

    only_in_orig = orig_trainable - dense_trainable
    only_in_dense = dense_trainable - orig_trainable

    print(f"[verify] trainable params: orig={len(orig_trainable)} dense={len(dense_trainable)}")
    if only_in_orig or only_in_dense:
        print(f"[verify] MISMATCH:")
        print(f"  only in orig:  {sorted(only_in_orig)[:20]}")
        print(f"  only in dense: {sorted(only_in_dense)[:20]}")
        return 1
    print("[verify] PASS: trainable parameter set matches between encoders.")

    # All trainable params should contain "adapter" in their name (because
    # freeze_dino=True freezes everything else).
    non_adapter = {n for n in dense_trainable if "adapter" not in n}
    if non_adapter:
        print(f"[verify] WARN: trainable non-adapter params: {sorted(non_adapter)[:20]}")
    else:
        print("[verify] PASS: every trainable param is an adapter param.")

    # Forward pass: shapes — DINOv2 blocks use xformers memory-efficient
    # attention which is CUDA-only, so we must run the forward on GPU.
    if not torch.cuda.is_available():
        print("[verify] CUDA unavailable; skipping forward-shape check.")
        return 0
    device = torch.device("cuda")
    dense = dense.to(device)
    dense.eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 518, 518, device=device)
        out = dense(x)

    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    maps = out["intermediate_token_maps"]
    assert isinstance(maps, list) and len(maps) == 4, (
        f"expected 4 intermediate maps, got {len(maps) if hasattr(maps, '__len__') else maps}"
    )
    for i, m in enumerate(maps):
        expected = (2, 1024, 37, 37)
        assert tuple(m.shape) == expected, (
            f"intermediate_token_maps[{i}] shape={tuple(m.shape)}, expected {expected}"
        )
    print(f"[verify] PASS: intermediate_token_maps shapes all (2, 1024, 37, 37).")
    print(f"[verify] grid_hw = {out['grid_hw']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
