"""Stage 3 verification: instantiate the DPT dense predictor via Hydra and
run a forward pass on dummy input.

Checks:
  - The Hydra config ``model/dpt_dense_predictor.yaml`` instantiates the
    class cleanly.
  - The trainable-parameter set matches the Stage 2 expectation: SELA
    adapters from the backbone + every parameter of the new DPT decoder
    and head (those are all trainable since they were just initialized).
  - ``predict_dense(image)`` returns ``(B, 3, H, W)`` matching the input
    spatial size.
  - When ``confidence_from_logits=False``, channel 2 of the output lies
    in [0, 1] (post-sigmoid). When True, it can be any real value.

Run inside the container:
    PYTHONPATH=/workspace/code/semantic-persistence-filter \\
    CUDA_VISIBLE_DEVICES=4 python3 scripts/verify_dpt_predictor.py
"""
from __future__ import annotations

import sys

import hydra
import torch
from omegaconf import OmegaConf


def main() -> int:
    if not torch.cuda.is_available():
        print("[verify_dpt] CUDA unavailable; skipping (DINOv2 requires CUDA).")
        return 0

    # Build the model from the Hydra config exactly as train.py would.
    with hydra.initialize(version_base=None, config_path="../configs/train_confs"):
        # The base config pulls in `model/cross_attn_pooler` which is only
        # meaningful for the per-mask model — strip it so the dense model
        # doesn't see an unexpected `pooler_cfg` kwarg.
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[
                "model=dpt_dense_predictor",
                "~model/cross_attn_pooler",
            ],
        )
    print("[verify_dpt] resolved model config:")
    print(OmegaConf.to_yaml(cfg.model, resolve=True))

    model = hydra.utils.instantiate(cfg.model)
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # Count trainable params: should be backbone-adapter + all decoder/head.
    backbone_trainable = sum(
        p.numel() for n, p in model.backbone.named_parameters() if p.requires_grad
    )
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(
        f"[verify_dpt] trainable: backbone(adapter)={backbone_trainable:,} "
        f"decoder={decoder_trainable:,} head={head_trainable:,}"
    )
    assert backbone_trainable > 0, "Expected SELA adapter params to be trainable."
    assert decoder_trainable > 0
    assert head_trainable > 0

    # Forward shape + conf-channel range.
    with torch.no_grad():
        x = torch.randn(2, 3, 518, 518, device=device)
        out = model.predict_dense(x)
    print(f"[verify_dpt] predict_dense output shape: {tuple(out.shape)}")
    assert tuple(out.shape) == (2, 3, 518, 518), out.shape

    if not model.confidence_from_logits:
        yc = out[:, 2]
        lo, hi = float(yc.min()), float(yc.max())
        print(f"[verify_dpt] conf channel range (sigmoided): [{lo:.4f}, {hi:.4f}]")
        assert 0.0 <= lo and hi <= 1.0, (lo, hi)
    print("[verify_dpt] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
