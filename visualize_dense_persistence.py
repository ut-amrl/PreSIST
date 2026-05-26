"""Render dense persistence-time heatmaps from a trained DPT-dense checkpoint.

For each input image the script writes one PNG per quantile in
`viz.quantiles.plot` (p25/p50/p75 by default), plus one PNG per horizon in
`viz.horizons.hours` (or `cfg.eval_scenario.query_horizons` when not set)
showing `P(persist > h)` overlaid on the RGB.

Survival probability for the horizon plots uses
`WeibullSurvivalFunction(k, lam)` per pixel — equivalent to invoking
`AbsPersistenceFilter.predict(..., observations=None)` per the filter base
contract (persistence_eval/methods/filters/base.py).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import FuncNorm, LogNorm, Normalize
from omegaconf import DictConfig, OmegaConf

import persistence_eval.utils.hydra_resolvers  # noqa: F401  (registers ${dot_to_p:…} etc.)
from persistence_eval.methods.inferencers.learned_model import _LABEL_TRANSFORM_KEYS

matplotlib.use("Agg")


_QUANTILE_TRANSFORM = "training.dataset.distillation_dataset.label_transform_log_ratio_p25_p75"


# ---------- model loading ----------

def _load_dense_model_and_meta(cfg: DictConfig, device: torch.device):
    """Instantiate the DPT-dense model from cfg.inferencer.model, load the
    checkpoint, and return (model, trainer_cfg DictConfig, img_size)."""
    from training.checkpointing import load_model_from_checkpoint

    # The dpt-dense inferencer config wires model via `defaults: - /model: …`,
    # which places the composed model under the top-level cfg.model (the leading
    # slash escapes the inferencer scope). Fall back to cfg.inferencer.model
    # for inferencer configs that compose the model locally.
    model_cfg = cfg.get("model") if "model" in cfg else cfg.inferencer.model
    model = hydra.utils.instantiate(model_cfg)
    model.to(device).eval()
    ckpt_meta = load_model_from_checkpoint(
        model, cfg.inferencer.checkpoint_path, map_location=device, strict=True,
    )

    trainer_cfg = None
    trainer_cfg_yaml = getattr(ckpt_meta, "trainer_cfg", None)
    if trainer_cfg_yaml:
        trainer_cfg = OmegaConf.create(trainer_cfg_yaml)

    img_size = 518
    if trainer_cfg is not None:
        val = OmegaConf.select(trainer_cfg, "trainer.loop.img_size")
        if val is not None:
            img_size = int(val)

    _assert_p25_p75_head(trainer_cfg)
    return model, trainer_cfg, img_size


def _assert_p25_p75_head(trainer_cfg) -> None:
    if trainer_cfg is None:
        raise RuntimeError(
            "Checkpoint is missing its embedded trainer_cfg — cannot verify the "
            "label-transform head. visualize_dense_persistence.py requires a "
            "checkpoint trained with label_transform_log_ratio_p25_p75."
        )
    target = OmegaConf.select(trainer_cfg, "train_dataset.label_transform._target_")
    if target != _QUANTILE_TRANSFORM:
        raise RuntimeError(
            f"Checkpoint head is {target!r}; this script only supports "
            f"{_QUANTILE_TRANSFORM!r}. Re-train with the p25/p75 transform or "
            "pick a different checkpoint."
        )
    # Belt-and-braces: keep the inferencer's table the source of truth.
    if _QUANTILE_TRANSFORM not in _LABEL_TRANSFORM_KEYS:
        raise RuntimeError(
            "_LABEL_TRANSFORM_KEYS no longer contains the p25_p75 entry; the "
            "inferencer code has changed in an incompatible way."
        )


def _resolve_norm_stats(trainer_cfg) -> dict:
    if trainer_cfg is None:
        return {}
    keys = ("y50_mean", "y50_std", "yr_mean", "yr_std")
    values = {k: OmegaConf.select(trainer_cfg, f"train_dataset.{k}") for k in keys}
    if all(v is None for v in values.values()):
        return {}
    if any(v is None for v in values.values()):
        raise ValueError(
            f"Checkpoint trainer_cfg has a partial target-norm stat set: {values}."
        )
    return {k: float(v) for k, v in values.items()}


# ---------- image sampling ----------

def _instantiate_dataset_views(cfg: DictConfig):
    """Return (image_mask_view_or_None, track_view_or_None). Both are lazily
    composed against the same backing dataset, matching the orchestrator."""
    backing = hydra.utils.instantiate(cfg.benchmark_dataset, _recursive_=False)
    image_view = (
        hydra.utils.instantiate(cfg.benchmark_dataset.image_mask_view, backing=backing)
        if hasattr(cfg.benchmark_dataset, "image_mask_view")
        else None
    )
    track_view = (
        hydra.utils.instantiate(cfg.benchmark_dataset.track_view, backing=backing)
        if hasattr(cfg.benchmark_dataset, "track_view")
        else None
    )
    return image_view, track_view


def _iter_input_images(cfg: DictConfig, image_view) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (stem, image_chw_uint8) pairs for each image we should visualize."""
    image_path = cfg.viz.input.get("image_path")
    if image_path:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        yield Path(image_path).stem, tensor
        return

    if image_view is None:
        raise RuntimeError(
            "benchmark_dataset has no image_mask_view and no viz.input.image_path "
            "was provided — nothing to visualize."
        )
    groups = list(image_view.iter_image_groups())
    num = cfg.viz.input.get("num_samples")
    if num is not None and num < len(groups):
        rng = random.Random(cfg.viz.input.get("seed", 0))
        groups = rng.sample(groups, int(num))
    for i, group in enumerate(groups):
        path = getattr(group, "image_path", None)
        stem = Path(path).stem if path else f"sample_{i:04d}"
        yield stem, group.image


# ---------- horizons ----------

def _resolve_horizons(cfg: DictConfig, track_view) -> list[float]:
    hours = cfg.viz.horizons.get("hours")
    if hours is not None:
        return [float(h) for h in hours]
    scenario = cfg.get("eval_scenario")
    if scenario is None:
        return []
    qh = OmegaConf.select(scenario, "query_horizons")
    if qh is not None:
        return [float(h) for h in qh]
    try:
        # Most providers in persistence_eval/sampling need a track_dataset to
        # compute their horizon list (e.g. percentile providers derive it from
        # the track persistence-time distribution).
        kwargs = {"track_dataset": track_view} if track_view is not None else {}
        provider = hydra.utils.instantiate(scenario, **kwargs)
    except Exception as exc:
        print(f"[viz] could not instantiate eval_scenario to read horizons: {exc!r}")
        return []
    for attr in ("query_horizons", "_query_horizons"):
        if hasattr(provider, attr):
            # Providers store horizons in seconds (see _log_query_horizons in
            # persistence_eval/sampling/episode_provider.py); the dense model's
            # p25/p75 head is in hours, so convert here.
            return [float(h) / 3600.0 for h in getattr(provider, attr)]
    print(
        f"[viz] eval_scenario provider {type(provider).__name__} has no "
        "(query_horizons|_query_horizons) attribute; skipping prob plots."
    )
    return []


# ---------- per-pixel decoding ----------

def _decode_quantile_and_prob_grids(
    raw: torch.Tensor,
    norm_stats: dict,
    quantiles: list[float],
    horizons_hours: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """raw: (1, 3, H, W) network output. Returns:
       - tq:   (Q, H, W) hours for each requested quantile
       - sh:   (H_horizons, H, W) P(persist > h) for each requested horizon.
       Either tensor may be empty if the corresponding list is empty.
    """
    if raw.dim() != 4 or raw.shape[0] != 1 or raw.shape[1] != 3:
        raise ValueError(f"expected (1, 3, H, W) network output; got {tuple(raw.shape)}")
    y_low = raw[0, 0]
    y_ratio = raw[0, 1]
    if norm_stats:
        y_low = y_low * norm_stats["y50_std"] + norm_stats["y50_mean"]
        y_ratio = y_ratio * norm_stats["yr_std"] + norm_stats["yr_mean"]

    eps = 1e-30
    t25 = torch.exp(y_low).clamp(min=eps)
    t75 = (t25 * (torch.exp(y_ratio) + 1.0)).clamp(min=eps)

    # Per-pixel Weibull fit from (t25, t75). Vectorized version of
    # weibull_median_from_quantiles (persistence_eval/methods/survival/weibull.py).
    log_t25 = torch.log(t25)
    log_t75 = torch.log(t75)
    a25 = math.log(-math.log1p(-0.25))
    a75 = math.log(-math.log1p(-0.75))
    # Guard against degenerate t75 ≈ t25 (would blow k up).
    denom = (log_t75 - log_t25).clamp(min=1e-6)
    k_grid = (a75 - a25) / denom
    log_lam = log_t25 - a25 / k_grid

    # Quantile heatmaps.
    if quantiles:
        a_p = torch.tensor(
            [math.log(-math.log1p(-float(p))) for p in quantiles],
            device=raw.device,
            dtype=raw.dtype,
        )  # (Q,)
        tq = torch.exp(log_lam.unsqueeze(0) + a_p.view(-1, 1, 1) / k_grid.unsqueeze(0))  # (Q, H, W)
    else:
        tq = torch.empty(0, *log_lam.shape, device=raw.device, dtype=raw.dtype)

    # Survival heatmaps S(h) = exp(-(h/lam)^k). Equivalent to invoking the
    # persistence filter with observations=None (filter base contract).
    if horizons_hours:
        h = torch.tensor(horizons_hours, device=raw.device, dtype=raw.dtype)  # (H_h,)
        # log(h) - log(lam): (H_h, H, W); raise to k: per-pixel k.
        log_h = torch.log(h.clamp(min=eps)).view(-1, 1, 1)
        sh = torch.exp(-torch.exp(k_grid.unsqueeze(0) * (log_h - log_lam.unsqueeze(0))))
    else:
        sh = torch.empty(0, *log_lam.shape, device=raw.device, dtype=raw.dtype)

    return tq, sh


# ---------- rendering ----------

def _make_quantile_norm(kind: str, vmin: float, vmax: float):
    if kind == "log":
        return LogNorm(vmin=vmin, vmax=vmax)
    if kind == "linear":
        return Normalize(vmin=vmin, vmax=vmax)
    if kind == "loglog":
        def fwd(x):
            x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
            return np.log1p(np.log1p(x))
        def inv(u):
            u = np.asarray(u, dtype=np.float64)
            return np.expm1(np.expm1(u))
        return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax)
    raise ValueError(f"unknown viz.quantiles.norm: {kind!r}")


def _resize_to_image(grid: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """grid: (N, H, W). Resize to target_hw with bilinear interpolation."""
    if grid.numel() == 0:
        return grid
    if grid.shape[-2:] == target_hw:
        return grid
    return F.interpolate(grid.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


def _save_overlay(rgb_np: np.ndarray, heat_np: np.ndarray, *, norm, cmap: str,
                  alpha: float, title: str, cbar_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(rgb_np.shape[1] / 100, rgb_np.shape[0] / 100), dpi=100)
    ax.imshow(rgb_np)
    im = ax.imshow(heat_np, cmap=cmap, norm=norm, alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_combined_quantile_grid(
    rgb_np: np.ndarray, tq_np: np.ndarray, quantiles: list[float], *,
    norm, cmap: str, alpha: float, out_path: Path,
) -> None:
    n = len(quantiles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, p, hg in zip(axes, quantiles, tq_np):
        ax.imshow(rgb_np)
        im = ax.imshow(hg, cmap=cmap, norm=norm, alpha=alpha)
        ax.set_title(f"p{int(round(p * 100))}")
        ax.axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("persistence (hours)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------- main pipeline ----------

@hydra.main(config_path="configs/train_confs", config_name="dense_viz", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting dense persistence visualization with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False))

    device = torch.device(cfg.viz.get("device", cfg.runtime.device))
    out_dir = Path(cfg.viz.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, trainer_cfg, img_size = _load_dense_model_and_meta(cfg, device)
    norm_stats = _resolve_norm_stats(trainer_cfg)
    from training.dataset.distillation_dataset import make_default_dinov2_transform
    transform = make_default_dinov2_transform(img_size)

    image_view, track_view = _instantiate_dataset_views(cfg)
    quantiles = [float(q) for q in cfg.viz.quantiles.plot]
    horizons = _resolve_horizons(cfg, track_view)
    print(f"[viz] quantiles={quantiles}; horizons (hours)={horizons}")

    q_norm = _make_quantile_norm(
        cfg.viz.quantiles.norm,
        float(cfg.viz.quantiles.vmin_hours),
        float(cfg.viz.quantiles.vmax_hours),
    )
    prob_norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = cfg.viz.cmap
    alpha = float(cfg.viz.alpha)

    for stem, img_chw_u8 in _iter_input_images(cfg, image_view):
        rgb_np = img_chw_u8.permute(1, 2, 0).numpy()
        H, W = rgb_np.shape[:2]

        net_in = transform(img_chw_u8).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model.predict_dense(net_in)  # (1, 3, h, w)
        raw = raw.float().cpu()

        tq, sh = _decode_quantile_and_prob_grids(raw, norm_stats, quantiles, horizons)
        tq = _resize_to_image(tq, (H, W)) if tq.numel() else tq
        sh = _resize_to_image(sh, (H, W)) if sh.numel() else sh

        tq_np = tq.numpy() if tq.numel() else None
        sh_np = sh.numpy() if sh.numel() else None

        # Sanity asserts (cheap, catch sign / parameter bugs).
        if tq_np is not None and len(quantiles) >= 2:
            for i in range(len(quantiles) - 1):
                if not (tq_np[i] <= tq_np[i + 1] + 1e-3).all():
                    print(f"[viz][warn] {stem}: quantile {quantiles[i]} > {quantiles[i+1]} at some pixels")
        if sh_np is not None and len(horizons) >= 2:
            order = np.argsort(horizons)
            sorted_sh = sh_np[order]
            for i in range(len(order) - 1):
                if not (sorted_sh[i] >= sorted_sh[i + 1] - 1e-3).all():
                    print(f"[viz][warn] {stem}: S(h) not non-increasing between horizons "
                          f"{horizons[order[i]]} and {horizons[order[i+1]]}")

        # Per-quantile overlays.
        if tq_np is not None:
            for p, hg in zip(quantiles, tq_np):
                tag = f"p{int(round(p * 100))}"
                _save_overlay(
                    rgb_np, hg, norm=q_norm, cmap=cmap, alpha=alpha,
                    title=f"{stem}  {tag}",
                    cbar_label="persistence (hours)",
                    out_path=out_dir / f"{stem}_{tag}.png",
                )
            _save_combined_quantile_grid(
                rgb_np, tq_np, quantiles,
                norm=q_norm, cmap=cmap, alpha=alpha,
                out_path=out_dir / f"{stem}_pquantiles.png",
            )

        # Per-horizon survival overlays.
        if sh_np is not None:
            for h, hg in zip(horizons, sh_np):
                tag = f"horizon_{h:g}h"
                _save_overlay(
                    rgb_np, hg, norm=prob_norm, cmap=cmap, alpha=alpha,
                    title=f"{stem}  P(persist > {h:g}h)",
                    cbar_label="P(persist > h)",
                    out_path=out_dir / f"{stem}_{tag}.png",
                )

        print(f"[viz] wrote outputs for {stem} → {out_dir}")


if __name__ == "__main__":
    main()
