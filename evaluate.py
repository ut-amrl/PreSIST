#!/usr/bin/env python3
"""
evaluate_hydra.py

Changes vs previous version:
1) Loads checkpoint using the SAME checkpoint utilities as training:
     training.checkpointing.load_training_state_from_checkpoint(...)
   (instead of model.load_state_dict)

2) Output path includes checkpoint epoch (read from the checkpoint meta):
     <run_root>/eval/epoch_<EPOCH>/<split>/

Other behavior:
- run_root inferred from ckpt path:
    <run_root>/checkpoints/<ckpt>.pth  -> run_root = parent(checkpoints)
- Evaluates train/val/test splits (skips a split if dataset can't be created).
- Saves metrics.json per split + optional overlay PNGs per split.
- Writes summary.json under <run_root>/eval/epoch_<EPOCH>/summary.json

Run (Hydra-style, same as training):
  python evaluate_hydra.py ckpt=/path/to/<run_root>/checkpoints/ckpt_best.pth

Optional overlay saving:
  python evaluate_hydra.py ckpt=... eval.viz=true eval.viz_max=32 eval.num_workers=8
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader

# Project imports (same as training)
from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId

from training.validator import SurvivalValidationRunner, ValConfig
from training.dataset.distillation_dataset import DistillationDatasetConfig

from training.checkpointing import load_training_state_from_checkpoint

import math
from collections import defaultdict

import numpy as np



# ----------------------------
# Batch helpers
# ----------------------------
def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _unpack_batch(batch: Union[Dict[str, Any], Tuple[Any, ...]]) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) >= 5:
        return {"image": batch[0], "mask": batch[1], "y50": batch[2], "yr": batch[3], "yc": batch[4]}
    raise ValueError(
        "Unsupported batch format. Expected dict with keys "
        "('image','mask','y50','yr','yc') or tuple (image, mask, y50, yr, yc, ...)."
    )


def _safe_float(v: Any) -> float:
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


# ----------------------------
# Dataloaders (same pattern as training)
# ----------------------------
def _make_loader(
    distillation_cfg: DistillationDatasetConfig,
    registry_cfg: RegistryConfig,
    *,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    relpath_to_image_id = RelpathToImageId(
        image_root=distillation_cfg.input_root_dir,
        registry_cfg=registry_cfg,
    )
    image_id_to_relpath = ImageIdToRelpath(
        image_root=distillation_cfg.input_root_dir,
        registry_cfg=registry_cfg,
        warn=print,
    )

    dataset = distillation_cfg.make_dataset(
        split=split,
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=img_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ----------------------------
# Visualization (save-to-file only)
# ----------------------------
def _tensor_to_uint8_img(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)  # HWC
    x = x.detach().float().cpu()
    mn = float(x.min())
    mx = float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    return x


def _mask_to_uint8(m: torch.Tensor) -> torch.Tensor:
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m[0]
    m = m.detach().float().cpu().clamp(0, 1)
    return (m * 255.0).to(torch.uint8)


def _save_overlay(
    out_path: Union[str, Path],
    image_chw: torch.Tensor,
    mask_hw_or_chw: torch.Tensor,
    text_lines: Tuple[str, ...],
    *,
    alpha: float = 0.45,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img_u8 = _tensor_to_uint8_img(image_chw)  # HWC
    m_u8 = _mask_to_uint8(mask_hw_or_chw)     # HW

    img = Image.fromarray(img_u8.numpy())
    if img.mode != "RGB":
        img = img.convert("RGB")

    m = Image.fromarray(m_u8.numpy(), mode="L")

    red = Image.new("RGB", img.size, (255, 0, 0))
    red_masked = Image.composite(red, Image.new("RGB", img.size, (0, 0, 0)), m)

    blended = Image.blend(img, Image.composite(red_masked, img, m), alpha=alpha)

    draw = ImageDraw.Draw(blended)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 6
    text = "\n".join(text_lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x0, y0 = 8, 8
    draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad], fill=(0, 0, 0))
    draw.multiline_text((x0, y0), text, fill=(255, 255, 255), font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(str(out_path))


def _as_1d_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().view(-1)
    return x.numpy()


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x_sorted, y in [0,1] for empirical CDF."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / float(xs.size)
    return xs, ys


def _plot_step_hist_overlay(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    out_path: Path,
    *,
    title: str,
    bins: int = 60,
    log_x: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    true_vals = true_vals[np.isfinite(true_vals)]
    pred_vals = pred_vals[np.isfinite(pred_vals)]

    if true_vals.size == 0 or pred_vals.size == 0:
        return

    # If log_x, we plot on log-scale x, but still compute bins in linear-x space
    # using positive-only data. (t50/t90 should be positive anyway.)
    if log_x:
        true_vals = true_vals[true_vals > 0]
        pred_vals = pred_vals[pred_vals > 0]
        if true_vals.size == 0 or pred_vals.size == 0:
            return

        xmin = min(true_vals.min(), pred_vals.min())
        xmax = max(true_vals.max(), pred_vals.max())
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin <= 0 or xmax <= xmin:
            return
        bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
    else:
        xmin = min(true_vals.min(), pred_vals.min())
        xmax = max(true_vals.max(), pred_vals.max())
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmax <= xmin:
            return
        bin_edges = np.linspace(xmin, xmax, bins + 1)

    # Density hist -> step lines
    h_true, _ = np.histogram(true_vals, bins=bin_edges, density=True)
    h_pred, _ = np.histogram(pred_vals, bins=bin_edges, density=True)

    # Convert bin edges + heights into step-plot coordinates
    # We'll use bin centers for a simple line, and step style for clarity.
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure()
    plt.step(centers, h_true, where="mid", label="dataset")
    plt.step(centers, h_pred, where="mid", label="pred")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    if log_x:
        plt.xscale("log")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_cdf_overlay(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    out_path: Path,
    *,
    title: str,
    log_x: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)

    # For log_x, only positive values make sense
    if log_x:
        true_vals = true_vals[np.isfinite(true_vals) & (true_vals > 0)]
        pred_vals = pred_vals[np.isfinite(pred_vals) & (pred_vals > 0)]
    else:
        true_vals = true_vals[np.isfinite(true_vals)]
        pred_vals = pred_vals[np.isfinite(pred_vals)]

    if true_vals.size == 0 or pred_vals.size == 0:
        return

    xt, yt = _ecdf(true_vals)
    xp, yp = _ecdf(pred_vals)
    if xt.size == 0 or xp.size == 0:
        return

    plt.figure()
    plt.plot(xt, yt, label="dataset")
    plt.plot(xp, yp, label="pred")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("CDF")
    plt.ylim(0.0, 1.0)
    if log_x:
        plt.xscale("log")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pred_vs_true_scatter(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    out_path: Path,
    *,
    title: str,
    log_xy: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    true_vals = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    pred_vals = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    if true_vals.size == 0 or pred_vals.size == 0:
        return

    # Keep aligned true/pred pairs even if lengths diverge unexpectedly.
    n = min(true_vals.size, pred_vals.size)
    true_vals = true_vals[:n]
    pred_vals = pred_vals[:n]

    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if log_xy:
        mask &= (true_vals > 0) & (pred_vals > 0)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]
    if true_vals.size == 0:
        return

    vmin = min(float(true_vals.min()), float(pred_vals.min()))
    vmax = max(float(true_vals.max()), float(pred_vals.max()))
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return
    if vmax <= vmin:
        eps = 1e-6 if vmin == 0.0 else max(abs(vmin) * 1e-6, 1e-12)
        vmin -= eps
        vmax += eps
        if log_xy and vmin <= 0:
            vmin = max(min(float(true_vals.min()), float(pred_vals.min())), 1e-12)
            vmax = max(vmax, vmin * 1.01)

    line_x = np.array([vmin, vmax], dtype=np.float64)

    plt.figure()
    plt.scatter(true_vals, pred_vals, s=10, alpha=0.35, label="samples")
    plt.plot(line_x, line_x, "k--", linewidth=1.2, label="ideal (y=x)")
    plt.title(title)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    if log_xy:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_calibration_curve(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    out_path: Path,
    *,
    title: str,
    n_bins: int = 10,
    log_xy: bool = False,
) -> None:
    """Bin true values by quantile, plot mean(pred|bin) vs mean(true|bin).

    A flat curve near the global mean(true) indicates regression-to-mean.
    A curve tracking the identity line indicates the model differentiates
    inputs.
    """
    import matplotlib.pyplot as plt

    true_vals = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    pred_vals = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    n = min(true_vals.size, pred_vals.size)
    if n == 0:
        return
    true_vals = true_vals[:n]
    pred_vals = pred_vals[:n]

    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if log_xy:
        mask &= (true_vals > 0) & (pred_vals > 0)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]
    if true_vals.size < 2:
        return

    # Quantile-based bin edges; dedup in case of ties.
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(true_vals, qs))
    if edges.size < 2:
        return

    bin_idx = np.clip(np.searchsorted(edges, true_vals, side="right") - 1, 0, edges.size - 2)

    bin_true_mean = []
    bin_pred_mean = []
    bin_pred_sem = []
    for b in range(edges.size - 1):
        m = bin_idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        bt = float(true_vals[m].mean())
        bp = float(pred_vals[m].mean())
        bs = float(pred_vals[m].std(ddof=1)) / np.sqrt(cnt) if cnt > 1 else 0.0
        bin_true_mean.append(bt)
        bin_pred_mean.append(bp)
        bin_pred_sem.append(bs)

    if not bin_true_mean:
        return

    bin_true_mean = np.asarray(bin_true_mean, dtype=np.float64)
    bin_pred_mean = np.asarray(bin_pred_mean, dtype=np.float64)
    bin_pred_sem = np.asarray(bin_pred_sem, dtype=np.float64)

    global_true_mean = float(true_vals.mean())
    vmin = float(min(bin_true_mean.min(), bin_pred_mean.min(), global_true_mean))
    vmax = float(max(bin_true_mean.max(), bin_pred_mean.max(), global_true_mean))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return

    line_x = np.array([vmin, vmax], dtype=np.float64)

    plt.figure()
    plt.errorbar(
        bin_true_mean,
        bin_pred_mean,
        yerr=bin_pred_sem,
        fmt="o-",
        capsize=3,
        label="bin mean(pred)",
    )
    plt.plot(line_x, line_x, "k--", linewidth=1.2, label="ideal (y=x)")
    plt.axhline(global_true_mean, color="r", linestyle=":", linewidth=1.0,
                label=f"global mean(true)={global_true_mean:.3g}")
    plt.title(title)
    plt.xlabel("bin mean(true)")
    plt.ylabel("bin mean(pred)")
    if log_xy:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _compute_regression_diagnostics(
    dists: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Per-variable diagnostics that pin down regression-to-mean behavior.

    Returns dict keyed by variable name. Each entry contains:
      pearson_r, spearman_rho, spread_ratio (std(pred)/std(true)),
      r2_vs_mean (1 - MSE/Var(true)), and basic moment sanity stats.
    """
    out: dict[str, dict[str, float]] = {}
    for var, tp in dists.items():
        true_vals = np.asarray(tp["true"], dtype=np.float64).reshape(-1)
        pred_vals = np.asarray(tp["pred"], dtype=np.float64).reshape(-1)
        n = min(true_vals.size, pred_vals.size)
        true_vals = true_vals[:n]
        pred_vals = pred_vals[:n]
        mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]

        entry: dict[str, float] = {"n": float(true_vals.size)}
        if true_vals.size < 2:
            out[var] = entry
            continue

        mu_t = float(true_vals.mean())
        mu_p = float(pred_vals.mean())
        sd_t = float(true_vals.std(ddof=1))
        sd_p = float(pred_vals.std(ddof=1))
        entry["true_mean"] = mu_t
        entry["pred_mean"] = mu_p
        entry["true_std"] = sd_t
        entry["pred_std"] = sd_p
        entry["spread_ratio"] = float(sd_p / sd_t) if sd_t > 0 else float("nan")

        # Pearson via np.corrcoef (avoid scipy dependency).
        if sd_t > 0 and sd_p > 0:
            entry["pearson_r"] = float(np.corrcoef(true_vals, pred_vals)[0, 1])
        else:
            entry["pearson_r"] = float("nan")

        # Spearman: Pearson on ranks. Use scipy if available, else manual ranks.
        try:
            from scipy.stats import spearmanr  # type: ignore
            rho, _ = spearmanr(true_vals, pred_vals)
            entry["spearman_rho"] = float(rho) if rho is not None else float("nan")
        except Exception:
            rt = np.argsort(np.argsort(true_vals)).astype(np.float64)
            rp = np.argsort(np.argsort(pred_vals)).astype(np.float64)
            if rt.std() > 0 and rp.std() > 0:
                entry["spearman_rho"] = float(np.corrcoef(rt, rp)[0, 1])
            else:
                entry["spearman_rho"] = float("nan")

        mse = float(np.mean((pred_vals - true_vals) ** 2))
        var_t = float(true_vals.var(ddof=0))
        entry["mse"] = mse
        entry["r2_vs_mean"] = float(1.0 - mse / var_t) if var_t > 0 else float("nan")

        out[var] = entry
    return out


@torch.no_grad()
def _collect_split_distributions(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_from_logits: bool,
    use_amp: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns:
      {
        "y50": {"true": np.ndarray, "pred": np.ndarray},
        "yr":  {"true": ..., "pred": ...},
        "yc":  {"true": ..., "pred": ...},
        "t50": {"true": ..., "pred": ...},
        "t90": {"true": ..., "pred": ...},
      }
    """
    model.eval()
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    use_amp = bool(use_amp and device.type == "cuda")

    buf: dict[str, dict[str, list[np.ndarray]]] = {
        k: {"true": [], "pred": []} for k in ("y50", "yr", "yc", "t50", "t90")
    }

    for batch in loader:
        b = _unpack_batch(batch)
        b = _to_device(b, device)

        image = b["image"]
        mask = b["mask"]
        y50_true = b["y50"]
        yr_true = b["yr"]
        yc_true = b["yc"].float()

        with autocast_ctx(enabled=use_amp):
            y50_hat, yr_hat, yc_hat = model(image, mask)
            yc_pred = torch.sigmoid(yc_hat) if confidence_from_logits else yc_hat

        # Derived times (compute in float for stability under AMP)
        t50_true = torch.exp(y50_true.float())
        t50_pred = torch.exp(y50_hat.float())
        t90_true = t50_true * (1.0 + torch.exp(yr_true.float()))
        t90_pred = t50_pred * (1.0 + torch.exp(yr_hat.float()))

        buf["y50"]["true"].append(_as_1d_numpy(y50_true))
        buf["y50"]["pred"].append(_as_1d_numpy(y50_hat))

        buf["yr"]["true"].append(_as_1d_numpy(yr_true))
        buf["yr"]["pred"].append(_as_1d_numpy(yr_hat))

        buf["yc"]["true"].append(_as_1d_numpy(yc_true))
        buf["yc"]["pred"].append(_as_1d_numpy(yc_pred))

        buf["t50"]["true"].append(_as_1d_numpy(t50_true))
        buf["t50"]["pred"].append(_as_1d_numpy(t50_pred))

        buf["t90"]["true"].append(_as_1d_numpy(t90_true))
        buf["t90"]["pred"].append(_as_1d_numpy(t90_pred))

    out: dict[str, dict[str, np.ndarray]] = {}
    for k in buf.keys():
        out[k] = {
            "true": np.concatenate(buf[k]["true"], axis=0) if buf[k]["true"] else np.array([], dtype=np.float64),
            "pred": np.concatenate(buf[k]["pred"], axis=0) if buf[k]["pred"] else np.array([], dtype=np.float64),
        }
    return out


def _save_distribution_plots(
    dists: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    *,
    bins: int = 60,
    log_x_for_times: bool = True,
) -> None:
    dist_dir = out_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    for var, tp in dists.items():
        true_vals = tp["true"]
        pred_vals = tp["pred"]

        log_x = bool(log_x_for_times and var in ("t50", "t90"))

        _plot_step_hist_overlay(
            true_vals,
            pred_vals,
            dist_dir / f"hist_{var}.png",
            title=f"{var}: step-hist density (dataset vs pred)",
            bins=bins,
            log_x=log_x,
        )
        _plot_cdf_overlay(
            true_vals,
            pred_vals,
            dist_dir / f"cdf_{var}.png",
            title=f"{var}: CDF (dataset vs pred)",
            log_x=log_x,
        )
        _plot_pred_vs_true_scatter(
            true_vals,
            pred_vals,
            dist_dir / f"scatter_{var}.png",
            title=f"{var}: true vs pred",
            log_xy=log_x,
        )
        _plot_calibration_curve(
            true_vals,
            pred_vals,
            dist_dir / f"calib_{var}.png",
            title=f"{var}: calibration (bin mean true vs mean pred)",
            n_bins=10,
            log_xy=log_x,
        )



@torch.no_grad()
def _save_some_overlays(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_images: int,
    confidence_from_logits: bool,
    use_amp: bool,
) -> int:
    if max_images <= 0:
        return 0

    model.eval()
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    use_amp = bool(use_amp and device.type == "cuda")

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in loader:
        b = _unpack_batch(batch)
        b = _to_device(b, device)

        # TODO figure out how to skip transformations for visualization (e.g., want to visualize original image/mask rather than transformed version)
        image = b["image"]
        mask = b["mask"]
        y50 = b["y50"]
        yr = b["yr"]
        yc = b["yc"].float()

        with autocast_ctx(enabled=use_amp):
            y50_hat, yr_hat, yc_hat = model(image, mask)
            yc_prob = torch.sigmoid(yc_hat) if confidence_from_logits else yc_hat

        # --- NEW: compute t50 / t90 (in float for numeric stability under AMP) ---
        t50_hat = torch.exp(y50_hat.float())
        t50_true = torch.exp(y50.float())
        t90_hat = t50_hat * (1.0 + torch.exp(yr_hat.float()))
        t90_true = t50_true * (1.0 + torch.exp(yr.float()))
        # ------------------------------------------------------------------------

        bs = int(image.shape[0])
        for i in range(bs):
            if saved >= max_images:
                return saved

            lines = (
                # NEW lines shown first
                f"t50 true: {float(t50_true[i].detach().cpu()):.4f} | pred: {float(t50_hat[i].detach().cpu()):.4f}",
                f"t90 true: {float(t90_true[i].detach().cpu()):.4f} | pred: {float(t90_hat[i].detach().cpu()):.4f}",
                # Existing lines
                f"y50 true: {float(y50[i].detach().cpu()):.4f} | pred: {float(y50_hat[i].detach().cpu()):.4f}",
                f"yr  true: {float(yr[i].detach().cpu()):.4f} | pred: {float(yr_hat[i].detach().cpu()):.4f}",
                f"yc  true: {float(yc[i].detach().cpu()):.4f} | pred: {float(yc_prob[i].detach().cpu()):.4f}",
            )

            out_path = viz_dir / f"overlay_{saved:04d}.png"
            _save_overlay(out_path, image[i], mask[i], lines)
            saved += 1

        if saved >= max_images:
            break

    return saved



# @torch.no_grad()
# def _save_some_overlays(
#     *,
#     model: nn.Module,
#     loader: DataLoader,
#     device: torch.device,
#     out_dir: Path,
#     max_images: int,
#     confidence_from_logits: bool,
#     use_amp: bool,
# ) -> int:
#     if max_images <= 0:
#         return 0

#     model.eval()
#     autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
#     use_amp = bool(use_amp and device.type == "cuda")

#     viz_dir = out_dir / "viz"
#     viz_dir.mkdir(parents=True, exist_ok=True)

#     saved = 0
#     for batch in loader:
#         b = _unpack_batch(batch)
#         b = _to_device(b, device)

#         image = b["image"]
#         mask = b["mask"]
#         y50 = b["y50"]
#         yr = b["yr"]
#         yc = b["yc"].float()

#         with autocast_ctx(enabled=use_amp):
#             y50_hat, yr_hat, yc_hat = model(image, mask)
#             yc_prob = torch.sigmoid(yc_hat) if confidence_from_logits else yc_hat

#         bs = int(image.shape[0])
#         for i in range(bs):
#             if saved >= max_images:
#                 return saved

#             lines = (
#                 f"y50 true: {float(y50[i].detach().cpu()):.4f} | pred: {float(y50_hat[i].detach().cpu()):.4f}",
#                 f"yr  true: {float(yr[i].detach().cpu()):.4f} | pred: {float(yr_hat[i].detach().cpu()):.4f}",
#                 f"yc  true: {float(yc[i].detach().cpu()):.4f} | pred: {float(yc_prob[i].detach().cpu()):.4f}",
#             )

#             out_path = viz_dir / f"overlay_{saved:04d}.png"
#             _save_overlay(out_path, image[i], mask[i], lines)
#             saved += 1

#         if saved >= max_images:
#             break

#     return saved


# ----------------------------
# Split evaluation
# ----------------------------
def _evaluate_split(
    *,
    split: str,
    model: nn.Module,
    validator: SurvivalValidationRunner,
    distillation_cfg: DistillationDatasetConfig,
    registry_cfg: RegistryConfig,
    train_cfg_obj: Any,
    device: torch.device,
    out_root_epoch: Path,
    dist_bins: int = 60,
    viz: bool,
    viz_max: int,
    num_workers: int,
) -> Optional[Dict[str, Any]]:
    img_size = int(getattr(train_cfg_obj, "img_size", 518))
    batch_size = int(getattr(train_cfg_obj, "batch_size", 8))

    try:
        loader = _make_loader(
            distillation_cfg,
            registry_cfg,
            split=split,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except Exception as e:
        print(f"[Eval] Split '{split}' could not be created ({e}). Skipping.")
        return None

    model.eval()
    agg_score, metrics = validator.validate(model, loader)

    split_dir = out_root_epoch / split
    split_dir.mkdir(parents=True, exist_ok=True)

    dist_plots_saved = False



    vcfg = getattr(train_cfg_obj, "validationConfig", ValConfig())
    if isinstance(vcfg, dict):
        vcfg = ValConfig(**vcfg)


    dists = _collect_split_distributions(
        model=model,
        loader=loader,
        device=device,
        confidence_from_logits=bool(getattr(vcfg, "confidence_from_logits", False)),
        use_amp=bool(getattr(vcfg, "use_amp", False)),
    )
    _save_distribution_plots(
        dists,
        split_dir,
        bins=dist_bins,
        log_x_for_times=True,
    )

    diagnostics = _compute_regression_diagnostics(dists)
    diag_path = split_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[Eval] {split:>5} regression diagnostics -> {diag_path}")

    overlays_saved = 0
    if viz:
        overlays_saved = _save_some_overlays(
            model=model,
            loader=loader,
            device=device,
            out_dir=split_dir,
            max_images=viz_max,
            confidence_from_logits=bool(getattr(vcfg, "confidence_from_logits", False)),
            use_amp=bool(getattr(vcfg, "use_amp", False)),
        )

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "aggregate_score": float(agg_score),
        "metrics": {k: _safe_float(v) for k, v in metrics.items()},
        "regression_diagnostics": diagnostics,
        "overlays_saved": int(overlays_saved),
        "batch_size": batch_size,
        "img_size": img_size,
    }

    metrics_path = split_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Eval] {split:>5} agg={float(agg_score):.6f} -> {metrics_path}")
    if viz:
        print(f"[Eval] {split:>5} overlays_saved={overlays_saved} -> {split_dir / 'viz'}")

    return payload


# ----------------------------
# Hydra entry (mirrors training usage)
# ----------------------------
@hydra.main(version_base=None, config_path="configs/train_confs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name

    if "ckpt" not in cfg or cfg.ckpt is None:
        raise ValueError("Missing cfg.ckpt. Provide ckpt=/path/to/checkpoints/<file>.pth via Hydra overrides.")

    ckpt_path = Path(str(cfg.ckpt)).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # run_root = parent of the checkpoint's directory (i.e., parent of ".../checkpoints")
    run_root = ckpt_path.parent.parent
    print(f"[Paths] ckpt:     {ckpt_path}")
    print(f"[Paths] run_root: {run_root}")
    print(f"[Config] name:    {config_name}")

    # Instantiate config objects exactly like training
    train_cfg_obj = hydra.utils.instantiate(cfg.trainer.loop)
    registry_cfg: RegistryConfig = hydra.utils.instantiate(cfg.registry_info)
    distillation_cfg: DistillationDatasetConfig = hydra.utils.instantiate(cfg.eval_dataset)

    # Device similar to training
    device_str = getattr(train_cfg_obj, "device", None)
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Instantiate model/loss/optimizer/scheduler (like training utilities do)
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    loss_fn: nn.Module = hydra.utils.instantiate(cfg.trainer.loss_function)
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler, optimizer=optimizer)

    model.to(device)
    loss_fn.to(device)

    # Validation config + consistency checks (like training)
    val_cfg = getattr(train_cfg_obj, "validationConfig", ValConfig())
    if isinstance(val_cfg, dict):
        val_cfg = ValConfig(**val_cfg)

    if hasattr(loss_fn, "confidence_from_logits") and hasattr(val_cfg, "confidence_from_logits"):
        if bool(loss_fn.confidence_from_logits) != bool(val_cfg.confidence_from_logits):
            raise ValueError(
                f"Mismatch in confidence_from_logits: loss_fn={loss_fn.confidence_from_logits} "
                f"val_cfg={val_cfg.confidence_from_logits}"
            )
        if hasattr(model, "confidence_from_logits"):
            if bool(model.confidence_from_logits) != bool(loss_fn.confidence_from_logits):
                raise ValueError(
                    f"Mismatch in confidence_from_logits: model={model.confidence_from_logits} "
                    f"loss_fn={loss_fn.confidence_from_logits}"
                )

    # Load checkpoint using training utility
    scaler = torch.cuda.amp.GradScaler(enabled=(getattr(train_cfg_obj, "use_amp", False) and device.type == "cuda"))
    meta = load_training_state_from_checkpoint(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        ckpt=str(ckpt_path),
        map_location=device,
        strict_model=False,
        load_optimizer=False,
        load_scheduler=False,
        load_scaler=False,
    )
    ckpt_epoch = int(getattr(meta, "epoch", -1))
    ckpt_global_step = int(getattr(meta, "global_step", -1))
    print(f"[CKPT] loaded epoch={ckpt_epoch} global_step={ckpt_global_step}")

    # Output root: epoch_<N>/eval/<eval_dataset>/<eval_timestamp>/
    # A new timestamp per run prevents collision when re-evaluating the same checkpoint.
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root_epoch = run_root / f"epoch_{ckpt_epoch:04d}" / "eval" / cfg.eval_dataset.name / eval_timestamp
    out_root_epoch.mkdir(parents=True, exist_ok=True)

    # Validator
    validator = SurvivalValidationRunner(cfg=val_cfg, loss_fn=loss_fn, device=device)

    # Optional eval toggles via cfg.eval.*
    # viz = bool(getattr(getattr(cfg, "eval", {}), "viz", False)) if hasattr(cfg, "eval") else False
    viz = True
    # viz_max = int(getattr(getattr(cfg, "eval", {}), "viz_max", 16)) if hasattr(cfg, "eval") else 16
    viz_max = 50
    num_workers = int(getattr(getattr(cfg, "eval", {}), "num_workers", 4)) if hasattr(cfg, "eval") else 4

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_name": config_name,
        "ckpt": str(ckpt_path),
        "run_root": str(run_root),
        "epoch": ckpt_epoch,
        "global_step": ckpt_global_step,
        "device": str(device),
        "splits": {},
    }

    splits_cfg = getattr(cfg, "eval_splits", None)
    if splits_cfg is None:
        splits = ("test", "train", "val")
    else:
        splits = tuple(str(s) for s in splits_cfg)
    print(f"[Eval] splits: {splits}")

    for split in splits:
        payload = _evaluate_split(
            split=split,
            model=model,
            validator=validator,
            distillation_cfg=distillation_cfg,
            registry_cfg=registry_cfg,
            train_cfg_obj=train_cfg_obj,
            device=device,
            out_root_epoch=out_root_epoch,
            viz=viz,
            viz_max=viz_max,
            num_workers=num_workers,
        )
        if payload is not None:
            results["splits"][split] = payload

    summary_path = out_root_epoch / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Eval] Summary -> {summary_path}")


if __name__ == "__main__":
    main()

"""
Example usage

Note: ckpt must be passed with a leading + because it is not defined in the config struct.

Evaluate best checkpoint:
  python evaluate.py \
    --config-name cross_attn_random_instwise_manlabel_2026_02_11_14_02_80_10_10 \
    +ckpt=/workspace/data/output/training_outputs/<train_dataset>/<model>/<timestamp>/checkpoints/ckpt_best.pth

Evaluate a specific epoch checkpoint (epoch 0 = pre-training, epoch N = after N epochs):
  python evaluate.py \
    --config-name cross_attn_random_instwise_manlabel_2026_02_11_14_02_80_10_10 \
    +ckpt=.../checkpoints/ckpt_epoch_0_pretrain.pth

  python evaluate.py \
    --config-name cross_attn_random_instwise_manlabel_2026_02_11_14_02_80_10_10 \
    +ckpt=.../checkpoints/ckpt_epoch_1.pth

Evaluate on a different dataset than training:
  python evaluate.py \
    --config-name cross_attn_random_instwise_manlabel_2026_02_11_14_02_80_10_10 \
    eval_dataset=ahg_eval \
    +ckpt=.../checkpoints/ckpt_best.pth

Output is written to:
  <checkpoint_dir_parent>/epoch_<N>/eval/<eval_dataset>/<eval_timestamp>/
    train/metrics.json
    val/metrics.json
    summary.json
"""