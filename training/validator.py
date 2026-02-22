from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


from utils.metrics import AverageMeter


@dataclass
class ValConfig:
    # If confidence head outputs logits, set True (recommended for BCEWithLogits)
    confidence_from_logits: bool = False

    # If confidence head outputs probabilities and you compute BCE, clamp for stability
    clamp_confidence_probs: bool = True
    eps: float = 1e-6

    # Optional extra metrics
    compute_original_time_metrics: bool = True
    compute_correlations: bool = True

    # Mixed precision
    use_amp: bool = False


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
        return {
            "image": batch[0],
            "mask": batch[1],
            "y50": batch[2],
            "yr": batch[3],
            "yc": batch[4],
        }
    raise ValueError(
        "Unsupported batch format. Expected dict with keys "
        "('image','mask','y50','yr','yc') or tuple (image, mask, y50, yr, yc, ...)."
    )


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std(unbiased=False) * y.std(unbiased=False)).clamp_min(eps)
    return float(((x * y).mean() / denom).cpu().item())


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    # Tie handling is approximate (argsort-based), but good enough for tracking.
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    rx = torch.empty_like(x)
    ry = torch.empty_like(y)
    rx[torch.argsort(x)] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    ry[torch.argsort(y)] = torch.arange(y.numel(), device=y.device, dtype=torch.float32)
    return _pearson_corr(rx, ry)


class SurvivalValidationRunner:
    """
    Validation runner for a model that:
      model(image, mask) -> (y50_hat, yr_hat, yc_hat)
    and a val_loader that yields (image, mask, y50, yr, yc) (dict or tuple).

    loss_fn is expected to have signature:
      loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc) -> scalar loss
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[ValConfig] = None,
    ):
        self.loss_fn = loss_fn
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cfg = cfg or ValConfig()

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        batchUseFreq: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        model = model.to(self.device)
        model.eval()

        # Meters
        m_total = AverageMeter()
        m_mse_y50 = AverageMeter()
        m_mse_yr = AverageMeter()
        m_conf_bce = AverageMeter()

        m_mae_y50 = AverageMeter()
        m_rmse_y50 = AverageMeter()
        m_mae_yr = AverageMeter()
        m_rmse_yr = AverageMeter()

        m_mae_conf = AverageMeter()
        m_mse_conf = AverageMeter()

        m_mae_t50 = AverageMeter()
        m_rmse_t50 = AverageMeter()
        m_mae_t90 = AverageMeter()
        m_rmse_t90 = AverageMeter()

        # For correlations / diagnostics
        cache_y50_hat, cache_y50 = [], []
        cache_yr_hat, cache_yr = [], []
        cache_yc_prob, cache_yc = [], []

        cache_rel_scores, cache_rel_labels = [], []

        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        autocast_ctx = torch.cuda.amp.autocast if self.device.type == "cuda" else torch.cpu.amp.autocast

        # for batch in tqdm(val_loader, desc="Validation", unit="batch"):
        # Enumerate the batches with tqdm for progress tracking, and also apply batchUseFreq skipping
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
            if (batch_idx % batchUseFreq) != 0:
                continue

            b = _unpack_batch(batch)
            b = _to_device(b, self.device)

            image = b["image"]
            mask = b["mask"]
            y50 = b["y50"]
            yr = b["yr"]
            yc = b["yc"].float()

            bs = int(y50.shape[0]) if torch.is_tensor(y50) and y50.ndim > 0 else int(image.shape[0])

            with autocast_ctx(enabled=use_amp):
                y50_hat, yr_hat, yc_hat = model(image, mask)

                total_loss, lossMetrics, lossDebugTerms = self.loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)

                mse_y50 = F.mse_loss(y50_hat, y50, reduction="mean")
                mse_yr = F.mse_loss(yr_hat, yr, reduction="mean")

                if self.cfg.confidence_from_logits:
                    yc_prob = torch.sigmoid(yc_hat)
                    conf_bce = F.binary_cross_entropy_with_logits(yc_hat, yc, reduction="mean")
                else:
                    yc_prob = yc_hat
                    if self.cfg.clamp_confidence_probs:
                        yc_prob = yc_prob.clamp(self.cfg.eps, 1.0 - self.cfg.eps)
                    conf_bce = F.binary_cross_entropy(yc_prob, yc, reduction="mean")

            # Update meters
            m_total.update(total_loss.item(), bs)
            m_mse_y50.update(mse_y50.item(), bs)
            m_mse_yr.update(mse_yr.item(), bs)
            m_conf_bce.update(conf_bce.item(), bs)

            err_y50 = (y50_hat - y50).float()
            err_yr = (yr_hat - yr).float()

            m_mae_y50.update(err_y50.abs().mean().item(), bs)
            m_rmse_y50.update(err_y50.pow(2).mean().sqrt().item(), bs)
            m_mae_yr.update(err_yr.abs().mean().item(), bs)
            m_rmse_yr.update(err_yr.pow(2).mean().sqrt().item(), bs)

            err_conf = (yc_prob - yc).float()
            m_mae_conf.update(err_conf.abs().mean().item(), bs)
            m_mse_conf.update(err_conf.pow(2).mean().item(), bs)

            # Cache for correlations
            if self.cfg.compute_correlations:
                cache_y50_hat.append(y50_hat.detach().float().cpu())
                cache_y50.append(y50.detach().float().cpu())
                cache_yr_hat.append(yr_hat.detach().float().cpu())
                cache_yr.append(yr.detach().float().cpu())
                cache_yc_prob.append(yc_prob.detach().float().cpu())
                cache_yc.append(yc.detach().float().cpu())

            # Optional: metrics in original time space
            if self.cfg.compute_original_time_metrics:
                t50_hat = torch.exp(y50_hat.float())
                t50_true = torch.exp(y50.float())
                t90_hat = t50_hat * (1.0 + torch.exp(yr_hat.float()))
                t90_true = t50_true * (1.0 + torch.exp(yr.float()))

                err_t50 = t50_hat - t50_true
                err_t90 = t90_hat - t90_true

                m_mae_t50.update(err_t50.abs().mean().item(), bs)
                m_rmse_t50.update(err_t50.pow(2).mean().sqrt().item(), bs)
                m_mae_t90.update(err_t90.abs().mean().item(), bs)
                m_rmse_t90.update(err_t90.pow(2).mean().sqrt().item(), bs)

        # Aggregate score (accuracy-like; higher is better)
        aggregate_score = 1.0 / (1.0 + float(m_total.avg))

        metrics: Dict[str, float] = {
            "total_loss": float(m_total.avg),
            "aggregate_score": float(aggregate_score),

            "mse_y50": float(m_mse_y50.avg),
            "mse_yr": float(m_mse_yr.avg),
            "bce_conf": float(m_conf_bce.avg),

            "mae_y50": float(m_mae_y50.avg),
            "rmse_y50": float(m_rmse_y50.avg),
            "mae_yr": float(m_mae_yr.avg),
            "rmse_yr": float(m_rmse_yr.avg),

            "mae_conf": float(m_mae_conf.avg),
            "mse_conf": float(m_mse_conf.avg),
        }

        if self.cfg.compute_original_time_metrics:
            metrics.update({
                "mae_t50": float(m_mae_t50.avg),
                "rmse_t50": float(m_rmse_t50.avg),
                "mae_t90": float(m_mae_t90.avg),
                "rmse_t90": float(m_rmse_t90.avg),
            })

        if self.cfg.compute_correlations and len(cache_y50_hat) > 0:
            y50_hat_all = torch.cat(cache_y50_hat, dim=0)
            y50_all = torch.cat(cache_y50, dim=0)
            yr_hat_all = torch.cat(cache_yr_hat, dim=0)
            yr_all = torch.cat(cache_yr, dim=0)
            yc_prob_all = torch.cat(cache_yc_prob, dim=0)
            yc_all = torch.cat(cache_yc, dim=0)

            metrics.update({
                "pearson_y50": _pearson_corr(y50_hat_all, y50_all),
                "pearson_yr": _pearson_corr(yr_hat_all, yr_all),
                "pearson_conf": _pearson_corr(yc_prob_all, yc_all),

                "spearman_y50": _spearman_corr(y50_hat_all, y50_all),
                "spearman_yr": _spearman_corr(yr_hat_all, yr_all),
                "spearman_conf": _spearman_corr(yc_prob_all, yc_all),
            })

        return aggregate_score, metrics


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    cfg: Optional[ValConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Convenience wrapper to run validation and return:
      aggregate_score (higher is better),
      metrics dict.
    """
    runner = SurvivalValidationRunner(loss_fn=loss_fn, device=device, cfg=cfg)
    return runner.validate(model, val_loader)
