from __future__ import annotations

# ---------------------------------------------------------------------------
# TODO / known concerns in this module — none are urgent, but worth a pass
# before relying on these numbers across runs or comparing to prior baselines.
#
# Semantics / comparability:
#   - The headline metric keys (`total_loss`, `mse_y50`, `mae_y50`,
#     `aggregate_score`, etc.) silently swapped from per-instance to
#     image-weighted. Older runs' values are not directly comparable;
#     per-instance variants are preserved under `*_per_instance`.
#   - `total_loss` is recomposed as `w50·MSE(y50) + wr·MSE(yr) + wc·BCE`.
#     If `loss_fn` uses Huber/SmoothL1 (or any non-MSE term), this is a
#     proxy, not the trained objective — best-checkpoint selection will
#     be biased toward outlier-suppression rather than the trained loss's
#     optimum. Fix: route `lossMetrics`/`lossDebugTerms` from `loss_fn`
#     into the accumulators instead of recomposing.
#   - `lossMetrics` and `lossDebugTerms` returned by `loss_fn` are
#     currently discarded entirely.
#   - The real `total_loss` from `loss_fn` is computed every batch but
#     only feeds `total_loss_per_instance`. The actual training-objective
#     value is alive in the loop and hidden behind a `_per_instance`
#     suffix while the proxy takes the headline name.
#   - Pearson/Spearman correlations are per-instance, not image-weighted —
#     inconsistent with the rest of the metrics. On instance-heavy images
#     (parking lots) this skews the correlation toward the dominant image.
#
# Performance:
#   - Per-instance meters still do ~6 `.item()` syncs per batch
#     (m_total / m_mse_* / m_mae_* / m_rmse_* / m_conf_*). The IW path is
#     sync-free; the per-instance path is the next bottleneck. Same fix
#     pattern: on-device accumulators, single drain after the loop.
#   - The two `torch.cuda.synchronize()` calls bracketing inference (for
#     phase timing) destroy GPU/dataloader overlap. Intentional for the
#     tqdm-postfix display; should be gated behind a `cfg.profile_phases`
#     flag for production runs.
#   - Correlation caches grow unbounded and force per-batch `.cpu()`
#     transfers (six per batch). Streaming statistics (E[x], E[y], E[xy],
#     E[x²], E[y²]) would be O(1) memory and sync-free.
#
# Cleanup:
#   - `_weighted_mean` helper is now dead code; kept only in case future
#     code wants it.
#   - `cache_rel_scores` / `cache_rel_labels` are declared but never read
#     or written — stub for relative-ranking metrics that was never wired.
#   - `torch.cuda.amp.autocast` is the legacy spelling; PyTorch 2.x
#     prefers `torch.amp.autocast("cuda", ...)`. Works, but warns.
#
# Latent foot-guns:
#   - No assert on `y50_hat.shape == y50.shape` (and similarly for yr/yc).
#     If a model head ever emits `(B,1)` while labels are `(B,)`, the
#     subtraction broadcasts to `(B,B)` and metrics become silently
#     nonsense. A one-shot first-batch assert would catch this.
#   - `validate()` leaves `self.model` in `.eval()` mode on exit. Today
#     the trainer's `_train_one_epoch` calls `self.model.train()` first
#     thing each epoch, so this is benign. Any future code path that
#     touches the model between validate() and the next epoch (extra
#     logging hooks, post-validate inference, end-of-training eval) would
#     silently run with dropout/BN off. Cheap fix: save/restore the prior
#     training flag inside `validate()`.
#   - The per-instance RMSE meters (`rmse_y50_per_instance`, etc.) are
#     `mean_over_batches(sqrt(per_batch_mse))`, which is biased — the IW
#     versions compute the true `sqrt(weighted_mean(se))` and are correct.
# ---------------------------------------------------------------------------

import time
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


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12) -> float:
    """Per-batch weighted scalar mean used to update an image-weighted meter."""
    values = values.float().reshape(-1)
    weights = weights.float().reshape(-1)
    wsum = weights.sum().clamp_min(eps)
    return float(((values * weights).sum() / wsum).item())


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
        max_batches: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Run validation.

        Args:
            max_batches: when set, stop pulling from `val_loader` after
                this many batches (counting only the batches actually
                processed, i.e. not the ones skipped by batchUseFreq).
                Use this to bound train-as-val time without iterating
                the entire dataloader (which would still pay the data
                loading cost on every skipped batch).
        """
        model = model.to(self.device)
        model.eval()

        # Meters (per-instance — every sample contributes equally)
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

        # Image-weighted accumulators. Each sample contributes weight
        # 1/n_instances_in_image (from the dataset's sample_weight) so
        # each image contributes equal expected weight, matching the
        # training-time image-balanced sampler.
        #
        # We accumulate Σ(w·x) on device per metric plus a single Σw,
        # then divide once at the end of the loop. This avoids the
        # per-batch GPU→CPU .item() syncs an AverageMeter approach
        # needs (one per IW metric per batch — ~8 per batch before).
        # float64 to stay safe over long val passes.
        iw_keys = [
            "se_y50", "se_yr", "bce_conf",
            "abs_y50", "abs_yr",
            "abs_conf", "se_conf",
        ]
        if self.cfg.compute_original_time_metrics:
            iw_keys += ["abs_t50", "se_t50", "abs_t90", "se_t90"]
        iw_sum: Dict[str, torch.Tensor] = {
            k: torch.zeros((), device=self.device, dtype=torch.float64)
            for k in iw_keys
        }
        iw_w_total = torch.zeros((), device=self.device, dtype=torch.float64)

        # Look up the loss-component weights so we can recompose an
        # image-weighted total_loss out of per-sample components.
        w50 = float(getattr(self.loss_fn, "w50", 1.0))
        wr  = float(getattr(self.loss_fn, "wr", 1.0))
        wc  = float(getattr(self.loss_fn, "wc", 1.0))

        # For correlations / diagnostics
        cache_y50_hat, cache_y50 = [], []
        cache_yr_hat, cache_yr = [], []
        cache_yc_prob, cache_yc = [], []

        cache_rel_scores, cache_rel_labels = [], []

        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        autocast_ctx = torch.cuda.amp.autocast if self.device.type == "cuda" else torch.cpu.amp.autocast

        # Per-phase timing accumulators (rolling means shown in tqdm postfix
        # so wall-time bottlenecks are visible at a glance):
        #   dl   = time spent waiting on DataLoader to yield the next batch
        #          (includes per-worker decode + transform + collate + h2d copy from queue)
        #   h2d  = transfer batch dict to device
        #   inf  = model forward + per-sample component loss compute
        #          (a torch.cuda.synchronize() call brackets this so timings
        #           reflect real GPU work, not just kernel-launch latency)
        #   post = AverageMeter updates + cache appends for correlations
        is_cuda = self.device.type == "cuda"

        # Set the tqdm total to reflect the actual stopping point so
        # the progress bar and ETA are accurate. With batchUseFreq>1 we
        # consume `batchUseFreq` batches from the loader per processed
        # batch; the cap is on processed (not consumed) so the consumed
        # total is `max_batches * batchUseFreq`.
        try:
            loader_total = len(val_loader)
        except TypeError:
            loader_total = None
        if max_batches is not None and loader_total is not None:
            tqdm_total = min(loader_total, int(max_batches) * max(1, batchUseFreq))
        elif max_batches is not None:
            tqdm_total = int(max_batches) * max(1, batchUseFreq)
        else:
            tqdm_total = loader_total

        pbar = tqdm(val_loader, desc="Validation", unit="batch", total=tqdm_total)
        n_timed = 0
        t_dl = t_h2d = t_inf = t_post = 0.0
        t_prev_end = time.perf_counter()
        for batch_idx, batch in enumerate(pbar):
            # Time waiting on the DataLoader (everything that happens
            # outside this `for` body)
            t_dl_now = time.perf_counter() - t_prev_end

            if (batch_idx % batchUseFreq) != 0:
                t_prev_end = time.perf_counter()
                continue

            # Early-stop when the caller has capped the number of
            # processed batches (e.g. for the train-as-val pass). We
            # `break` instead of `continue`-ing so the DataLoader's
            # workers don't keep producing batches we'll discard.
            if max_batches is not None and n_timed >= int(max_batches):
                pbar.close()
                break

            b = _unpack_batch(batch)
            b = _to_device(b, self.device)
            if is_cuda:
                torch.cuda.synchronize()
            t_h2d_now = time.perf_counter() - t_prev_end - t_dl_now
            t_inf_start = time.perf_counter()

            image = b["image"]
            mask = b["mask"]
            y50 = b["y50"]
            yr = b["yr"]
            yc = b["yc"].float()

            # When the validation model is a DenormalizingValidationModel,
            # its outputs (y50_hat, yr_hat) have had the dataset's z-score
            # inverted, but the targets in the batch are still z-scored.
            # Without matching spaces, every metric here would compare
            # un-normalized preds against normalized targets — silently
            # inflating SE and collapsing aggregate_score toward 0. We
            # apply the same inverse-affine to the targets so both sides
            # are un-normalized. No-op for non-`_norm` runs (buffers
            # absent → `None`).
            _y50_mean = getattr(model, "y50_mean", None)
            _y50_std = getattr(model, "y50_std", None)
            _yr_mean = getattr(model, "yr_mean", None)
            _yr_std = getattr(model, "yr_std", None)
            if all(b is not None for b in (_y50_mean, _y50_std, _yr_mean, _yr_std)):
                y50 = y50.float() * _y50_std + _y50_mean
                yr = yr.float() * _yr_std + _yr_mean
            # Per-sample weights (1/n_instances_in_image) for the parallel
            # image-weighted meters. Defaults to uniform if absent (e.g.
            # legacy loaders that don't emit sample_weight).
            sample_weight = b.get("sample_weight", None)
            if sample_weight is None:
                sample_weight = torch.ones_like(y50.float())
            else:
                sample_weight = sample_weight.float()

            bs = int(y50.shape[0]) if torch.is_tensor(y50) and y50.ndim > 0 else int(image.shape[0])

            with autocast_ctx(enabled=use_amp):
                y50_hat, yr_hat, yc_hat = model(image, mask)

                total_loss, lossMetrics, lossDebugTerms = self.loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)

                # Per-sample component losses (squared error / BCE) for
                # weighted aggregation.
                se_y50 = (y50_hat - y50).float().pow(2)
                se_yr = (yr_hat - yr).float().pow(2)

                if self.cfg.confidence_from_logits:
                    yc_prob = torch.sigmoid(yc_hat)
                    bce_per = F.binary_cross_entropy_with_logits(yc_hat, yc, reduction="none")
                else:
                    yc_prob = yc_hat
                    if self.cfg.clamp_confidence_probs:
                        yc_prob = yc_prob.clamp(self.cfg.eps, 1.0 - self.cfg.eps)
                    bce_per = F.binary_cross_entropy(yc_prob, yc, reduction="none")

                mse_y50 = se_y50.mean()
                mse_yr = se_yr.mean()
                conf_bce = bce_per.mean()

            # End of inference block (model fwd + per-sample component compute).
            # Force a sync so the GPU time is reflected in t_inf, not bled
            # into the next iteration's t_dl.
            if is_cuda:
                torch.cuda.synchronize()
            t_inf_now = time.perf_counter() - t_inf_start
            t_post_start = time.perf_counter()

            # ----- per-instance meters (existing behavior) -----
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

            # ----- image-weighted accumulators -----
            # Stays on device; we divide once after the loop. float64
            # to avoid precision loss on long val passes.
            sw64 = sample_weight.reshape(-1).double()
            iw_w_total += sw64.sum()
            iw_sum["se_y50"]   += (se_y50.reshape(-1).double()         * sw64).sum()
            iw_sum["se_yr"]    += (se_yr.reshape(-1).double()          * sw64).sum()
            iw_sum["bce_conf"] += (bce_per.reshape(-1).double()        * sw64).sum()
            iw_sum["abs_y50"]  += (err_y50.abs().reshape(-1).double()  * sw64).sum()
            iw_sum["abs_yr"]   += (err_yr.abs().reshape(-1).double()   * sw64).sum()
            iw_sum["abs_conf"] += (err_conf.abs().reshape(-1).double() * sw64).sum()
            iw_sum["se_conf"]  += (err_conf.pow(2).reshape(-1).double() * sw64).sum()

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

                iw_sum["abs_t50"] += (err_t50.abs().reshape(-1).double()  * sw64).sum()
                iw_sum["se_t50"]  += (err_t50.pow(2).reshape(-1).double() * sw64).sum()
                iw_sum["abs_t90"] += (err_t90.abs().reshape(-1).double()  * sw64).sum()
                iw_sum["se_t90"]  += (err_t90.pow(2).reshape(-1).double() * sw64).sum()

            # End of post-processing block (meter updates + correlation caches).
            t_post_now = time.perf_counter() - t_post_start
            t_dl   += t_dl_now
            t_h2d  += t_h2d_now
            t_inf  += t_inf_now
            t_post += t_post_now
            n_timed += 1
            pbar.set_postfix({
                "dl":   f"{t_dl   / n_timed * 1000:.0f}ms",
                "h2d":  f"{t_h2d  / n_timed * 1000:.0f}ms",
                "inf":  f"{t_inf  / n_timed * 1000:.0f}ms",
                "post": f"{t_post / n_timed * 1000:.0f}ms",
            })
            t_prev_end = time.perf_counter()

        # Drain the IW accumulators in a single host transfer:
        # stack all on-device sums, divide by Σw, and ship the whole
        # vector to CPU once. RMSE is then sqrt(true_mean_se) — note
        # this differs from the per-instance RMSE meters, which
        # average per-batch sqrt(mse) and so are slightly biased.
        iw_keys_order = list(iw_sum.keys())
        if len(iw_keys_order) > 0:
            stacked = torch.stack([iw_sum[k] for k in iw_keys_order])
            denom = iw_w_total.clamp_min(1e-12)
            iw_vals_list = (stacked / denom).cpu().tolist()
            iw = dict(zip(iw_keys_order, iw_vals_list))
        else:
            iw = {}

        total_iw = w50 * iw["se_y50"] + wr * iw["se_yr"] + wc * iw["bce_conf"]

        # Aggregate score is computed from the image-weighted total loss
        # so early-stopping / best-checkpoint selection matches the
        # training-time objective when image_balanced_sampling is on.
        # The per-instance total_loss is still reported (as
        # `total_loss_per_instance`) for visibility.
        aggregate_score = 1.0 / (1.0 + total_iw)

        metrics: Dict[str, float] = {
            "total_loss": float(total_iw),
            "total_loss_per_instance": float(m_total.avg),
            "aggregate_score": float(aggregate_score),

            "mse_y50": float(iw["se_y50"]),
            "mse_yr": float(iw["se_yr"]),
            "bce_conf": float(iw["bce_conf"]),

            "mae_y50": float(iw["abs_y50"]),
            "rmse_y50": float(iw["se_y50"] ** 0.5),
            "mae_yr": float(iw["abs_yr"]),
            "rmse_yr": float(iw["se_yr"] ** 0.5),

            "mae_conf": float(iw["abs_conf"]),
            "mse_conf": float(iw["se_conf"]),

            # Per-instance variants kept for visibility
            "mse_y50_per_instance": float(m_mse_y50.avg),
            "mse_yr_per_instance": float(m_mse_yr.avg),
            "bce_conf_per_instance": float(m_conf_bce.avg),
            "mae_y50_per_instance": float(m_mae_y50.avg),
            "rmse_y50_per_instance": float(m_rmse_y50.avg),
            "mae_yr_per_instance": float(m_mae_yr.avg),
            "rmse_yr_per_instance": float(m_rmse_yr.avg),
            "mae_conf_per_instance": float(m_mae_conf.avg),
            "mse_conf_per_instance": float(m_mse_conf.avg),
        }

        if self.cfg.compute_original_time_metrics:
            metrics.update({
                "mae_t50": float(iw["abs_t50"]),
                "rmse_t50": float(iw["se_t50"] ** 0.5),
                "mae_t90": float(iw["abs_t90"]),
                "rmse_t90": float(iw["se_t90"] ** 0.5),

                "mae_t50_per_instance": float(m_mae_t50.avg),
                "rmse_t50_per_instance": float(m_rmse_t50.avg),
                "mae_t90_per_instance": float(m_mae_t90.avg),
                "rmse_t90_per_instance": float(m_rmse_t90.avg),
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
