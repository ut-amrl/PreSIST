import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

ConfidenceLoss = Literal["bce", "mse", "huber"]

class WeightedMultiLoss(nn.Module):
    """
    Weighted multitask loss for:
      - y50 (log t50) in R
      - yr  (log((t90/t50)-1)) in R
      - yc  (confidence) in [0,1] (or logits, if configured)

    Options:
      - useHuberForLogVals: use Huber (Smooth L1) instead of MSE for y50 and yr
      - confidence_loss: choose BCE / MSE / Huber for yc
      - confidence_from_logits: if True and confidence_loss="bce", use BCEWithLogits
    """

    def __init__(
        self,
        w50: float = 1.0,
        wr: float = 1.0,
        wc: float = 0.5,
        useHuberForLogVals: bool = False,
        delta_50: float = 1.0,
        delta_r: float = 1.0,
        confidence_loss: ConfidenceLoss = "mse",
        confidence_from_logits: bool = False,
        delta_c: float = 0.25,
        clamp_confidence_probs: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w50 = w50
        self.wr = wr
        self.wc = wc

        self.useHuberForLogVals = useHuberForLogVals
        self.delta_50 = delta_50
        self.delta_r = delta_r

        self.confidence_loss = confidence_loss
        self.confidence_from_logits = confidence_from_logits
        self.delta_c = delta_c

        self.clamp_confidence_probs = clamp_confidence_probs
        self.eps = eps

        if self.confidence_from_logits and self.confidence_loss != "bce":
            raise ValueError("confidence_from_logits=True is only valid when confidence_loss='bce'.")

    @staticmethod
    def _huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=delta)

    def _log_loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        if self.useHuberForLogVals:
            return self._huber(pred, target, delta)
        return F.mse_loss(pred, target)

    def _confidence_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure target is float tensor for all cases
        target = target.to(dtype=pred.dtype)

        if self.confidence_loss == "bce":
            if self.confidence_from_logits:
                # pred is logits, target in [0,1]
                return F.binary_cross_entropy_with_logits(pred, target)
            else:
                # pred should already be probabilities in [0,1]
                if self.clamp_confidence_probs:
                    pred = pred.clamp(self.eps, 1.0 - self.eps)
                return F.binary_cross_entropy(pred, target)

        if self.confidence_loss == "mse":
            return F.mse_loss(pred, target)

        if self.confidence_loss == "huber":
            return self._huber(pred, target, self.delta_c)

        raise ValueError(f"Unknown confidence_loss: {self.confidence_loss!r}")

    def forward(
        self,
        y50_hat: torch.Tensor, y50: torch.Tensor,
        yr_hat: torch.Tensor,  yr: torch.Tensor,
        yc_hat: torch.Tensor,  yc: torch.Tensor,
        extraLossArgs: Optional[dict] = None
    ) -> torch.Tensor:
        loss_50 = self._log_loss(y50_hat, y50, self.delta_50)
        loss_r  = self._log_loss(yr_hat,  yr,  self.delta_r)
        loss_c  = self._confidence_loss(yc_hat, yc)

        fullLoss = (self.w50 * loss_50) + (self.wr * loss_r) + (self.wc * loss_c)
        losses = {
            "loss_50": loss_50.detach(),
            "loss_r": loss_r.detach(),
            "loss_c": loss_c.detach(),
            "full_loss": fullLoss.detach()
        }

        debugInfo = None

        return fullLoss, losses, debugInfo
