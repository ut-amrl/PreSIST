"""Rasterize per-instance labels into dense per-pixel supervision tensors.

Each rasterizer takes ``(instance_masks, instance_labels)`` and produces:

  - ``target_map``:    ``(C, H, W)`` per-pixel target (C=3 for our labels).
  - ``loss_mask``:     ``(1, H, W)`` uint8/bool — which pixels participate
                       in the loss (i.e. the union of valid instance pixels
                       under the chosen overlap policy).
  - ``pixel_weights``: ``(1, H, W)`` float — per-pixel weight from the
                       normalization policy. Multiplies the per-pixel loss
                       before reduction.

The rasterizer family is intentionally split into two orthogonal pieces:

  - :class:`TargetRasterizer` decides what to write at each pixel and which
    pixels count (overlap policy).
  - :class:`TargetNormalizer` decides what per-pixel weight each pixel gets
    (area normalization policy).

Both expose a ``cache_fingerprint()`` and ``__version__`` so the dense
supervision cache (see :mod:`dense_supervision_cache`) can invalidate
correctly when either piece changes.

Conventions:
  - ``instance_masks``: ``(N, H, W)`` non-negative tensor. Strictly
    positive pixels (``> 0``) are "inside the instance"; values are
    treated as binary here.
  - ``instance_labels``: ``(N, C)`` float tensor with the per-instance
    target vector. We do NOT apply any activation (e.g. sigmoid on conf)
    inside the rasterizer — the rasterized targets must match the model's
    output convention, which is governed by ``confidence_from_logits``
    elsewhere.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Normalizers — per-pixel weight from per-instance area
# ---------------------------------------------------------------------------
class TargetNormalizer(ABC):
    """Produces a per-pixel weight map given the per-pixel instance-area map.

    Implementations should be pure functions of their constructor args; the
    cache_fingerprint() captures those args so cached tensors are invalidated
    when normalizer config changes.
    """

    __version__: int = 1

    @abstractmethod
    def per_instance_weight(
        self,
        area: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map a ``(N,)`` tensor of per-instance areas (in pixels) to a
        ``(N,)`` per-instance weight. Each pixel of instance i gets that
        weight.

        ``instance_labels`` is an optional ``(N, C)`` tensor of the same
        labels the rasterizer is about to paint. Most subclasses ignore
        it; the label-binned-frequency normalizer uses it to look up a
        per-instance weight indexed by label value.
        """
        raise NotImplementedError

    def cache_fingerprint(self) -> str:
        payload = {
            "class": type(self).__name__,
            "version": int(self.__class__.__version__),
            "config": self._config_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _config_dict(self) -> dict:
        """Override to include any tunable parameters in the cache key."""
        return {}


class NoNormalization(TargetNormalizer):
    """Every pixel gets weight 1.0; large instances dominate by area."""

    def per_instance_weight(
        self,
        area: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ones_like(area, dtype=torch.float32)


class PerInstanceAreaNormalizer(TargetNormalizer):
    """``weight[pixel in instance i] = 1 / area(i)`` so each instance
    contributes equal total weight to the loss regardless of size."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = float(eps)

    def per_instance_weight(
        self,
        area: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return 1.0 / area.clamp(min=self.eps).to(torch.float32)

    def _config_dict(self) -> dict:
        return {"eps": self.eps}


class SqrtAreaNormalizer(TargetNormalizer):
    """``weight[pixel in instance i] = 1 / sqrt(area(i))``. Softens both
    extremes — large instances dominate less than under :class:`NoNormalization`
    but small ones less than under per-instance-area."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = float(eps)

    def per_instance_weight(
        self,
        area: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return 1.0 / area.clamp(min=self.eps).to(torch.float32).sqrt()

    def _config_dict(self) -> dict:
        return {"eps": self.eps}


class LabelBinnedFrequencyNormalizer(TargetNormalizer):
    """Per-instance weight = base weight × inverse-frequency weight from a
    precomputed histogram of one label channel.

    The frequency table is a numpy archive with two arrays:
      - ``bin_edges`` (B+1,) float — bin edges of the histogram, sorted.
      - ``weights``   (B,)   float — per-bin weight (typically ``1/count_b``
        normalized so the mean weight over training samples is ~1).

    The script ``scripts/compute_label_inverse_freq.py`` produces it.

    Composes multiplicatively with a base normalizer (default
    :class:`NoNormalization`, so freq-weighting alone applies). Most often
    paired with :class:`PerInstanceAreaNormalizer` to combine area- and
    label-rarity weighting.

    The table's binary contents are hashed into the rasterizer cache
    fingerprint, so regenerating the table invalidates the cache
    automatically.
    """

    __version__: int = 1

    def __init__(
        self,
        freq_table_path: str,
        label_channel: int = 0,
        base: Optional[TargetNormalizer] = None,
    ):
        super().__init__()
        self.freq_table_path = str(freq_table_path)
        self.label_channel = int(label_channel)
        self.base = base if base is not None else NoNormalization()

        path = Path(self.freq_table_path)
        if not path.exists():
            raise FileNotFoundError(
                f"LabelBinnedFrequencyNormalizer: freq table not found at "
                f"{path}. Generate it with "
                f"scripts/compute_label_inverse_freq.py first."
            )
        data = np.load(str(path))
        if "bin_edges" not in data or "weights" not in data:
            raise KeyError(
                f"Freq table at {path} missing required arrays "
                f"('bin_edges', 'weights'). Keys present: {list(data.keys())}"
            )
        self.bin_edges = torch.from_numpy(np.asarray(data["bin_edges"])).to(torch.float32)
        self.weights = torch.from_numpy(np.asarray(data["weights"])).to(torch.float32)
        if int(self.bin_edges.numel()) != int(self.weights.numel()) + 1:
            raise ValueError(
                f"Freq table shape mismatch at {path}: "
                f"len(bin_edges)={int(self.bin_edges.numel())} must be "
                f"len(weights)+1={int(self.weights.numel()) + 1}."
            )
        with open(path, "rb") as fh:
            self._content_hash = hashlib.sha256(fh.read()).hexdigest()[:16]

    def per_instance_weight(
        self,
        area: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if instance_labels is None:
            raise ValueError(
                "LabelBinnedFrequencyNormalizer requires instance_labels to "
                "be passed through from the rasterizer; got None."
            )
        if instance_labels.shape[0] != area.shape[0]:
            raise ValueError(
                f"instance_labels rows ({instance_labels.shape[0]}) must match "
                f"area length ({area.shape[0]})."
            )
        y = instance_labels[:, self.label_channel].to(torch.float32).contiguous()
        # bucketize returns indices in [0, B] where 0 = below first edge and
        # B = at/above last edge. Clamp to [1, B] then subtract to land in
        # the per-bin weight indices [0, B-1].
        n_bins = int(self.weights.numel())
        idx = torch.bucketize(y, self.bin_edges, right=False).clamp(min=1, max=n_bins) - 1
        freq_w = self.weights[idx]
        base_w = self.base.per_instance_weight(area, instance_labels)
        return freq_w * base_w

    def _config_dict(self) -> dict:
        return {
            "freq_table_path": self.freq_table_path,
            "label_channel": self.label_channel,
            "content_hash": self._content_hash,
            "base_class": type(self.base).__name__,
            "base_version": int(type(self.base).__version__),
            "base_config": self.base._config_dict(),
        }


# ---------------------------------------------------------------------------
# Rasterizers — overlap policy + per-pixel target writing
# ---------------------------------------------------------------------------
class TargetRasterizer(ABC):
    """Decide which pixels get which instance's label.

    Subclasses implement ``rasterize`` directly. The shared utility
    ``_apply_normalizer`` is provided so each subclass can call it after
    computing per-pixel instance ownership.
    """

    __version__: int = 1

    def __init__(self, normalizer: TargetNormalizer):
        self.normalizer = normalizer

    def cache_fingerprint(self) -> str:
        payload = {
            "class": type(self).__name__,
            "version": int(self.__class__.__version__),
            "config": self._config_dict(),
            "normalizer": self.normalizer.cache_fingerprint(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _config_dict(self) -> dict:
        return {}

    @abstractmethod
    def rasterize(
        self,
        instance_masks: torch.Tensor,   # (N, H, W) non-negative
        instance_labels: torch.Tensor,  # (N, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(target_map (C,H,W), loss_mask (1,H,W) bool,
        pixel_weights (1,H,W) float)``.

        When ``N == 0`` (no instances), all outputs are zeros at the
        spatial size implied by the caller via a separate code path (the
        rasterizer can't infer H/W from an empty mask tensor). Subclasses
        should handle this by reading ``instance_masks.shape[-2:]``.
        """
        raise NotImplementedError

    # ---- helpers shared by concrete subclasses ----
    def _empty_outputs(self, H: int, W: int, C: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = torch.zeros((C, H, W), dtype=torch.float32)
        loss_mask = torch.zeros((1, H, W), dtype=torch.bool)
        pixel_weights = torch.zeros((1, H, W), dtype=torch.float32)
        return target, loss_mask, pixel_weights

    def _binarize(self, instance_masks: torch.Tensor) -> torch.Tensor:
        return (instance_masks > 0).to(torch.bool)


def _paint_from_owner(
    owner: torch.Tensor,           # (H, W) int — instance index per pixel; -1 = no owner
    instance_labels: torch.Tensor, # (N, C)
    normalizer: TargetNormalizer,
    instance_areas: torch.Tensor,  # (N,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = owner.shape
    C = int(instance_labels.shape[1])
    valid = owner >= 0
    target = torch.zeros((C, H, W), dtype=torch.float32)
    pixel_weights = torch.zeros((1, H, W), dtype=torch.float32)

    if valid.any():
        # Gather labels per-pixel.
        owner_clamped = owner.clamp(min=0)  # safe index for invalid (will be masked out)
        per_pixel_label = instance_labels[owner_clamped]                  # (H, W, C)
        target = per_pixel_label.permute(2, 0, 1) * valid.to(torch.float32)

        per_instance_weight = normalizer.per_instance_weight(
            instance_areas, instance_labels
        )  # (N,)
        per_pixel_weight = per_instance_weight[owner_clamped]              # (H, W)
        pixel_weights = (per_pixel_weight * valid.to(torch.float32)).unsqueeze(0)

    loss_mask = valid.unsqueeze(0)
    return target, loss_mask, pixel_weights


class SmallerWinsRasterizer(TargetRasterizer):
    """Overlap pixels go to the *smaller* instance (by area).

    Rationale: smaller masks are usually the more specific label
    (e.g. "car" inside "parking lot"). Pixels are deterministically
    assigned to whichever covering instance has the smallest area.
    """

    def rasterize(self, instance_masks, instance_labels):
        N = int(instance_masks.shape[0])
        H, W = int(instance_masks.shape[-2]), int(instance_masks.shape[-1])
        C = int(instance_labels.shape[1])
        if N == 0:
            return self._empty_outputs(H, W, C)

        binary = self._binarize(instance_masks)              # (N, H, W) bool
        areas = binary.sum(dim=(1, 2)).to(torch.int64)        # (N,)

        # For each pixel, choose the instance with the smallest area among
        # all instances covering it. We assign each instance an order =
        # rank-by-ascending-area and pick the min order at each pixel.
        order = torch.argsort(torch.argsort(areas))           # (N,) — rank
        # Build "score" tensor: pixels not covered by instance i get a
        # large value so they're not selected; covered pixels get the
        # instance's order. min over N gives the smallest-area instance.
        large = order.max().item() + 1
        # (N, H, W) -> we'd allocate N*H*W; use loop-free trick via gather:
        # Instead compute per-pixel best by iterating instances sorted by
        # area ascending and painting only at pixels not yet owned.
        owner = torch.full((H, W), -1, dtype=torch.int64)
        # Iterate from largest area first so smaller areas overwrite later.
        sort_idx = torch.argsort(areas, descending=True)
        for i in sort_idx.tolist():
            covered = binary[i]
            owner = torch.where(covered, torch.tensor(i, dtype=owner.dtype), owner)

        return _paint_from_owner(owner, instance_labels, self.normalizer, areas)


class LargerWinsRasterizer(TargetRasterizer):
    """Overlap pixels go to the larger instance (by area)."""

    def rasterize(self, instance_masks, instance_labels):
        N = int(instance_masks.shape[0])
        H, W = int(instance_masks.shape[-2]), int(instance_masks.shape[-1])
        C = int(instance_labels.shape[1])
        if N == 0:
            return self._empty_outputs(H, W, C)

        binary = self._binarize(instance_masks)
        areas = binary.sum(dim=(1, 2)).to(torch.int64)
        owner = torch.full((H, W), -1, dtype=torch.int64)
        # Paint smallest first, larger overwrites.
        sort_idx = torch.argsort(areas, descending=False)
        for i in sort_idx.tolist():
            covered = binary[i]
            owner = torch.where(covered, torch.tensor(i, dtype=owner.dtype), owner)

        return _paint_from_owner(owner, instance_labels, self.normalizer, areas)


class ExcludeOverlapRasterizer(TargetRasterizer):
    """Pixels covered by more than one instance are excluded from the loss.

    Singly-covered pixels receive the unique covering instance's label;
    multiply-covered pixels have ``loss_mask == 0`` and contribute
    nothing.
    """

    def rasterize(self, instance_masks, instance_labels):
        N = int(instance_masks.shape[0])
        H, W = int(instance_masks.shape[-2]), int(instance_masks.shape[-1])
        C = int(instance_labels.shape[1])
        if N == 0:
            return self._empty_outputs(H, W, C)

        binary = self._binarize(instance_masks)
        areas = binary.sum(dim=(1, 2)).to(torch.int64)
        coverage = binary.to(torch.int32).sum(dim=0)          # (H, W) — how many instances cover each pixel
        singly = coverage == 1
        # For singly-covered pixels, owner = argmax along N (the single
        # instance whose mask is True). For multiply-covered (or zero),
        # owner = -1.
        owner = torch.where(
            singly,
            binary.to(torch.int64).argmax(dim=0),
            torch.full_like(coverage, -1, dtype=torch.int64),
        )

        return _paint_from_owner(owner, instance_labels, self.normalizer, areas)


def _multi_owner_paint(
    binary: torch.Tensor,           # (N, H, W) bool
    weights_per_inst: torch.Tensor, # (N,) — per-pixel contribution weight for instance i
    instance_labels: torch.Tensor,  # (N, C)
    normalizer: TargetNormalizer,
    instance_areas: torch.Tensor,   # (N,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper for averaging overlap policies. Computes per-pixel target as
    a weighted average of all covering instances' labels, and combines the
    normalizer weight into the pixel weight."""
    N, H, W = binary.shape
    C = int(instance_labels.shape[1])

    # Per-instance contribution: pixels with `binary[i]==True` contribute weight
    # `weights_per_inst[i]` to the per-pixel weighted average.
    contrib = binary.to(torch.float32) * weights_per_inst.view(N, 1, 1)      # (N, H, W)
    denom = contrib.sum(dim=0)                                                # (H, W)
    valid = denom > 0

    # Weighted sum of labels at each pixel.
    # labels: (N, C); we want (C, H, W) = sum_n contrib[n,h,w] * labels[n,:]
    target = torch.einsum("nhw,nc->chw", contrib, instance_labels)
    target = target / denom.clamp(min=1e-12)
    target = target * valid.to(torch.float32)

    # Combine normalizer's per-instance weight into the pixel weight.
    # Each pixel's weight is the sum of normalizer weights of covering
    # instances divided by N_covering (so the magnitude is comparable to
    # the non-averaging policies). Simpler choice: use the mean.
    inst_norm_weight = normalizer.per_instance_weight(
        instance_areas, instance_labels
    )         # (N,)
    pixel_weight_num = (binary.to(torch.float32) * inst_norm_weight.view(N, 1, 1)).sum(dim=0)
    pixel_weight_den = binary.to(torch.float32).sum(dim=0).clamp(min=1.0)
    pixel_weights = (pixel_weight_num / pixel_weight_den).unsqueeze(0)
    pixel_weights = pixel_weights * valid.to(torch.float32).unsqueeze(0)

    loss_mask = valid.unsqueeze(0)
    return target, loss_mask, pixel_weights


class AverageOverlapRasterizer(TargetRasterizer):
    """At overlap pixels, target is the unweighted mean of covering instance labels."""

    def rasterize(self, instance_masks, instance_labels):
        N = int(instance_masks.shape[0])
        H, W = int(instance_masks.shape[-2]), int(instance_masks.shape[-1])
        C = int(instance_labels.shape[1])
        if N == 0:
            return self._empty_outputs(H, W, C)
        binary = self._binarize(instance_masks)
        areas = binary.sum(dim=(1, 2)).to(torch.int64)
        per_inst_contrib = torch.ones(N, dtype=torch.float32)
        return _multi_owner_paint(binary, per_inst_contrib, instance_labels, self.normalizer, areas)


class AreaWeightedAverageRasterizer(TargetRasterizer):
    """At overlap pixels, target is a weighted mean of covering instance
    labels with weight ``1 / area(i)`` (smaller instances dominate more)."""

    def rasterize(self, instance_masks, instance_labels):
        N = int(instance_masks.shape[0])
        H, W = int(instance_masks.shape[-2]), int(instance_masks.shape[-1])
        C = int(instance_labels.shape[1])
        if N == 0:
            return self._empty_outputs(H, W, C)
        binary = self._binarize(instance_masks)
        areas = binary.sum(dim=(1, 2)).to(torch.int64)
        per_inst_contrib = 1.0 / areas.clamp(min=1).to(torch.float32)
        return _multi_owner_paint(binary, per_inst_contrib, instance_labels, self.normalizer, areas)
