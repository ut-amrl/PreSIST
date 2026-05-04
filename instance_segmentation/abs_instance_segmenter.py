from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Optional, Tuple
import torch
import numpy as np
from utils.mask_utils import dedupe_mask_indices_by_iou, filter_mask_indices_by_min_area
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload, DebugPayloadFactory


def _dilate_mask_2d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a 2D bool mask by a square structuring element.

    Implemented as a max-pool over a (2r+1) window with stride 1 and r padding.
    """
    if radius <= 0:
        return mask
    import torch.nn.functional as F
    x = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    pooled = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return pooled.squeeze(0).squeeze(0) > 0.5


# =========================
# Core, model-agnostic API
# =========================

@dataclass
class SegmentationResults:
    """
    Results for one image.

    masks:  (N, H, W) bool tensor (False=background, True=instance)
    boxes:  (N, 4) float32 XYXY tensor
    scores: (N,) float32 tensor (model confidence if available)
    """
    ids: torch.Tensor            # int64, shape (N,)
    masks: torch.Tensor          # bool, shape (N, H, W)
    boxes: torch.Tensor          # float32, shape (N, 4) in XYXY
    scores: Optional[torch.Tensor] = None  # float32, shape (N,)

    def __len__(self) -> int:
        return self.masks.shape[0] if self.masks.ndim == 3 else 0

    @staticmethod
    def empty(*, h: int, w: int) -> SegmentationResults:
        return SegmentationResults(
            ids=torch.zeros((0,), dtype=torch.int64),
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
        )


    # Add utility methods to save this to file
    def saveToFile(self, filePath: str, imagePath: Optional[str] = None):
        """Saves segmentation results to a .npz file."""

        np.savez_compressed(
            filePath,
            ids=self.ids.cpu().numpy(),
            masks=self.masks.cpu().numpy(),
            boxes=self.boxes.cpu().numpy(),
            scores=self.scores.cpu().numpy() if self.scores is not None else None,
            imagePath=imagePath
        )

    # Add another utility method to load from file
    @staticmethod
    def loadFromFile(filePath: str) -> SegmentationResults:
        """Loads segmentation results from a .npz file."""
        # data = np.load(filePath, allow_pickle=True)
        with np.load(filePath, allow_pickle=True) as data:
            ids = torch.from_numpy(data['ids'])
            masks = torch.from_numpy(data['masks'])
            boxes = torch.from_numpy(data['boxes'])
            scores = torch.from_numpy(data['scores']) if data['scores'] is not None else None
            imagePath = data['imagePath'].item() if 'imagePath' in data else None


            segmentationResults = SegmentationResults(
                ids=ids,
                masks=masks,
                boxes=boxes,
                scores=scores
            )

            return segmentationResults, imagePath

    def divideIntoSubsets(self, max_subset_size):
        """
        Divides the segmentation results into subsets of at most max_subset_size instances.
        Returns a list of SegmentationResults objects.
        """

        numGroups = ceil(len(self) / max_subset_size)
        targetGroupSize = ceil(len(self) / numGroups)

        subsets = []

        n = len(self)
        for i in range(0, n, targetGroupSize):
            subset = SegmentationResults(
                ids=self.ids[i:i+targetGroupSize],
                masks=self.masks[i:i+targetGroupSize],
                boxes=self.boxes[i:i+targetGroupSize],
                scores=self.scores[i:i+targetGroupSize] if self.scores is not None else None,
            )
            subsets.append(subset)
        return subsets

    def divideIntoSpatialSubsets(
        self,
        max_subset_size,
        *,
        iou_threshold: float = 0.2,
        dilation_iou_threshold: float = 0.15,
        dilation_frac: float = 0.02,
        dilation_min_px: int = 3,
        dilation_max_px: int = 8,
        union_size_cap_frac: float = 0.5,
    ):
        """
        Spatially-aware grouping into subsets of at most ``max_subset_size`` instances.

        Two-step:
          1. Build "must-stick" connected components via union-find on:
             - mask IoU > iou_threshold (true overlap), and
             - dilated-mask IoU > dilation_iou_threshold (touching / near-touching),
               with per-mask dilation radius
               clip(dilation_frac * avg_bbox_dim, dilation_min_px, dilation_max_px).
             A bbox-IoU prefilter limits mask work to candidate pairs.
             To prevent single-link chaining (e.g. dense retail shelves merging
             into one giant component), unions are skipped when they would push
             the resulting component above
             ceil(max_subset_size * union_size_cap_frac) members.
          2. Recursive proportional median-split on super-nodes
             (one super-node per component, weighted by member count, positioned
             at the mean bbox centroid). Splits along the longer-spread axis at
             the weight-proportional boundary, targeting ceil(N / max_subset_size)
             groups. If a single component exceeds max_subset_size on its own,
             it is split internally on its members' centroids.

        Returns a list of SegmentationResults whose unions partition self.
        """
        n = len(self)
        if n == 0:
            return []
        if n <= max_subset_size:
            return [self]

        boxes = self.boxes.float()
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        widths = (x2 - x1).clamp(min=0)
        heights = (y2 - y1).clamp(min=0)
        avg_dim = (widths + heights) / 2.0
        bbox_areas = widths * heights

        # Pairwise bbox IoU (vectorized) — used as a cheap candidate filter for
        # the more expensive mask passes below.
        ix1 = torch.maximum(x1[:, None], x1[None, :])
        iy1 = torch.maximum(y1[:, None], y1[None, :])
        ix2 = torch.minimum(x2[:, None], x2[None, :])
        iy2 = torch.minimum(y2[:, None], y2[None, :])
        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        bbox_inter = iw * ih
        bbox_union = bbox_areas[:, None] + bbox_areas[None, :] - bbox_inter
        bbox_iou = torch.where(bbox_union > 0, bbox_inter / bbox_union, torch.zeros_like(bbox_union))

        # Per-mask dilation radius in pixels (used for the touching check).
        dilation_radii = torch.clamp(
            avg_dim * dilation_frac, min=dilation_min_px, max=dilation_max_px
        ).round().to(torch.int64)
        max_radius = int(dilation_radii.max().item()) if n > 0 else 0

        # Candidate pairs: bboxes overlap OR are within max_radius pixels of each
        # other on both axes (so a touching union is still possible).
        gap_x = torch.maximum(x1[:, None] - x2[None, :], x1[None, :] - x2[:, None]).clamp(min=0)
        gap_y = torch.maximum(y1[:, None] - y2[None, :], y1[None, :] - y2[:, None]).clamp(min=0)
        candidate = (bbox_inter > 0) | ((gap_x <= 2 * max_radius) & (gap_y <= 2 * max_radius))
        # Only consider upper triangle to avoid duplicate work.
        candidate = torch.triu(candidate, diagonal=1)
        cand_pairs = candidate.nonzero(as_tuple=False)

        parent = list(range(n))
        comp_size = [1] * n
        union_size_cap = max(1, ceil(max_subset_size * union_size_cap_frac))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if comp_size[ra] + comp_size[rb] > union_size_cap:
                return False
            # Attach smaller under larger to keep the tree shallow.
            if comp_size[ra] < comp_size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            comp_size[ra] += comp_size[rb]
            return True

        masks = self.masks
        # Materialize once for bool reductions.
        if masks.dtype != torch.bool:
            masks_bool = masks.to(torch.bool)
        else:
            masks_bool = masks
        mask_areas = masks_bool.reshape(n, -1).sum(dim=1).to(torch.float32)

        for pair_idx in range(cand_pairs.shape[0]):
            i = int(cand_pairs[pair_idx, 0].item())
            j = int(cand_pairs[pair_idx, 1].item())
            if find(i) == find(j):
                continue

            # Pass 1: true overlap (mask IoU > iou_threshold).
            inter = (masks_bool[i] & masks_bool[j]).sum().to(torch.float32)
            if inter.item() > 0:
                union_area = mask_areas[i] + mask_areas[j] - inter
                if union_area.item() > 0:
                    miou = (inter / union_area).item()
                    if miou > iou_threshold:
                        union(i, j)
                        continue

            # Pass 2: touching via per-mask dilation. Dilate only the local
            # crop covering both masks plus the larger of the two radii to
            # bound work.
            ri = int(dilation_radii[i].item())
            rj = int(dilation_radii[j].item())
            r = max(ri, rj)
            if r <= 0:
                continue
            H, W = masks_bool.shape[1], masks_bool.shape[2]
            # Tight bbox covering both, padded by r on each side.
            xa = max(0, int(min(x1[i].item(), x1[j].item())) - r)
            ya = max(0, int(min(y1[i].item(), y1[j].item())) - r)
            xb = min(W, int(max(x2[i].item(), x2[j].item())) + r)
            yb = min(H, int(max(y2[i].item(), y2[j].item())) + r)
            if xb <= xa or yb <= ya:
                continue
            crop_i = masks_bool[i, ya:yb, xa:xb]
            crop_j = masks_bool[j, ya:yb, xa:xb]
            # max_pool2d-based dilation on each crop with its own radius.
            di = _dilate_mask_2d(crop_i, ri) if ri > 0 else crop_i
            dj = _dilate_mask_2d(crop_j, rj) if rj > 0 else crop_j
            d_inter = (di & dj).sum().to(torch.float32)
            if d_inter.item() == 0:
                continue
            d_union = di.sum().to(torch.float32) + dj.sum().to(torch.float32) - d_inter
            if d_union.item() <= 0:
                continue
            d_iou = (d_inter / d_union).item()
            if d_iou > dilation_iou_threshold:
                union(i, j)

        # Build components: list of member-index lists keyed by root.
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        components = list(comps.values())

        # Per-component centroid (mean of member centroids) and weight.
        comp_cx = torch.tensor([float(cx[m].mean().item()) for m in components])
        comp_cy = torch.tensor([float(cy[m].mean().item()) for m in components])
        comp_weight = torch.tensor([len(m) for m in components], dtype=torch.int64)

        target_num_groups = ceil(n / max_subset_size)

        # Step 2a: split components across `target_num_groups` chunks via
        # weight-proportional recursive median split. Components are atomic at
        # this stage; oversized components are handled in step 2b.
        def split_components(comp_indices: list, k: int) -> list:
            if k <= 1 or len(comp_indices) == 0:
                return [comp_indices]
            xs = comp_cx[comp_indices]
            ys = comp_cy[comp_indices]
            spread_x = float(xs.max() - xs.min()) if len(xs) > 1 else 0.0
            spread_y = float(ys.max() - ys.min()) if len(ys) > 1 else 0.0
            coords = xs if spread_x >= spread_y else ys
            order = torch.argsort(coords)
            sorted_idx = [comp_indices[int(o.item())] for o in order]
            k_left = k // 2
            k_right = k - k_left
            total_w = sum(int(comp_weight[c].item()) for c in sorted_idx)
            target_left_w = total_w * k_left / k
            cum = 0
            boundary = 0
            for pos, ci in enumerate(sorted_idx):
                cum += int(comp_weight[ci].item())
                if cum >= target_left_w:
                    boundary = pos + 1
                    break
            boundary = max(1, min(len(sorted_idx) - 1, boundary)) if len(sorted_idx) > 1 else len(sorted_idx)
            left = sorted_idx[:boundary]
            right = sorted_idx[boundary:]
            return split_components(left, k_left) + split_components(right, k_right)

        comp_groups = split_components(list(range(len(components))), target_num_groups)

        # Step 2b: flatten each comp_group to member indices; if any group's
        # total weight exceeds max_subset_size, recursively split its members
        # by centroid.
        def split_member_indices(member_indices: list) -> list:
            if len(member_indices) <= max_subset_size:
                return [member_indices]
            xs = cx[member_indices]
            ys = cy[member_indices]
            spread_x = float(xs.max() - xs.min()) if len(xs) > 1 else 0.0
            spread_y = float(ys.max() - ys.min()) if len(ys) > 1 else 0.0
            coords = xs if spread_x >= spread_y else ys
            order = torch.argsort(coords)
            sorted_members = [member_indices[int(o.item())] for o in order]
            k = ceil(len(sorted_members) / max_subset_size)
            k_left = k // 2
            k_right = k - k_left
            boundary = (len(sorted_members) * k_left) // k
            boundary = max(1, min(len(sorted_members) - 1, boundary))
            return split_member_indices(sorted_members[:boundary]) + split_member_indices(sorted_members[boundary:])

        final_member_groups = []
        for cg in comp_groups:
            members = []
            for ci in cg:
                members.extend(components[ci])
            if not members:
                continue
            if len(members) <= max_subset_size:
                final_member_groups.append(members)
            else:
                final_member_groups.extend(split_member_indices(members))

        subsets = []
        for grp in final_member_groups:
            idx = torch.tensor(grp, dtype=torch.int64)
            subset = SegmentationResults(
                ids=self.ids[idx],
                masks=self.masks[idx],
                boxes=self.boxes[idx],
                scores=self.scores[idx] if self.scores is not None else None,
            )
            subsets.append(subset)
        return subsets


class AbsInstanceSegmenter(ABC):
    """Abstract base class for single-image instance segmentation models."""


    def __init__(
        self,
        name: str,
        *,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
    ):
        self.name = name
        self.min_area_perc = float(min_area_perc)
        self.dedupe_iou_threshold = float(dedupe_iou_threshold)
        self.confidence_threshold = None if confidence_threshold is None else float(confidence_threshold)

        if self.min_area_perc < 0:
            raise ValueError("min_area_perc must be >= 0.")
        if self.dedupe_iou_threshold < 0:
            raise ValueError("dedupe_iou_threshold must be >= 0.")


    def getName(self) -> str:
        """Returns the name of this segmenter."""
        return self.name

    def segmentInstances(self, image: torch.Tensor, image_id: str) -> Tuple[SegmentationResults, DebugPayload]:
        """
        Args:
            image: torch.Tensor, shape (C, H, W), dtype=uint8, RGB.
            image_id: stable identifier for this image (used by debug-output
                paths).

        Returns:
            (SegmentationResults, DebugPayload). The caller (the logger) is
            responsible for calling ``flush()`` on the payload once it has
            been written.
        """
        self._validateImage(image)
        debug_payload = DebugPayloadFactory.instance().create("segmentation", image_id)
        # ``segmentInstancesRaw`` still returns ``(results, sub_debug)`` for
        # backward compat with subclasses. No segmenter currently produces
        # actual debug data — sub_debug is typically ``DebugPayload.empty()``
        # — but if a subclass later returns real entries, merging here lets
        # the configured payload pick them up under the segmentation phase
        # without changing every subclass signature.
        raw_results, sub_debug = self.segmentInstancesRaw(image)
        if not isinstance(raw_results, SegmentationResults):
            raise TypeError("segmentInstancesRaw() must return SegmentationResults.")
        debug_payload.merge(sub_debug)
        processed = self._postprocessSegmentationResults(image=image, results=raw_results)
        return processed, debug_payload

    @abstractmethod
    def segmentInstancesRaw(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        """
        Model-specific segmentation implementation.

        Subclasses should return (unfiltered SegmentationResults in image space, DebugPayload).
        Subclasses with no debug data may return (results, DebugPayload.empty()) or (results, None).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _validateImage(self, image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image of shape (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

    def _postprocessSegmentationResults(
        self,
        *,
        image: torch.Tensor,
        results: SegmentationResults,
    ) -> SegmentationResults:
        _, h, w = image.shape
        processed = self._normalizeSegmentationResults(results, h=h, w=w)

        # Print the number of masks before filtering
        # print(f"Number of masks before filtering: {len(processed)}")

        if len(processed) == 0:
            return processed

        # 1) Area filtering
        min_area_pixels = int(self.min_area_perc * h * w)
        if min_area_pixels > 0:
            keep = torch.tensor(
                filter_mask_indices_by_min_area(
                    masks=processed.masks,
                    min_area_pixels=min_area_pixels,
                ),
                dtype=torch.long,
                device=processed.ids.device,
            )
            processed = self._selectByIndices(processed, keep)
            if len(processed) == 0:
                return SegmentationResults.empty(h=h, w=w)

        # Print the number of masks after area filtering
        # print(f"Number of masks after area filtering: {len(processed)}")

        # 2) IoU-based deduplication
        dedupe_keep = dedupe_mask_indices_by_iou(
            masks=processed.masks,
            scores=processed.scores,
            iou_threshold=self.dedupe_iou_threshold,
        )
        processed = self._selectByIndices(
            processed, torch.tensor(dedupe_keep, dtype=torch.long, device=processed.ids.device)
        )
        # Print the number of masks after deduplication
        print(f"Number of masks after deduplication: {len(processed)}")
        if len(processed) == 0:
            return SegmentationResults.empty(h=h, w=w)

        # 3) Confidence thresholding
        if self.confidence_threshold is not None and processed.scores is not None:
            keep = (processed.scores >= self.confidence_threshold).nonzero(as_tuple=True)[0]
            processed = self._selectByIndices(processed, keep)

        return processed

    def _normalizeSegmentationResults(self, results: SegmentationResults, *, h: int, w: int) -> SegmentationResults:
        ids = results.ids.to(dtype=torch.int64)
        masks = results.masks.to(dtype=torch.bool)
        boxes = results.boxes.to(dtype=torch.float32)
        scores = results.scores.to(dtype=torch.float32) if results.scores is not None else None

        if masks.ndim != 3:
            raise ValueError("Expected masks to have shape (N, H, W).")
        n = masks.shape[0]
        if masks.shape[1] != h or masks.shape[2] != w:
            raise ValueError(f"Expected masks shape (N, {h}, {w}); got {tuple(masks.shape)}.")
        if ids.ndim != 1 or ids.shape[0] != n:
            raise ValueError("Expected ids shape (N,).")
        if boxes.ndim != 2 or boxes.shape[0] != n or boxes.shape[1] != 4:
            raise ValueError("Expected boxes shape (N, 4).")
        if scores is not None and (scores.ndim != 1 or scores.shape[0] != n):
            raise ValueError("Expected scores shape (N,) when provided.")

        return SegmentationResults(ids=ids, masks=masks, boxes=boxes, scores=scores)

    def _selectByIndices(self, results: SegmentationResults, indices: torch.Tensor) -> SegmentationResults:
        return SegmentationResults(
            ids=results.ids[indices],
            masks=results.masks[indices],
            boxes=results.boxes[indices],
            scores=results.scores[indices] if results.scores is not None else None,
        )

