"""Unit tests for the dense target rasterizer family + cache.

Hand-crafted 2-instance overlap scenario:

    Image is a 4x4 grid.
    Instance 0 occupies rows 0-2 (12 pixels), label = [1.0, 2.0, 0.0].
    Instance 1 occupies rows 2-3 (8 pixels),  label = [5.0, 6.0, 1.0].
    Row 2 is the 4-pixel overlap.

We check each overlap policy's target_map/loss_mask/pixel_weights.

Also verifies the disk cache hits on a second call.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

from persistence_parameter_networks.data.dense_target_rasterizer import (
    AreaWeightedAverageRasterizer,
    AverageOverlapRasterizer,
    ExcludeOverlapRasterizer,
    LargerWinsRasterizer,
    NoNormalization,
    PerInstanceAreaNormalizer,
    SmallerWinsRasterizer,
)
from persistence_parameter_networks.data.dense_supervision_cache import (
    DenseSupervisionCache,
)


def _two_instance_setup():
    H, W = 4, 4
    m0 = torch.zeros((H, W), dtype=torch.uint8)
    m0[0:3, :] = 1  # rows 0,1,2 — area 12
    m1 = torch.zeros((H, W), dtype=torch.uint8)
    m1[2:4, :] = 1  # rows 2,3 — area 8
    masks = torch.stack([m0, m1], dim=0)
    labels = torch.tensor([[1.0, 2.0, 0.0], [5.0, 6.0, 1.0]], dtype=torch.float32)
    return masks, labels


def _row(target_map, r):
    return target_map[:, r, :].mean(dim=-1).tolist()


def test_smaller_wins() -> None:
    masks, labels = _two_instance_setup()
    r = SmallerWinsRasterizer(NoNormalization())
    target, loss_mask, pixel_weights = r.rasterize(masks, labels)

    # Row 0/1: only instance 0 covers -> label0
    assert _row(target, 0) == [1.0, 2.0, 0.0]
    assert _row(target, 1) == [1.0, 2.0, 0.0]
    # Row 2: overlap -> instance 1 (smaller area = 8 < 12)
    assert _row(target, 2) == [5.0, 6.0, 1.0]
    # Row 3: only instance 1 -> label1
    assert _row(target, 3) == [5.0, 6.0, 1.0]
    # Loss mask: all pixels are valid (union of both)
    assert loss_mask.sum().item() == 16
    # Pixel weights with NoNormalization: 1 everywhere valid
    assert pixel_weights.sum().item() == 16.0
    print("[ok] SmallerWinsRasterizer")


def test_larger_wins() -> None:
    masks, labels = _two_instance_setup()
    r = LargerWinsRasterizer(NoNormalization())
    target, _, _ = r.rasterize(masks, labels)
    # Row 2 (overlap) -> instance 0 (larger area = 12 > 8)
    assert _row(target, 2) == [1.0, 2.0, 0.0]
    print("[ok] LargerWinsRasterizer")


def test_exclude_overlap() -> None:
    masks, labels = _two_instance_setup()
    r = ExcludeOverlapRasterizer(NoNormalization())
    target, loss_mask, _ = r.rasterize(masks, labels)
    # Row 2 should be excluded -> loss_mask=0 there, target=0
    assert loss_mask[0, 2, :].sum().item() == 0, loss_mask[0, 2, :]
    assert _row(target, 2) == [0.0, 0.0, 0.0]
    # Other rows: valid
    assert loss_mask[0, 0:2, :].sum().item() == 8
    assert loss_mask[0, 3, :].sum().item() == 4
    print("[ok] ExcludeOverlapRasterizer")


def test_average_overlap() -> None:
    masks, labels = _two_instance_setup()
    r = AverageOverlapRasterizer(NoNormalization())
    target, _, _ = r.rasterize(masks, labels)
    # Row 2: mean of [1,2,0] and [5,6,1] = [3,4,0.5]
    assert _row(target, 2) == [3.0, 4.0, 0.5]
    print("[ok] AverageOverlapRasterizer")


def test_area_weighted_average() -> None:
    masks, labels = _two_instance_setup()
    r = AreaWeightedAverageRasterizer(NoNormalization())
    target, _, _ = r.rasterize(masks, labels)
    # Row 2 weighted by 1/area:
    #   w0 = 1/12, w1 = 1/8
    #   target = (w0*[1,2,0] + w1*[5,6,1]) / (w0 + w1)
    w0, w1 = 1.0 / 12.0, 1.0 / 8.0
    num = [w0 * 1 + w1 * 5, w0 * 2 + w1 * 6, w0 * 0 + w1 * 1]
    den = w0 + w1
    expected = [n / den for n in num]
    row2 = _row(target, 2)
    for got, want in zip(row2, expected):
        assert abs(got - want) < 1e-6, (got, want)
    print("[ok] AreaWeightedAverageRasterizer")


def test_per_instance_area_normalizer() -> None:
    masks, labels = _two_instance_setup()
    r = SmallerWinsRasterizer(PerInstanceAreaNormalizer())
    _, _, pixel_weights = r.rasterize(masks, labels)
    # Row 0/1 (owned by inst 0, area 12) -> weight 1/12
    assert abs(pixel_weights[0, 0, 0].item() - 1.0 / 12.0) < 1e-6
    assert abs(pixel_weights[0, 1, 0].item() - 1.0 / 12.0) < 1e-6
    # Row 2 (owned by inst 1, smaller wins, area 8) -> weight 1/8
    assert abs(pixel_weights[0, 2, 0].item() - 1.0 / 8.0) < 1e-6
    # Row 3 (inst 1) -> 1/8
    assert abs(pixel_weights[0, 3, 0].item() - 1.0 / 8.0) < 1e-6
    print("[ok] PerInstanceAreaNormalizer")


def test_cache_hit() -> None:
    masks, labels = _two_instance_setup()
    r = SmallerWinsRasterizer(NoNormalization())
    with tempfile.TemporaryDirectory() as tmp:
        cache = DenseSupervisionCache(
            root_dir=Path(tmp),
            rasterizer=r,
            label_transform=lambda d: d,  # placeholder
            enable=True,
        )
        # First call computes + writes.
        out1 = cache.get(
            image_id="img-001",
            instance_ids=[0, 1],
            instance_masks=masks,
            instance_labels=labels,
        )
        path = cache._entry_path("img-001", [0, 1])
        assert path is not None and path.exists(), path
        # Manifest exists.
        manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        assert cache.fingerprint in manifest
        # Second call returns cached tensors (same content).
        out2 = cache.get(
            image_id="img-001",
            instance_ids=[0, 1],
            instance_masks=masks,
            instance_labels=labels,
        )
        for a, b in zip(out1, out2):
            assert torch.equal(a, b)
        # Different instance_ids should produce a different cache file.
        path_alt = cache._entry_path("img-001", [0])
        assert path != path_alt
    print("[ok] DenseSupervisionCache hits + per-instance-set keying")


def main() -> int:
    test_smaller_wins()
    test_larger_wins()
    test_exclude_overlap()
    test_average_overlap()
    test_area_weighted_average()
    test_per_instance_area_normalizer()
    test_cache_hit()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
