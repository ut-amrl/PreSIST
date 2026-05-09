from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from persistence_eval.core.domain import PersistenceParameterEstimate


_CONFIDENCE_CHANNEL = "confidence"


@dataclass
class PixelPersistenceMap:
    keys: tuple[str, ...]
    data: torch.Tensor
    coverage: torch.Tensor

    def channel_index(self, name: str) -> int:
        if name == _CONFIDENCE_CHANNEL:
            return len(self.keys)
        return self.keys.index(name)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


def _flatten_params(
    params: dict[str, float | list[float]],
    schema: list[tuple[str, int | None]],
) -> list[float]:
    flat: list[float] = []
    for key, length in schema:
        value = params[key]
        if length is None:
            flat.append(float(value))  # type: ignore[arg-type]
        else:
            assert isinstance(value, list) and len(value) == length
            flat.extend(float(v) for v in value)
    return flat


def _build_schema(
    estimates: Sequence[PersistenceParameterEstimate],
    default: PersistenceParameterEstimate,
) -> tuple[list[tuple[str, int | None]], tuple[str, ...]]:
    reference_keys = set(default.params.keys())
    for k, est in enumerate(estimates):
        if set(est.params.keys()) != reference_keys:
            raise ValueError(
                f"Estimate {k} has param keys {sorted(est.params.keys())}, expected"
                f" {sorted(reference_keys)}"
            )

    schema: list[tuple[str, int | None]] = []
    flat_keys: list[str] = []
    for key in sorted(reference_keys):
        ref_value = default.params[key]
        if isinstance(ref_value, list):
            length = len(ref_value)
            for k, est in enumerate(estimates):
                value = est.params[key]
                if not isinstance(value, list) or len(value) != length:
                    raise ValueError(
                        f"Param {key!r} has list length {length} in default but"
                        f" {value!r} in estimate {k}"
                    )
            schema.append((key, length))
            flat_keys.extend(f"{key}_{i}" for i in range(length))
        else:
            for k, est in enumerate(estimates):
                if isinstance(est.params[key], list):
                    raise ValueError(
                        f"Param {key!r} is scalar in default but list in estimate {k}"
                    )
            schema.append((key, None))
            flat_keys.append(key)

    return schema, tuple(flat_keys)


def build_pixel_persistence_map(
    masks: torch.Tensor,
    estimates: Sequence[PersistenceParameterEstimate],
    default: PersistenceParameterEstimate,
) -> PixelPersistenceMap:
    if masks.dtype != torch.bool:
        raise ValueError(f"masks must be bool, got {masks.dtype}")
    if masks.ndim != 3:
        raise ValueError(f"masks must be (K, H, W), got shape {tuple(masks.shape)}")
    K, H, W = masks.shape
    if len(estimates) != K:
        raise ValueError(f"Got {len(estimates)} estimates for {K} masks")

    schema, flat_keys = _build_schema(estimates, default)
    P = len(flat_keys)
    device = masks.device

    if K == 0:
        default_flat = _flatten_params(default.params, schema)
        default_conf = (
            float(default.confidence) if default.confidence is not None else math.nan
        )
        default_vec = torch.tensor(
            default_flat + [default_conf], dtype=torch.float32, device=device
        )
        data = default_vec.expand(H, W, P + 1).contiguous()
        coverage = torch.zeros((H, W), dtype=torch.int64, device=device)
        return PixelPersistenceMap(keys=flat_keys, data=data, coverage=coverage)

    values = torch.tensor(
        [_flatten_params(est.params, schema) for est in estimates],
        dtype=torch.float32,
        device=device,
    )  # (K, P)

    confidences = torch.tensor(
        [
            float(est.confidence) if est.confidence is not None else 0.0
            for est in estimates
        ],
        dtype=torch.float32,
        device=device,
    )  # (K,)
    has_conf = torch.tensor(
        [est.confidence is not None for est in estimates],
        dtype=torch.float32,
        device=device,
    )  # (K,)

    masks_f = masks.to(torch.float32)
    coverage = masks.sum(dim=0, dtype=torch.int64)  # (H, W)
    safe_cov = coverage.clamp(min=1).to(torch.float32).unsqueeze(-1)  # (H, W, 1)

    param_sum = torch.einsum("khw,kp->hwp", masks_f, values)  # (H, W, P)
    param_avg = param_sum / safe_cov  # (H, W, P)

    conf_sum = torch.einsum("khw,k->hw", masks_f, confidences * has_conf)  # (H, W)
    conf_count = torch.einsum("khw,k->hw", masks_f, has_conf)  # (H, W)
    conf_avg = torch.where(
        conf_count > 0,
        conf_sum / conf_count.clamp(min=1.0),
        torch.full_like(conf_sum, math.nan),
    )  # (H, W)

    default_flat = _flatten_params(default.params, schema)
    default_params = torch.tensor(default_flat, dtype=torch.float32, device=device)
    default_conf = (
        float(default.confidence) if default.confidence is not None else math.nan
    )

    uncovered = (coverage == 0).unsqueeze(-1)  # (H, W, 1)
    param_filled = torch.where(uncovered, default_params.expand_as(param_avg), param_avg)
    conf_filled = torch.where(
        coverage == 0,
        torch.full_like(conf_avg, default_conf),
        conf_avg,
    )

    data = torch.cat([param_filled, conf_filled.unsqueeze(-1)], dim=-1)  # (H, W, P+1)
    return PixelPersistenceMap(keys=flat_keys, data=data, coverage=coverage)


def _unflatten_params(
    flat_values: torch.Tensor,
    schema: list[tuple[str, int | None]],
) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = {}
    cursor = 0
    for key, length in schema:
        if length is None:
            out[key] = float(flat_values[cursor].item())
            cursor += 1
        else:
            out[key] = [float(flat_values[cursor + i].item()) for i in range(length)]
            cursor += length
    return out


def _schema_from_keys(keys: tuple[str, ...]) -> list[tuple[str, int | None]]:
    schema: list[tuple[str, int | None]] = []
    i = 0
    while i < len(keys):
        key = keys[i]
        if "_" in key and key.rsplit("_", 1)[-1].isdigit():
            base, idx = key.rsplit("_", 1)
            if int(idx) == 0:
                length = 1
                while (
                    i + length < len(keys)
                    and keys[i + length] == f"{base}_{length}"
                ):
                    length += 1
                schema.append((base, length))
                i += length
                continue
        schema.append((key, None))
        i += 1
    return schema


def lookup_pixel_persistence(
    pixel_map: PixelPersistenceMap,
    coords: torch.Tensor,
) -> list[PersistenceParameterEstimate]:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got shape {tuple(coords.shape)}")
    H, W = pixel_map.height, pixel_map.width
    ys = coords[:, 0].to(torch.long)
    xs = coords[:, 1].to(torch.long)
    if torch.any(ys < 0) or torch.any(ys >= H) or torch.any(xs < 0) or torch.any(xs >= W):
        raise IndexError(f"coords out of bounds for image of size ({H}, {W})")

    gathered = pixel_map.data[ys, xs, :]  # (N, P+1)
    coverages = pixel_map.coverage[ys, xs]  # (N,)
    schema = _schema_from_keys(pixel_map.keys)
    P = len(pixel_map.keys)

    results: list[PersistenceParameterEstimate] = []
    for n in range(gathered.shape[0]):
        params = _unflatten_params(gathered[n, :P], schema)
        conf_val = float(gathered[n, P].item())
        confidence = None if math.isnan(conf_val) else conf_val
        results.append(
            PersistenceParameterEstimate(
                params=params,
                confidence=confidence,
                metadata={
                    "coverage": int(coverages[n].item()),
                    "source": "pixel_map",
                },
            )
        )
    return results
