from __future__ import annotations

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_parameter_networks.inference.mask_poolers import MaskPooler


# Maps a label_transform target string to the output keys the decoded
# values should be reported under. Each entry is (short_low, short_high,
# long_low, long_high). The long-form keys match the dataset's raw
# percentile fields (what the LLM pipeline emits and what the
# weibull_pucpr_* configs expect); short-form aliases are emitted too so
# survival generators that key on `t50`/`t90` etc. continue to work.
_LABEL_TRANSFORM_KEYS: dict[str, Tuple[str, str, str, str]] = {
    "training.dataset.distillation_dataset.label_transform_log_ratio": (
        "t50", "t90", "median_persistence_hours", "p90_persistence_hours",
    ),
    "training.dataset.distillation_dataset.label_transform_log_ratio_p25_p75": (
        "t25", "t75", "p25_persistence_hours", "p75_persistence_hours",
    ),
}
_DEFAULT_TRANSFORM = "training.dataset.distillation_dataset.label_transform_log_ratio"


class LearnedModelInferencer(AbsPersistenceParameterInferencer):
    """Persistence parameter inferencer backed by a trained checkpoint.

    Architecture-agnostic: the same code path runs both per-mask models
    (``PersistenceObjConditionedCrossAttn``) and dense models
    (``DensePersistencePredictor`` subclasses) via the
    :func:`training.task.build_validation_model` factory. The factory
    returns a :class:`~training.task.MaskedValidationModel` whose
    ``forward_multi(image, masks)`` produces ``(K, 3)`` so we do one
    forward per image even when the group has many masks.

    Output-key resolution is driven by the ``label_transform._target_``
    embedded in the checkpoint's ``trainer_cfg`` YAML. ``output_param_keys``
    in the Hydra config overrides this when set (legacy/manual escape
    hatch).
    """

    def __init__(
        self,
        model_cfg,
        checkpoint_path: str,
        device: str = "cuda",
        output_param_keys: Optional[list[str]] = None,
        pooler: Optional[MaskPooler] = None,
        name: Optional[str] = None,
    ) -> None:
        # Surfaced by BenchmarkOrchestrator.from_cfg as the method-name
        # folder under persistence_eval_outputs/<dataset>/<name>/...; set
        # via cfg.inferencer.name in the per-checkpoint eval config so
        # different checkpoints don't collide in one folder.
        self.name = name
        import hydra.utils
        import torch.nn as nn
        from training.checkpointing import load_model_from_checkpoint
        from training.task import build_validation_model

        self._device = torch.device(device)
        # BenchmarkOrchestrator recursively instantiates cfg.model before
        # passing it in, so model_cfg typically arrives as a live nn.Module.
        # When invoked outside the orchestrator (e.g. tests) it may still be
        # a DictConfig; instantiate in that case.
        if isinstance(model_cfg, nn.Module):
            model = model_cfg
        else:
            model = hydra.utils.instantiate(model_cfg)
        model.to(self._device)
        model.eval()
        self._ckpt_meta = load_model_from_checkpoint(
            model, checkpoint_path, map_location=self._device, strict=True,
        )

        # Wrap the bare model in the per-mask/dense-aware validation
        # model so infer() doesn't have to branch on architecture. When
        # the checkpoint was trained with normalized targets, the same
        # stats live in the embedded trainer_cfg YAML and the factory
        # adds a denormalizing wrapper so outputs land in log-space.
        norm_stats = self._resolve_norm_stats()
        self._val_model = build_validation_model(model, pooler=pooler, **norm_stats)
        self._val_model.to(self._device).eval()

        (
            self._lo_keys,
            self._hi_keys,
            self._confidence_from_logits,
        ) = self._resolve_decode(output_param_keys)
        self._img_size = self._resolve_img_size()
        self._transform = self._build_transform(self._img_size)

    @staticmethod
    def _build_transform(size: int):
        # Match training/dataset/distillation_dataset.make_default_dinov2_transform:
        # resize to (size, size), uint8 → float32 in [0,1], ImageNet normalize.
        from training.dataset.distillation_dataset import make_default_dinov2_transform
        return make_default_dinov2_transform(size)

    def _resolve_img_size(self) -> int:
        # Trainer YAML stores img_size at trainer.loop.img_size; default 518
        # (matches make_default_dinov2_transform default). Mismatched img_size
        # will silently misalign the model with its checkpoint, so prefer
        # whatever the checkpoint was trained at.
        trainer_cfg_yaml = getattr(self._ckpt_meta, "trainer_cfg", None)
        if trainer_cfg_yaml:
            try:
                cfg = OmegaConf.create(trainer_cfg_yaml)
                img_size = OmegaConf.select(cfg, "trainer.loop.img_size")
                if img_size is not None:
                    return int(img_size)
            except Exception as exc:
                print(
                    f"[LearnedModelInferencer] Could not read trainer.loop.img_size "
                    f"from trainer_cfg ({exc!r}); falling back to 518."
                )
        return 518

    def _resolve_norm_stats(self) -> dict:
        """Pull target z-score stats from the checkpoint's embedded
        trainer_cfg YAML. Returns ``{}`` for non-`_norm` checkpoints
        (no wrapping applied); returns all four when present.
        """
        keys = ("y50_mean", "y50_std", "yr_mean", "yr_std")
        trainer_cfg_yaml = getattr(self._ckpt_meta, "trainer_cfg", None)
        if not trainer_cfg_yaml:
            return {}
        try:
            cfg = OmegaConf.create(trainer_cfg_yaml)
            values = {k: OmegaConf.select(cfg, f"train_dataset.{k}") for k in keys}
        except Exception as exc:
            print(
                f"[LearnedModelInferencer] Could not read target-norm stats "
                f"from trainer_cfg ({exc!r}); assuming non-normalized checkpoint."
            )
            return {}
        if all(v is None for v in values.values()):
            return {}
        if any(v is None for v in values.values()):
            raise ValueError(
                f"Checkpoint's trainer_cfg has a partial target-norm stat set: "
                f"{values}. Either all four (y50_mean/std, yr_mean/std) must "
                f"be present, or none."
            )
        return values

    def _resolve_decode(
        self, output_param_keys_override: Optional[list[str]]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool]:
        """Pick the (low_keys, high_keys) tuples and confidence_from_logits.

        Each "keys" tuple lists every alias the decoded low/high value
        should be reported under (short + long form), so any downstream
        survival-generator config — keyed on ``t50``/``t90`` or
        ``p25_persistence_hours``/``p75_persistence_hours`` — finds what
        it expects in ``estimate.params``.

        Order: explicit override → embedded trainer_cfg → log_ratio default.
        """
        confidence_from_logits = False
        short_lo, short_hi, long_lo, long_hi = _LABEL_TRANSFORM_KEYS[_DEFAULT_TRANSFORM]

        trainer_cfg_yaml = getattr(self._ckpt_meta, "trainer_cfg", None)
        if trainer_cfg_yaml:
            try:
                cfg = OmegaConf.create(trainer_cfg_yaml)
                transform_target = OmegaConf.select(cfg, "train_dataset.label_transform._target_")
                if transform_target in _LABEL_TRANSFORM_KEYS:
                    short_lo, short_hi, long_lo, long_hi = _LABEL_TRANSFORM_KEYS[transform_target]
                elif transform_target is not None:
                    print(
                        f"[LearnedModelInferencer] Unknown label_transform target "
                        f"{transform_target!r}; falling back to {_DEFAULT_TRANSFORM!r}."
                    )
                conf_logits = OmegaConf.select(
                    cfg, "trainer.loss_function.confidence_from_logits"
                )
                if conf_logits is not None:
                    confidence_from_logits = bool(conf_logits)
            except Exception as exc:
                print(
                    f"[LearnedModelInferencer] Could not parse trainer_cfg from checkpoint "
                    f"({exc!r}); falling back to defaults t50/t90, conf-from-probs."
                )

        if output_param_keys_override:
            if len(output_param_keys_override) != 2:
                raise ValueError(
                    "output_param_keys override must be a 2-list [low_key, high_key]; "
                    f"got {output_param_keys_override!r}."
                )
            # Override replaces all aliases — caller wants exactly these keys.
            return (
                (output_param_keys_override[0],),
                (output_param_keys_override[1],),
                confidence_from_logits,
            )

        return (short_lo, long_lo), (short_hi, long_hi), confidence_from_logits

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        # Transform expects a CHW uint8 tensor and resizes to img_size.
        img_tensor = self._transform(group.image).unsqueeze(0).to(self._device)  # (1, 3, S, S)

        # Masks must be resized to the same (S, S) the model was trained at.
        # NEAREST avoids fractional values that would corrupt the binary mask.
        masks = group.masks.unsqueeze(1).float()                  # (K, 1, H, W)
        masks = resize(
            masks,
            [self._img_size, self._img_size],
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(1).to(self._device)                              # (K, S, S)

        with torch.no_grad():
            out = self._val_model.forward_multi(img_tensor, masks)          # (K, 3)

        estimates = [
            PersistenceParameterEstimate(params=self._decode_outputs(out[k]))
            for k in range(out.shape[0])
        ]
        return estimates, None

    def _decode_outputs(self, row: torch.Tensor) -> dict[str, float]:
        # row: (3,) — columns [y_low, y_ratio, y_conf]. Inverse of
        # label_transform_log_ratio{,_p25_p75}:
        #   t_low  = exp(y_low)
        #   t_high = t_low * (exp(y_ratio) + 1)
        #   conf   = sigmoid(y_conf) if trained with logits, else y_conf
        y_low, y_ratio, y_conf = row[0], row[1], row[2]
        t_low = float(torch.exp(y_low))
        t_high = float(t_low * float(torch.exp(y_ratio) + 1.0))
        conf = float(torch.sigmoid(y_conf) if self._confidence_from_logits else y_conf)
        params: dict[str, float] = {"confidence": conf}
        for k in self._lo_keys:
            params[k] = t_low
        for k in self._hi_keys:
            params[k] = t_high
        return params
