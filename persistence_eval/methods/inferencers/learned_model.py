from __future__ import annotations

import time

import numpy as np
import torch
from omegaconf import DictConfig

from persistence_eval.core.domain import AnchorObservation, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer


class LearnedModelInferencer(AbsPersistenceParameterInferencer):
    """
    Persistence parameter inferencer backed by a trained neural network checkpoint.
    Loaded via Hydra instantiation; model_cfg is passed from cfg.model by the orchestrator.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        checkpoint_path: str,
        device: str = "cuda",
        output_param_keys: list[str] | None = None,
    ) -> None:
        import hydra.utils
        from training.checkpointing import load_training_state_from_checkpoint

        self._device = torch.device(device)
        self._output_param_keys = output_param_keys or ["t50", "t90"]
        self._model = hydra.utils.instantiate(model_cfg)
        self._model.to(self._device)
        self._model.eval()
        load_training_state_from_checkpoint(self._model, checkpoint_path)
        self._transform = self._build_transform()

    @staticmethod
    def _build_transform():
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def infer(self, anchor: AnchorObservation) -> PersistenceParameterEstimate:
        estimates = self.infer_batch(
            image=anchor.image,
            masks=[anchor.mask],
            anchor_time=anchor.anchor_time,
            track_ids=[anchor.track_id],
        )
        return estimates[0]

    def infer_batch(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        anchor_time: float,
        track_ids: list[str],
    ) -> list[PersistenceParameterEstimate]:
        from training.dataset.distillation_dataset import label_transform_log_ratio

        img_tensor = self._transform(image).unsqueeze(0).to(self._device)  # 1×3×H×W

        results = []
        with torch.no_grad():
            for mask in masks:
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self._device)
                y50_hat, yr_hat, yc_hat = self._model(img_tensor, mask_tensor)
                params = self._decode_outputs(y50_hat, yr_hat, yc_hat, label_transform_log_ratio)
                results.append(PersistenceParameterEstimate(params=params))
        return results

    def _decode_outputs(self, y50_hat, yr_hat, yc_hat, label_transform_log_ratio) -> dict[str, float]:
        # Inverse of label_transform_log_ratio: model outputs are in log-ratio space
        # Exact mapping is TBD pending review of distillation_dataset implementation
        raise NotImplementedError("Output decoding not yet implemented — see label_transform_log_ratio")
