# Import pytorch dataloader
from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId
from torch.utils.data import DataLoader
from training.trainer import TrainerConfig, Trainer

from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId
from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath

import torch.nn as nn
import torch
from typing import Optional, Union, Dict, Any, Tuple
from training.validator import SurvivalValidationRunner, ValConfig
from training.dataset.distillation_dataset import DistillationDatasetConfig
from training.dataset.dense_collate import make_dense_collate
from training.task import (
    DensePersistenceTask,
    PerMaskPersistenceTask,
    TrainingTask,
)
from persistence_parameter_networks.models.dense_persistence_predictor import (
    DensePersistencePredictor,
)
from persistence_parameter_networks.losses.dense_losses import DenseWeightedMultiLoss
from persistence_parameter_networks.losses.weighted_multi_loss import WeightedMultiLoss

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import os
from datetime import datetime
import wandb

def train(
        task: TrainingTask,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        validator: SurvivalValidationRunner,
        output_dir: str,
        cfg: TrainerConfig,
        full_cfg_yaml: str,
        resumeFromCheckpoint: Optional[str] = None,
        val_on_train_loader: Optional[torch.utils.data.DataLoader] = None,
        ):

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        validator=validator,
        output_dir=output_dir,
        cfg=cfg,
        full_cfg_yaml=full_cfg_yaml,
        val_on_train_loader=val_on_train_loader,
    )

    if resumeFromCheckpoint is not None:
        print(f"Resuming training from checkpoint: {resumeFromCheckpoint}")
        trainer.resume_from_checkpoint(resumeFromCheckpoint)

    trainer.train()


def createDataloaders(distillationDatasetConfig: DistillationDatasetConfig, registryConfig: RegistryConfig, trainerConfig: TrainerConfig):

    print("[createDataloaders] building registry id<->relpath helpers...", flush=True)
    relpath_to_image_id = RelpathToImageId(
        image_root=distillationDatasetConfig.input_root_dir,
        registry_cfg=registryConfig,   # your existing Hydra RegistryConfig object
    )

    image_id_to_relpath = ImageIdToRelpath(
        image_root=distillationDatasetConfig.input_root_dir,
        registry_cfg=registryConfig,
        warn=print,  # or your logger.warning
    )

    print("[createDataloaders] building training dataset...", flush=True)
    trainingDataset = distillationDatasetConfig.make_dataset(
        split="train",
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=trainerConfig.img_size,
    )
    print("[createDataloaders] building validation dataset...", flush=True)
    validationDataset = distillationDatasetConfig.make_dataset(
        split="val",
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=trainerConfig.img_size,
    )

    # Create dataloaders. When image_balanced_sampling is enabled, draw
    # instances with weight = 1/n_instances_in_image so each image is
    # sampled with equal expected probability regardless of how many
    # instances it has.
    train_workers = getattr(trainerConfig, "num_workers", 4)
    val_workers = getattr(trainerConfig, "val_num_workers", None) or train_workers
    if getattr(trainerConfig, "image_balanced_sampling", False):
        weights = torch.tensor(trainingDataset._instance_weights, dtype=torch.double)
        per_epoch_samples = (
            int(trainerConfig.train_samples_per_epoch)
            if getattr(trainerConfig, "train_samples_per_epoch", None) is not None
            else len(trainingDataset)
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=per_epoch_samples,
            replacement=True,
        )
        n_distinct_imgs = len(set(rec_idx for rec_idx, _ in trainingDataset._instance_index))
        print(f"[createDataloaders] image_balanced_sampling on: "
              f"{n_distinct_imgs} distinct images, {len(trainingDataset)} instances; "
              f"samples_per_epoch={per_epoch_samples}")
        trainLoader = DataLoader(
            trainingDataset,
            batch_size=trainerConfig.batch_size,
            sampler=train_sampler,
            num_workers=train_workers,
            pin_memory=True,
            persistent_workers=train_workers > 0,
        )
    else:
        trainLoader = DataLoader(
            trainingDataset,
            batch_size=trainerConfig.batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=True,
            persistent_workers=train_workers > 0,
        )
    val_bs = getattr(trainerConfig, "val_batch_size", None) or trainerConfig.batch_size
    valLoader = DataLoader(
        validationDataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )

    return trainLoader, valLoader


def createDenseDataloaders(
    distillationDatasetConfig: DistillationDatasetConfig,
    registryConfig: RegistryConfig,
    trainerConfig: TrainerConfig,
    loss_fn: nn.Module,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three loaders for dense training.

    - ``train_loader``: dense mode + flat/concat collate. The collate
      includes per-instance tensors only when the configured loss
      requires them (``loss_fn.requires_per_instance_tensors()``), so
      the training hot path is decoupled from instance-count variance
      under the default loss.
    - ``val_loader``: per_instance mode — same population the per-mask
      model uses for validation, scored via the task's pooling facade
      so dense-vs-per-mask val numbers stay apples-to-apples.
    - ``val_on_train_loader``: per_instance mode on the train split.
      Separate from train_loader since the dense train batches aren't
      consumable by the existing validator.
    """
    print("[createDenseDataloaders] building registry id<->relpath helpers...", flush=True)
    relpath_to_image_id = RelpathToImageId(
        image_root=distillationDatasetConfig.input_root_dir,
        registry_cfg=registryConfig,
    )
    image_id_to_relpath = ImageIdToRelpath(
        image_root=distillationDatasetConfig.input_root_dir,
        registry_cfg=registryConfig,
        warn=print,
    )

    if distillationDatasetConfig.dense_rasterizer is None:
        raise ValueError(
            "train_dataset.dense_rasterizer must be set when training a "
            "DensePersistencePredictor. See configs/train_confs/train_dataset/"
            "*_dense.yaml for an example."
        )

    print("[createDenseDataloaders] building dense training dataset...", flush=True)
    trainingDataset = distillationDatasetConfig.make_dataset(
        split="train",
        mode="dense",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=trainerConfig.img_size,
    )
    print("[createDenseDataloaders] building per-instance validation dataset...", flush=True)
    validationDataset = distillationDatasetConfig.make_dataset(
        split="val",
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=trainerConfig.img_size,
    )
    print("[createDenseDataloaders] building per-instance val-on-train dataset...", flush=True)
    valOnTrainDataset = distillationDatasetConfig.make_dataset(
        split="train",
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=trainerConfig.img_size,
    )

    needs_per_instance = bool(
        getattr(loss_fn, "requires_per_instance_tensors", lambda: False)()
    )
    train_collate = make_dense_collate(include_per_instance=needs_per_instance)

    train_workers = getattr(trainerConfig, "num_workers", 4)
    val_workers = getattr(trainerConfig, "val_num_workers", None) or train_workers
    val_bs = getattr(trainerConfig, "val_batch_size", None) or trainerConfig.batch_size

    trainLoader = DataLoader(
        trainingDataset,
        batch_size=trainerConfig.batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=train_workers > 0,
        collate_fn=train_collate,
    )
    valLoader = DataLoader(
        validationDataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    # Deterministic sequential pass for val-on-train so the diagnostic
    # is reproducible across epochs and across model variants. The
    # validator's max_batches cap controls compute.
    valOnTrainLoader = DataLoader(
        valOnTrainDataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    return trainLoader, valLoader, valOnTrainLoader


def loadModelAndBackpropStuff(cfg: DictConfig) -> nn.Module:
    model = hydra.utils.instantiate(cfg.model)
    loss_fn = hydra.utils.instantiate(cfg.trainer.loss_function)

    optimizer = hydra.utils.instantiate(
        cfg.trainer.optimizer,
        params=model.parameters(),
        _convert_="all",   # Avoid ListConfig objects in optimizer params, convert to plain list/dict
    )
    lr_scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler, optimizer=optimizer)

    # Move necessary stuff to device
    device = torch.device(cfg.trainer.loop.device if cfg.trainer.loop.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    loss_fn.to(device)
    return model, loss_fn, optimizer, lr_scheduler, device


def createValidator(validationConfig: ValConfig, loss_fn: nn.Module, device: torch.device) -> SurvivalValidationRunner:
    validator = SurvivalValidationRunner(cfg=validationConfig, loss_fn=loss_fn, device=device)
    return validator


# Use hydra config to load in relevant configuration
@hydra.main(version_base=None, config_path="configs/train_confs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Access Hydra's own config
    hydra_cfg = HydraConfig.get()

    # The config filename (e.g. "cross_attn_random_instwise_manlabel_...") is used as the wandb run name
    config_name = hydra_cfg.job.config_name

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Instantiate the training loop config (TrainerConfig) from the nested loop sub-config
    trainer_config = hydra.utils.instantiate(cfg.trainer.loop)

    registryConfig = hydra.utils.instantiate(cfg.registry_info)

    datasetConfig = hydra.utils.instantiate(cfg.train_dataset)

    # Model+loss are built before the dataloaders for the dense branch so we
    # can dispatch on the model type and ask the loss whether it needs
    # per-instance tensors in the dense collate.
    print("Creating model, loss function, optimizer, and lr scheduler...")
    model, loss_fn, optimizer, lr_scheduler, device = loadModelAndBackpropStuff(cfg)

    is_dense = isinstance(model, DensePersistencePredictor)
    print(f"Creating dataloaders... (dense_mode={is_dense})")
    valOnTrainLoader: Optional[torch.utils.data.DataLoader] = None
    if is_dense:
        trainLoader, valLoader, valOnTrainLoader = createDenseDataloaders(
            datasetConfig, registryConfig, trainer_config, loss_fn
        )
    else:
        trainLoader, valLoader = createDataloaders(datasetConfig, registryConfig, trainer_config)

    # Wrap the (model, loss_fn) pair in the appropriate TrainingTask. The
    # task owns the forward + loss call shape; the trainer only sees
    # task.compute_loss(batch).
    task: TrainingTask = (
        DensePersistenceTask(model=model, loss_fn=loss_fn)
        if is_dense
        else PerMaskPersistenceTask(model=model, loss_fn=loss_fn)
    )

    # Determine output directory. When resuming, reuse the original run's directory so
    # checkpoints, logs, and metrics stay together. Checkpoint path: <run_dir>/checkpoints/<file>.pth
    # Fresh runs: <model_output_root_dir>/<train_dataset.name>/<model.model_name>/<timestamp>/
    resumeFromCheckpoint = None
    if "resume_ckpt" in cfg and cfg.resume_ckpt is not None:
        resumeFromCheckpoint = cfg.resume_ckpt
        output_dir = str(os.path.dirname(os.path.dirname(resumeFromCheckpoint)))
        print(f"Resuming from checkpoint: {resumeFromCheckpoint}")
        print(f"Reusing original output directory: {output_dir}")
    else:
        output_dir = os.path.join(
            cfg.model_output_root_dir,
            cfg.train_dataset.name,
            model.model_name,
            cfg.trainer.name,
            date_time,
        )
        print("No checkpoint specified to resume from. Starting fresh training run.")

    os.makedirs(output_dir, exist_ok=True)

    val_config = trainer_config.validationConfig

    # Make sure the loss function and the validation config both have the same value for confidence_from_logits
    if hasattr(loss_fn, "confidence_from_logits") and hasattr(val_config, "confidence_from_logits"):
        if loss_fn.confidence_from_logits != val_config.confidence_from_logits:
            raise ValueError(f"Mismatch in confidence_from_logits between loss function and validation config. Loss function has {loss_fn.confidence_from_logits} but validation config has {val_config.confidence_from_logits}. Please make sure these are the same.")
        # Also verify that the model's output predictor has the same value for confidence_from_logits if it has that attribute
        if hasattr(model, "confidence_from_logits"):
            if model.confidence_from_logits != loss_fn.confidence_from_logits:
                raise ValueError(f"Mismatch in confidence_from_logits between model and loss function. Model has {model.confidence_from_logits} but loss function has {loss_fn.confidence_from_logits}. Please make sure these are the same.")

    # The validator scores via the task's per-mask validation forward
    # (identity for per-mask, pooling facade for dense). It always
    # wants a per-instance loss. For per-mask training that's just
    # ``loss_fn``; for dense training we build a matching
    # ``WeightedMultiLoss`` whose confidence-channel handling mirrors
    # the dense training loss so val numbers stay calibrated.
    if is_dense and isinstance(loss_fn, DenseWeightedMultiLoss):
        validator_loss_fn = WeightedMultiLoss(
            w50=loss_fn.w50,
            wr=loss_fn.wr,
            wc=loss_fn.wc,
            useHuberForLogVals=loss_fn.useHuberForLogVals,
            delta_50=loss_fn.delta_50,
            delta_r=loss_fn.delta_r,
            confidence_loss=loss_fn.confidence_loss,
            confidence_from_logits=loss_fn.confidence_from_logits,
            delta_c=loss_fn.delta_c,
            clamp_confidence_probs=loss_fn.clamp_confidence_probs,
            eps=loss_fn.eps,
        ).to(device)
    else:
        validator_loss_fn = loss_fn
    validator = createValidator(val_config, validator_loss_fn, device)

    # Initialize wandb logging if enabled in the config
    if trainer_config.use_wandb:
        wandb.init(
            entity=trainer_config.wandb_entity,
            project=trainer_config.wandb_project,
            name=f"{cfg.train_dataset.name}__{model.model_name}__{cfg.trainer.name}",
            config={**cfg},
        )

    train(
        task=task,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=trainLoader,
        val_loader=valLoader,
        validator=validator,
        output_dir=output_dir,
        cfg=trainer_config,
        full_cfg_yaml=OmegaConf.to_yaml(cfg, resolve=True),
        resumeFromCheckpoint=resumeFromCheckpoint,
        val_on_train_loader=valOnTrainLoader,
    )


if __name__ == "__main__":
    main()