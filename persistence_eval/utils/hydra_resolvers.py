"""Custom OmegaConf resolvers used by the perpetua train/eval entry-points.

Registering a resolver makes it available inside Hydra config interpolations
under `${<name>:<arg>}`. We use them to embed sweep coordinates (training
fraction, seed) into directory and method names without dots, slashes, or
other characters that confuse downstream tools.

Usage in YAML:
    paths:
      model_dir: ${paths.model_root}/${experiment.name}/frac${dot_to_p:${experiment.subsample_fraction}}_seed${experiment.subsample_seed}

The entry-point scripts (`train_persistence_models.py`, `eval_persistence.py`)
import this module before `@hydra.main` runs so resolvers are registered
in time for config compose.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def _dot_to_p(value) -> str:
    """Replace '.' with 'p' in the string form of `value`.

    Used to make `subsample_fraction=0.1` → `0p1` so it can land in a
    path/method-name segment without breaking shell globs, file extensions,
    or anything that treats '.' as a separator.
    """
    return str(value).replace(".", "p")


def register_all() -> None:
    """Idempotently register every resolver this module defines."""
    OmegaConf.register_new_resolver("dot_to_p", _dot_to_p, replace=True)


# Register on import so a single `import persistence_eval.utils.hydra_resolvers`
# at the top of an entry-point script is enough.
register_all()
