"""Torch <-> JAX bridge for the Perpetua mixture filter wrapper.

JAX is imported lazily so the rest of the codebase doesn't need it. Calling
any function in this module will import jax (CPU mode is enforced via the
`JAX_PLATFORMS=cpu` env var unless the caller has overridden it).

All conversions go through numpy. JAX device=cpu is the only supported
configuration — no GPU memory pressure on the torch CUDA stack.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np
import torch


def ensure_jax_cpu_env() -> None:
    """Ensure JAX defaults to CPU before its first import.

    Idempotent. Any later JAX import in the same process will see CPU.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


@lru_cache(maxsize=1)
def _jax_modules() -> dict[str, Any]:
    """Lazy-import JAX and upstream wrappers; cache the bundle."""
    ensure_jax_cpu_env()
    import jax
    import jax.numpy as jnp
    from jax._src.tree_util import Partial

    # Upstream — vendored at src/perpetua-code/.
    from src.utils import priors as upstream_priors
    from src.filters import (
        MixturePersistenceFilters as upstream_mixture_filter,
        PersistenceFilter as upstream_pf,
    )

    return {
        "jax": jax,
        "jnp": jnp,
        "Partial": Partial,
        "priors": upstream_priors,
        "mixture_filter": upstream_mixture_filter,
        "single_filter": upstream_pf,
    }


def torch_to_jnp(t: torch.Tensor):
    """Convert a torch tensor to a jax.numpy array via numpy (CPU only)."""
    jnp = _jax_modules()["jnp"]
    return jnp.asarray(t.detach().cpu().numpy())


def jnp_to_torch(arr, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert a jax array to a torch tensor (CPU)."""
    return torch.as_tensor(np.asarray(arr), dtype=dtype)


def list_to_jnp_1d(values, dtype=None):
    jnp = _jax_modules()["jnp"]
    if dtype is None:
        return jnp.asarray(values, dtype=jnp.float32)
    return jnp.asarray(values, dtype=dtype)
