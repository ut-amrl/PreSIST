import torch
import os
from typing import Union


def derive_served_model_name(model_name: str, torch_dtype: str, quantization: bool = False) -> str:

    model_basename = os.path.basename(model_name)
    model_normalized = model_basename.lower()

    if (not isinstance(torch_dtype, str)):
        torch_dtype = str(torch_dtype)

    dtype_str = torch_dtype.split(".")[-1].lower() if isinstance(torch_dtype, str) else str(torch_dtype).split(".")[-1].lower()

    name = f"{model_normalized}__{dtype_str}"
    if quantization:
        name = f"{name}__{quantization}"
    return name


def resolve_torch_dtype(torch_dtype: Union[torch.dtype, str]) -> Union[torch.dtype, str]:
    if isinstance(torch_dtype, str):
        if torch_dtype == "auto":
            return "auto"
        name = torch_dtype.split(".")[-1]
        try:
            return getattr(torch, name)
        except AttributeError as e:
            raise ValueError(
                f"Unknown torch dtype string {torch_dtype!r}. "
                f"Expected 'auto' or something like 'bfloat16', 'float16', 'float32', etc."
            ) from e
    return torch_dtype