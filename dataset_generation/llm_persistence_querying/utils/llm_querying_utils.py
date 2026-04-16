import torch
import os
from typing import Union, Any


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


def resolve_served_model_name(
    client: Any,
    *,
    model_name: str,
    derived_name: str,
    base_url: str,
) -> str:
    """Query /v1/models and return the name the server actually advertises.

    vLLM serves models under a derived name (e.g. "qwen3-30b__bfloat16");
    TRT-LLM uses the raw HF path (e.g. "Qwen/Qwen3-30B-A3B-Thinking-2507").
    Tries derived_name first, then model_name (HF path), then raises.
    """
    ids = [m.id for m in client.models.list().data]
    if derived_name in ids:
        print(f"Server model resolved: {derived_name!r}")
        return derived_name
    if model_name in ids:
        print(
            f"Derived name {derived_name!r} not found on server; "
            f"falling back to HF model name {model_name!r}"
        )
        return model_name
    raise RuntimeError(
        f"Could not resolve a served model name.\n"
        f"Tried derived name: {derived_name!r}\n"
        f"Tried HF name:      {model_name!r}\n"
        f"Server advertises:  {ids}\n"
        f"base_url={base_url!r}"
    )


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