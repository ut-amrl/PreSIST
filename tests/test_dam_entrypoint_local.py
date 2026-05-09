import os

import numpy as np
import pytest
import torch
from PIL import Image

from dam import DescribeAnythingModel, disable_torch_init


def test_dam_entrypoint_local_inference_smoke():
    """
    Smoke test that mirrors the external repo's use_server=False path:
    - disable_torch_init()
    - construct DescribeAnythingModel with mapped prompt_mode
    - call get_description(image, mask, prompt, **gen_kwargs)
    - verify we get a reasonably long, non-empty string back.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DAM inference test")

    model_path = os.environ.get("DAM_MODEL_PATH", "nvidia/DAM-3B")

    disable_torch_init()

    # External code maps "focal_prompt" -> "full+focal_crop"
    prompt_mode = "focal_prompt"
    if prompt_mode == "focal_prompt":
        resolved_prompt_mode = "full+focal_crop"
    else:
        resolved_prompt_mode = prompt_mode

    dam = DescribeAnythingModel(
        model_path=model_path,
        conv_mode="v1",
        prompt_mode=resolved_prompt_mode,
    )
    dam.eval()

    # Small random RGB image and full mask.
    h, w = 336, 336
    img_arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    image_pil = Image.fromarray(img_arr, mode="RGB")
    mask_arr = np.ones((h, w), dtype=np.uint8) * 255
    mask_pil = Image.fromarray(mask_arr, mode="L")

    prompt = "<image>\nDescribe the masked region in detail."

    gen_kwargs = dict(
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=128,
        streaming=False,
    )

    out = dam.get_description(
        image_pil,
        mask_pil,
        prompt,
        **gen_kwargs,
    )

    if isinstance(out, str):
        text = out.strip()
    else:
        text = "".join(out).strip()

    # Rough sanity checks on the output string.
    assert isinstance(text, str)
    assert len(text) > 10
    # Guard against pathological outputs.
    assert len(text) < 4096
    # The model should not echo the raw <image> token back verbatim.
    assert "<image>" not in text

