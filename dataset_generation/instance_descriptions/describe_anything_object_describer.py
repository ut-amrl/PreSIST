from utils.image_utils import tensor_image_to_pil, mask_to_pil
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
import torch
from typing import Optional, Dict, Tuple

from PIL import Image
from io import BytesIO
import base64
import numpy as np
import requests

from dam import DescribeAnythingModel, disable_torch_init  # from the repo


class DescribeAnythingObjectDescriber(AbstractObjectDescriber):
    def __init__(
        self,
        model_path: str = "nvidia/DAM-3B",
        device: Optional[torch.device] = None,
        prompt_file: Optional[str] = None,
        prompt: Optional[str] = None,
        conv_mode: str = "v1",
        prompt_mode: str = "focal_prompt",
        *,
        # --- NEW ---
        use_server: bool = True,
        # server_url: str = "http://localhost:9002",
        server_base_url: str = "http://localhost",
        server_port: int = 9002,
        server_model: str = "describe_anything_model",
        server_api_key: str = "api_key",
        server_timeout_s: float = 120.0,
        server_batch_size: int = 32,
        # -----------
        temperature: float = 0.2,
        top_p: float = 0.5,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        streaming: bool = False,
        name: Optional[str] = None,
    ):
        """
        Thin wrapper around NVidia's DescribeAnythingModel.

        NEW:
            use_server:
                If True, call a DAM server (OpenAI-compatible API). If False, run locally.

            server_base_url:
                Base URL for DAM server, e.g. "http://localhost"

            server_port:
                Port for DAM server, e.g. 8000

            server_model:
                Model name exposed by the server, e.g. "describe_anything_model"

            server_api_key:
                Dummy key used by the OpenAI-compatible client (server typically ignores it)

            server_timeout_s:
                Requests timeout (seconds) for server calls.

            server_batch_size:
                Max number of masks per /batch/chat/completions call. Must not exceed
                the server's DAM_MAX_BATCH_SIZE (default 32). Instances are split using
                SegmentationResults.divideIntoSubsets() for even distribution.
        """
        if name is None:
            raise ValueError("name must be provided for DescribeAnythingObjectDescriber")
        super().__init__(name=name)

        disable_torch_init()  # same as in examples to speed up init

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")

        if (prompt_file is None) and (prompt is None):
            prompt = "<image>\nDescribe the masked region in detail."
        elif prompt_file is not None:
            with open(prompt_file, "r") as f:
                prompt = f.read()

        self.use_server = use_server
        # self.server_url = server_url.rstrip("/")
        self.server_url = f"{server_base_url.rstrip('/')}:{server_port}"
        self.server_model = server_model
        self.server_api_key = server_api_key
        self.server_timeout_s = float(server_timeout_s)
        self.server_batch_size = int(server_batch_size)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_mode = prompt_mode
        self.conv_mode = conv_mode
        self.model_path = model_path
        self.prompt = prompt
        self.initialized = False

        # generation kwargs (used both locally and for server payload)
        self.gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            streaming=streaming,
        )

    def initialize(self):
        if self.use_server:
            # No local model to load
            self.dam = None
            self.initialized = True
            return

        # Mirror the CLI mapping in examples/dam_with_sam.py
        prompt_modes = {"focal_prompt": "full+focal_crop"}
        resolved_prompt_mode = prompt_modes.get(self.prompt_mode, self.prompt_mode)

        self.dam = DescribeAnythingModel(
            model_path=self.model_path,
            conv_mode=self.conv_mode,
            prompt_mode=resolved_prompt_mode,
        ).to(self.device)
        self.dam.eval()
        self.initialized = True

    # ---------- server helpers ----------

    def _prompt_for_server(self) -> str:
        """
        Server expects normal chat text (no '<image>' token).
        """
        text = (self.prompt or "").replace("<image>", "").strip()
        # Their examples prepend a newline; harmless either way.
        if text and not text.startswith("\n"):
            text = "\n" + text
        return text or "\nDescribe the masked region in detail."



    def _rgba_png_data_url_from_numpy(self, rgb_np: np.ndarray, mask_l: Image.Image) -> str:
        """
        Encode (RGB numpy array + PIL mask) to a base64 RGBA PNG data URL.
        Accepts a pre-converted numpy RGB array so the caller can convert once
        and reuse across many masks.
        """
        mask = np.array(mask_l.convert("L"))
        rgba = np.zeros((*rgb_np.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb_np
        rgba[..., 3] = (mask > 0).astype(np.uint8) * 255
        buf = BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _rgba_png_data_url(self, rgb_image: Image.Image, mask_l: Image.Image) -> str:
        """
        Encode (RGB + mask-as-alpha) to a base64 data URL.
        Converts rgb_image to numpy on each call; prefer _rgba_png_data_url_from_numpy
        when the same image is reused across many masks.
        """
        return self._rgba_png_data_url_from_numpy(np.array(rgb_image.convert("RGB")), mask_l)

    def _post_chat_completions(self, payload: dict) -> dict:
        """
        Repo examples use POST {server_url}/chat/completions (no /v1).
        Some OpenAI-compatible servers also accept /v1/chat/completions,
        so we try both.
        """
        headers = {"Content-Type": "application/json"}
        # (Optional) include Authorization header—harmless if ignored
        if self.server_api_key:
            headers["Authorization"] = f"Bearer {self.server_api_key}"

        urls_to_try = [
            f"{self.server_url}/chat/completions",
            f"{self.server_url}/v1/chat/completions",
        ]

        last_err = None
        for url in urls_to_try:
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.server_timeout_s)
                if r.status_code == 200:
                    return r.json()
                last_err = RuntimeError(f"DAM server error {r.status_code} from {url}: {r.text}")
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Failed to call DAM server at {self.server_url}. Last error: {last_err}")

    def _get_description_server(self, pil_image: Image.Image, pil_mask: Image.Image) -> str:
        """
        Non-streaming server call (returns full string).
        Note: If you set streaming=True in constructor, we still return full text here.
        """
        img_url = self._rgba_png_data_url(pil_image, pil_mask)
        prompt_text = self._prompt_for_server()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        payload = {
            "model": self.server_model,
            "messages": messages,
            "max_tokens": int(self.gen_kwargs["max_new_tokens"]),
            "temperature": float(self.gen_kwargs["temperature"]),
            "top_p": float(self.gen_kwargs["top_p"]),
            "num_beams": int(self.gen_kwargs["num_beams"]),
            # matches the raw example payload; safe if server ignores
            "use_cache": True,
            # we intentionally do not request stream=True here
        }

        result = self._post_chat_completions(payload)
        # OpenAI-style response: choices[0].message.content
        return (result["choices"][0]["message"]["content"] or "").strip()

    def _post_batch_chat_completions(self, payload: dict) -> dict:
        """
        POST to /batch/chat/completions on the DAM server.
        The fork exposes this only at the root path (no /v1/ prefix).
        """
        headers = {"Content-Type": "application/json"}
        if self.server_api_key:
            headers["Authorization"] = f"Bearer {self.server_api_key}"

        url = f"{self.server_url}/batch/chat/completions"
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.server_timeout_s)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"DAM batch server error {r.status_code} from {url}: {r.text}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to call DAM batch endpoint at {url}. Error: {e}") from e

    def _describe_objects_batch_server(
        self,
        pil_image: Image.Image,
        instances,
    ) -> Dict[int, str]:
        """
        Describe all instances in batch using /batch/chat/completions.
        Instances are split into evenly-distributed chunks via divideIntoSubsets().
        """
        prompt_text = self._prompt_for_server()
        descriptions: Dict[int, str] = {}

        # Convert once; reused for every mask instead of converting inside the loop.
        rgb_np = np.array(pil_image.convert("RGB"))

        chunks = instances.divideIntoSubsets(self.server_batch_size)
        total = len(instances)
        processed = 0

        for chunk in chunks:
            chunk_size = len(chunk)
            print(f"Describing instances {processed + 1}–{processed + chunk_size} of {total} (batch)...")

            requests_list = []
            obj_ids = []
            for i, mask in enumerate(chunk.masks):
                obj_id = int(chunk.ids[i].item())
                obj_ids.append(obj_id)
                data_uri = self._rgba_png_data_url_from_numpy(rgb_np, mask_to_pil(mask))

                requests_list.append({
                    "model": self.server_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    "max_tokens": int(self.gen_kwargs["max_new_tokens"]),
                    "temperature": float(self.gen_kwargs["temperature"]),
                    "top_p": float(self.gen_kwargs["top_p"]),
                    "num_beams": int(self.gen_kwargs["num_beams"]),
                    "use_cache": True,
                })

            payload = {
                "model": self.server_model,
                "requests": requests_list,
            }

            result = self._post_batch_chat_completions(payload)
            for idx, item in enumerate(result.get("results", [])):
                obj_id = obj_ids[idx]
                if "error" in item:
                    print(f"WARNING: DAM batch error for obj_id {obj_id}: {item['error']}")
                    descriptions[obj_id] = ""
                else:
                    descriptions[obj_id] = (item["choices"][0]["message"]["content"] or "").strip()

            processed += chunk_size

        return descriptions

    # ---------- main API ----------

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
    ) -> Tuple[Dict[int, str], DebugPayload]:
        """
        Describe objects in the image based on the provided instances.
        """
        if not self.initialized:
            self.initialize()

        pil_image = tensor_image_to_pil(image)


        if self.use_server:
            return self._describe_objects_batch_server(pil_image, instances), DebugPayload.empty()

        descriptions: Dict[int, str] = {}
        for i, mask in enumerate(instances.masks):
            obj_id = int(instances.ids[i].item())
            pil_mask = mask_to_pil(mask)

            with torch.inference_mode():
                out = self.dam.get_description(
                    pil_image,
                    pil_mask,
                    self.prompt,
                    **self.gen_kwargs,
                )
                if isinstance(out, str):
                    description = out.strip()
                else:
                    description = "".join(out).strip()

            descriptions[obj_id] = description

        return descriptions, DebugPayload.empty()



# from utils.image_utils import tensor_image_to_pil, mask_to_pil
# from instance_segmentation.abs_instance_segmenter import SegmentationResults
# from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
# import torch
# import numpy as np
# from typing import Optional, Dict


# from PIL import Image
# from dam import DescribeAnythingModel, disable_torch_init  # from the repo


# class DescribeAnythingObjectDescriber(AbstractObjectDescriber):
#     def __init__(
#         self,
#         model_path: str = "nvidia/DAM-3B",
#         device: Optional[torch.device] = None,
#         prompt_file: Optional[str] = None,
#         prompt: Optional[str] = None,
#         conv_mode: str = "v1",
#         prompt_mode: str = "focal_prompt",
#         *,
#         temperature: float = 0.2,
#         top_p: float = 0.5,
#         num_beams: int = 1,
#         max_new_tokens: int = 512,
#         streaming: bool = False,
#         name: Optional[str] = None,
#     ):
#         """
#         Thin wrapper around NVidia's DescribeAnythingModel.

#         Args:
#             model_path:
#                 Path or Hugging Face ID for the DAM checkpoint.
#                 In their example they use "nvidia/DAM-3B". :contentReference[oaicite:0]{index=0}

#             device:
#                 Torch device to run DAM on. If None, picks CUDA if available, else CPU.

#             prompt:
#                 Text prompt given to the model, e.g.
#                 "<image>\\nDescribe the masked region in detail."

#             conv_mode:
#                 Conversation template. The examples use "v1". :contentReference[oaicite:1]{index=1}

#             prompt_mode:
#                 High-level prompt mode string. In the example script they accept
#                 "focal_prompt" and internally map it to "full+focal_crop". :contentReference[oaicite:2]{index=2}
#                 We preserve that behavior below.

#             temperature, top_p, num_beams, max_new_tokens, streaming:
#                 Generation parameters forwarded to `get_description`.
#         """

#         if name is None:
#             raise ValueError("name must be provided for DescribeAnythingObjectDescriber")
#         super().__init__(name=name)
#         disable_torch_init()  # same as in examples to speed up init

#         if (prompt is not None) and (prompt_file is not None):
#             raise ValueError("Only one of prompt or prompt_file should be provided.")

#         if (prompt_file is None) and (prompt is None):
#             prompt = "<image>\nDescribe the masked region in detail."
#         elif prompt_file is not None:
#             with open(prompt_file, 'r') as f:
#                 prompt = f.read()


#         self.device = device or torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )
#         self.prompt_mode = prompt_mode
#         self.conv_mode = conv_mode
#         self.model_path = model_path
#         self.prompt = prompt
#         self.initialized = False
#         self.gen_kwargs = dict(
#             temperature=temperature,
#             top_p=top_p,
#             num_beams=num_beams,
#             max_new_tokens=max_new_tokens,
#             streaming=streaming,
#         )
#         # self.initialize()

#     def initialize(self):
#         # Mirror the CLI mapping in examples/dam_with_sam.py
#         prompt_modes = {
#             "focal_prompt": "full+focal_crop",
#         }

#         resolved_prompt_mode = prompt_modes.get(self.prompt_mode, self.prompt_mode)
#         # Create the DAM model exactly like the example script
#         self.dam = DescribeAnythingModel(
#             model_path=self.model_path,
#             conv_mode=self.conv_mode,
#             prompt_mode=resolved_prompt_mode,
#         ).to(self.device)
#         self.dam.eval()
#         self.initialized = True


#     # ---------- main API ----------

#     def describeObjects(
#         self,
#         image: torch.Tensor,
#         instances: SegmentationResults,
#     ) -> Dict[int, str]:
#         """
#         Describe objects in the image based on the provided instances.

#         Args:
#             image:
#                 Input image tensor of shape (C, H, W).
#             instances:
#                 SegmentationResults with:
#                     - instances.masks: iterable of per-instance masks (H, W) or (1, H, W)
#                     - instances.ids: tensor/list of instance IDs
#                     - instances.boxes: tensor of (N, 4) in XYXY format (currently unused)

#         Returns:
#             dict[obj_id -> description string]
#         """
#         if not self.initialized:
#             self.initialize()
#         pil_image = tensor_image_to_pil(image)
#         descriptions: Dict[int, str] = {}

#         for i, mask in enumerate(instances.masks):
#             obj_id = int(instances.ids[i].item())
#             # obj_bb = instances.boxes[i]  # available if you later want to use it

#             pil_mask = mask_to_pil(mask)

#             with torch.inference_mode():

#                 # Call Describe Anything Model
#                 out = self.dam.get_description(
#                     pil_image,
#                     pil_mask,
#                     self.prompt,
#                     **self.gen_kwargs,
#                 )

#             # If streaming=False, `out` is a full string;
#             # If streaming=True, it's an iterator of tokens, like in the example script.
#             if isinstance(out, str):
#                 description = out.strip()
#             else:
#                 description = "".join(out).strip()

#             descriptions[obj_id] = description

#         return descriptions
