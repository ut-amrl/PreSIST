from __future__ import annotations

import os
from time import time
from typing import Any, Optional, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from openai import OpenAI

from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
import os

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import derive_served_model_name, resolve_served_model_name, resolve_torch_dtype

# def _derive_served_model_name(model_name, torch_dtype, quantization=False):

#     model_basename = os.path.basename(model_name)
#     model_normalized = model_basename.lower()

#     if isinstance(torch_dtype, str):
#         dtype_str = torch_dtype.split(".")[-1]
#     else:
#         dtype_str = str(torch_dtype).split(".")[-1]

#     dtype_str = dtype_str.lower()

#     name = f"{model_normalized}__{dtype_str}"
#     if quantization:
#         name = f"{name}__{quantization}"
#     return name


class QwenLLMQuerier(AbsLLMQuerier):
    """
    Qwen querier with two modes:

    1) In-memory (Transformers): loads model + tokenizer in-process and uses generate()
    2) vLLM server (OpenAI-compatible): does NOT load model locally; sends requests to vLLM

    Switch via `use_vllm_server`.

    Notes:
    - In vLLM mode, the server applies the chat template. We send chat `messages`.
    - In in-memory mode, we apply Qwen's chat template locally.
    """

    def __init__(
        self,
        *,
        # --- mode ---
        use_vllm_server: bool = False,

        # --- model identity / naming ---
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",   # used for local mode; also used for tokenizer in local mode
        name: Optional[str] = None,

        # --- local (Transformers) runtime ---
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        attn_implementation: str = "flash_attention_2",  # "flash_attention_2" or "sdpa"

        # --- vLLM server runtime ---
        # vllm_base_url: str = "http://localhost:9003/v1",
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9001,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_verify_on_init: bool = True,
        vllm_request_timeout_s: float = 120.0,

        # --- generation / sampling defaults (per request in vLLM mode; local kwargs in local mode) ---
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,

        # --- optional: extra request params for vLLM ---
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
    ):
        if name is None:
            raise ValueError("name must be provided for QwenLLMQuerier")

        super().__init__(name=name)

        self.use_vllm_server = use_vllm_server
        self.model_name = model_name

        # Store sampling defaults (used in both modes)
        self._default_sampling: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,         # we'll map to max_tokens for vLLM
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        if self.use_vllm_server:

            self.vllm_served_model_name = derive_served_model_name(self.model_name, torch_dtype)

        # -------------------------
        # vLLM server mode
        # -------------------------
        self.vllm_client = None
        # self.vllm_served_model_name: Optional[str] = None
        # Set the vllm_served_model_name to match the bash script's configured name
        # Bash version:
        # MODEL_BASENAME="$(basename "$MODEL")"
        # MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"

        # SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${DTYPE}"
        # if [ -n "$QUANTIZATION" ]; then
        # SERVED_MODEL_NAME="${SERVED_MODEL_NAME}__${QUANTIZATION}"
        # fi


        self.vllm_extra_request_params = vllm_extra_request_params or {}

        # -------------------------
        # local mode
        # -------------------------
        self.model = None
        self.tokenizer = None
        self.generation_kwargs: Dict[str, Any] = {}

        self.totalQueryTime = 0.0
        self.numQueries = 0

        print("Using QwenLLMQuerier with use_vllm_server =", self.use_vllm_server)

        self.initialized = False
        self.vllm_url_root = vllm_url_root
        self.vllm_port = vllm_port
        self.vllm_url_suffix = vllm_url_suffix
        self.vllm_api_key = vllm_api_key
        self.vllm_timeout_request_s = vllm_request_timeout_s
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty



    # -------------------------
    # Helpers
    # -------------------------



    def initialize(self):

        if (self.initialized):
            return

        if self.use_vllm_server:
            vllm_base_url = f"{self.vllm_url_root}:{self.vllm_port}{self.vllm_url_suffix}"
            self.vllm_base_url = vllm_base_url.rstrip("/")
            from openai import OpenAI  # Import here to avoid dependency if not using vLLM mode
            self.vllm_client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key,
                timeout=self.vllm_timeout_request_s,
            )

            self._resolve_served_model_name()

        else:
            # Initialize local Transformers model+tokenizer
            resolved_dtype = resolve_torch_dtype(self.torch_dtype)

            # Try requested attention impl; fall back to sdpa if needed
            self.model = self._load_local_model_with_fallback(
                model_name=self.model_name,
                device_map=self.device_map,
                torch_dtype=resolved_dtype,
                attn_implementation=self.attn_implementation,
            )
            self.model.eval()
            self.model.config.use_cache = True

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            self.generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True if self.temperature > 0 else False,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
            }


    def _load_local_model_with_fallback(
        self,
        *,
        model_name: str,
        device_map: Union[str, Dict[str, Any]],
        torch_dtype: Union[torch.dtype, str],
        attn_implementation: str,
    ):
        # If user asked for flash_attention_2, try it first; otherwise use what they requested directly.
        tried: List[str] = []
        order = [attn_implementation]
        if attn_implementation != "sdpa":
            order.append("sdpa")

        last_exc: Optional[Exception] = None
        for impl in order:
            tried.append(impl)
            try:
                m = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation=impl,
                )
                if impl == "flash_attention_2":
                    print("Using FlashAttention 2")
                else:
                    print(f"Using attention implementation: {impl}")
                return m
            except Exception as e:
                last_exc = e
                print(f"Failed to load with attn_implementation={impl} ({e}); trying fallback...")

        raise RuntimeError(f"Could not load model {model_name!r}. Tried attn implementations: {tried}") from last_exc

    def _resolve_served_model_name(self) -> None:
        assert self.vllm_client is not None
        self.vllm_served_model_name = resolve_served_model_name(
            self.vllm_client,
            model_name=self.model_name,
            derived_name=self.vllm_served_model_name,
            base_url=self.vllm_base_url,
        )

    def _verify_vllm_server_model(self) -> None:
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None
        models = self.vllm_client.models.list()
        ids = [m.id for m in models.data]
        if self.vllm_served_model_name not in ids:
            raise RuntimeError(
                "vLLM server model mismatch.\n"
                f"Expected served_model_name={self.vllm_served_model_name!r}\n"
                f"Server advertises: {ids}\n"
                f"base_url={self.vllm_base_url!r}"
            )

    # -------------------------
    # Public API
    # -------------------------

    def queryLLM(
        self,
        prompt: str,
        *,
        # Optional per-call overrides
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Single-turn query.

        - Local mode: applies Qwen chat template locally and calls transformers.generate().
        - vLLM mode: calls /v1/chat/completions with messages=[...] and returns message.content.

        Args:
            prompt: user prompt (include any instructions yourself)
            max_new_tokens, temperature, top_p, repetition_penalty: optional per-call overrides
            stop: optional stop strings (vLLM mode supported; local mode not wired by default)
            seed: optional deterministic sampling seed (vLLM supports it)
        """
        # Start a timer

        self.initialize()  # Ensure initialization before first query

        start_time = time()
        if self.use_vllm_server:
            vllmOut = self._query_vllm(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # repetition_penalty=repetition_penalty,
                stop=stop,
                seed=seed,
            )

        else:
            vllmOut = self._query_local(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        end_time = time()
        query_time = end_time - start_time
        self.totalQueryTime += query_time
        self.numQueries += 1
        average_time = self.totalQueryTime / self.numQueries
        print(f"Query time: {query_time:.2f}s (average over {self.numQueries} queries: {average_time:.2f}s)")
        return vllmOut

    # -------------------------
    # Mode implementations
    # -------------------------

    def _query_vllm(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        # repetition_penalty: Optional[float],
        stop: Optional[List[str]],
        seed: Optional[int],
    ) -> str:
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None

        # vLLM/OpenAI uses max_tokens
        params: Dict[str, Any] = {
            "max_tokens": max_new_tokens if max_new_tokens is not None else self._default_sampling["max_new_tokens"],
            "temperature": temperature if temperature is not None else self._default_sampling["temperature"],
            "top_p": top_p if top_p is not None else self._default_sampling["top_p"],
            # "repetition_penalty": (
            #     repetition_penalty if repetition_penalty is not None else self._default_sampling["repetition_penalty"]
            # ),
        }
        if stop is not None:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = seed

        resp = self.vllm_client.chat.completions.create(
            model=self.vllm_served_model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
            **self.vllm_extra_request_params,
        )
        return (resp.choices[0].message.content or "").strip()

    def _query_local(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
    ) -> str:
        assert self.model is not None
        assert self.tokenizer is not None

        # Apply per-call overrides to generation kwargs
        gen = dict(self.generation_kwargs)
        if max_new_tokens is not None:
            gen["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen["temperature"] = temperature
            gen["do_sample"] = True if temperature > 0 else False
        if top_p is not None:
            gen["top_p"] = top_p
        if repetition_penalty is not None:
            gen["repetition_penalty"] = repetition_penalty

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen)

        generated_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


# from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
# from typing import Any, Optional

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class QwenLLMQuerier(AbsLLMQuerier):
#     """
#     Wrapper around a local Qwen2.5-7B-Instruct model.

#     - Loads model + tokenizer from Hugging Face.
#     - Uses the Qwen chat template internally.
#     - queryLLM() takes a plain text prompt and returns the model's reply text.
#     """

#     def __init__(
#         self,
#         model_name: str = "Qwen/Qwen2.5-7B-Instruct",
#         device_map: str | dict[str, Any] = "auto",
#         torch_dtype: torch.dtype | str = "auto",
#         max_new_tokens: int = 2048,
#         temperature: float = 0.2,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.05,
#         name: Optional[str] = None,
#     ):
#         """
#         Args:
#             model_name: Hugging Face model ID.
#             device_map: Passed to from_pretrained (e.g. "auto", "cuda", {"": 0}).
#             torch_dtype: "auto" or a torch dtype (e.g. torch.bfloat16).
#             max_new_tokens: Default max tokens to generate.
#             temperature: Sampling temperature.
#             top_p: Nucleus sampling parameter.
#             repetition_penalty: >1.0 to reduce repetition.
#         """

#         if name is None:
#             raise ValueError("name must be provided for QwenLLMQuerier")

#         super().__init__(name=name)

#         self.model_name = model_name

#         # ---- normalize torch_dtype coming from Hydra ----
#         if isinstance(torch_dtype, str):
#             if torch_dtype == "auto":
#                 resolved_dtype = "auto"
#             else:
#                 # support both "bfloat16" and "torch.bfloat16"
#                 name = torch_dtype.split(".")[-1]
#                 try:
#                     resolved_dtype = getattr(torch, name)
#                 except AttributeError as e:
#                     raise ValueError(
#                         f"Unknown torch dtype string {torch_dtype!r}. "
#                         f"Expected something like 'bfloat16', 'float16', 'float32', etc."
#                     ) from e
#         else:
#             resolved_dtype = torch_dtype

#         # Load model
#         # self.model = AutoModelForCausalLM.from_pretrained(
#         #     model_name,
#         #     torch_dtype=resolved_dtype,
#         #     device_map=device_map,
#         #     trust_remote_code=True,
#         #     attn_implementation="sdpa" # Consider removing
#         # )
#         # Load model (try flash-attn, fallback to sdpa)
#         try:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map=device_map,
#                 trust_remote_code=True,
#                 attn_implementation="flash_attention_2",
#             )
#             print("Using FlashAttention 2")
#         except Exception as e:
#             print(f"FlashAttention 2 unavailable ({e}); falling back to SDPA")
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map=device_map,
#                 trust_remote_code=True,
#                 attn_implementation="sdpa",
#             )

#         self.model.eval()
#         self.model.config.use_cache = True



#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#         # Default generation config
#         self.generation_kwargs = {
#             "max_new_tokens": max_new_tokens,
#             "do_sample": True if temperature > 0 else False,
#             "temperature": temperature,
#             "top_p": top_p,
#             "repetition_penalty": repetition_penalty,
#         }

#         # self.generation_kwargs = {
#         #     "max_new_tokens": max_new_tokens,
#         #     "do_sample": True if temperature > 0 else False,
#         #     "temperature": temperature,
#         #     "top_p": top_p,
#         #     "repetition_penalty": repetition_penalty,
#         #     "eos_token_id": self.tokenizer.eos_token_id,
#         #     "pad_token_id": self.tokenizer.eos_token_id,
#         # }


#     def queryLLM(self, prompt: str) -> str:
#         """
#         Send a single-turn prompt to Qwen and return the decoded response.

#         The `prompt` should already include any instructions (e.g.,
#         "Respond with ONLY valid JSON...").
#         """
#         # Wrap your raw prompt into a single user message and apply Qwen's chat template.
#         messages = [
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ]

#         # Convert messages → model input text via Qwen chat template
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         # Tokenize and move to the correct device
#         model_inputs = self.tokenizer(
#             [text],
#             return_tensors="pt",
#         ).to(self.model.device)

#         print("Sending prompt to Qwen model...")
#         # print(self.model.hf_device_map)

#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **model_inputs,
#                 **self.generation_kwargs,
#             )

#         print("Raw output generated.")

#         # Strip the prompt tokens from the output to get *only* the reply
#         generated_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]

#         response = self.tokenizer.batch_decode(
#             generated_ids,
#             skip_special_tokens=True,
#         )[0]

#         return response.strip()