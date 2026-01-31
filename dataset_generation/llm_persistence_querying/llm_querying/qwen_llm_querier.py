from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenLLMQuerier(AbsLLMQuerier):
    """
    Wrapper around a local Qwen2.5-7B-Instruct model.

    - Loads model + tokenizer from Hugging Face.
    - Uses the Qwen chat template internally.
    - queryLLM() takes a plain text prompt and returns the model's reply text.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str | dict[str, Any] = "auto",
        torch_dtype: torch.dtype | str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        name: Optional[str] = None,
    ):
        """
        Args:
            model_name: Hugging Face model ID.
            device_map: Passed to from_pretrained (e.g. "auto", "cuda", {"": 0}).
            torch_dtype: "auto" or a torch dtype (e.g. torch.bfloat16).
            max_new_tokens: Default max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            repetition_penalty: >1.0 to reduce repetition.
        """

        if name is None:
            raise ValueError("name must be provided for QwenLLMQuerier")

        super().__init__(name=name)

        self.model_name = model_name

        # ---- normalize torch_dtype coming from Hydra ----
        if isinstance(torch_dtype, str):
            if torch_dtype == "auto":
                resolved_dtype = "auto"
            else:
                # support both "bfloat16" and "torch.bfloat16"
                name = torch_dtype.split(".")[-1]
                try:
                    resolved_dtype = getattr(torch, name)
                except AttributeError as e:
                    raise ValueError(
                        f"Unknown torch dtype string {torch_dtype!r}. "
                        f"Expected something like 'bfloat16', 'float16', 'float32', etc."
                    ) from e
        else:
            resolved_dtype = torch_dtype

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Default generation config
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

    def queryLLM(self, prompt: str) -> str:
        """
        Send a single-turn prompt to Qwen and return the decoded response.

        The `prompt` should already include any instructions (e.g.,
        "Respond with ONLY valid JSON...").
        """
        # Wrap your raw prompt into a single user message and apply Qwen's chat template.
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Convert messages → model input text via Qwen chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize and move to the correct device
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                **self.generation_kwargs,
            )

        # Strip the prompt tokens from the output to get *only* the reply
        generated_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response.strip()