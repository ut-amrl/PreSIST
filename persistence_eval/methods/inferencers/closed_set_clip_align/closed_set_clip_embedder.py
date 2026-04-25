from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from persistence_eval.methods.inferencers.closed_set_semantics.closed_set_semantics import ClosedSetSemanticClassList


class CachedCLIPAlignment:
    """Stores pre-computed normalized CLIP text embeddings keyed by class name."""

    def __init__(self, cache_file: str | None = None) -> None:
        self.mapping: dict[str, torch.Tensor] = {}
        if cache_file is not None:
            self.load_cache(cache_file)

    def load_cache(self, cache_file: str) -> None:
        self.mapping = torch.load(cache_file, weights_only=True)

    def save_cache(self, output_file: str) -> None:
        torch.save(self.mapping, output_file)

    def get_embedding(self, semantic_class: str) -> torch.Tensor:
        if semantic_class not in self.mapping:
            raise KeyError(f"Class '{semantic_class}' not in cache.")
        return self.mapping[semantic_class]

    def class_names(self) -> list[str]:
        return sorted(self.mapping.keys())


class CLIPEmbedder:
    """Wraps a HuggingFace CLIP model for text and image embedding."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def compute_clip_embedding(self, semantic_class: str) -> torch.Tensor:
        return self.compute_clip_embeddings([semantic_class])[0]

    def compute_clip_embeddings(self, semantic_classes: list[str]) -> torch.Tensor:
        """Returns (N, D) normalized float32 tensor on CPU."""
        self._load()
        inputs = self._processor(text=semantic_classes, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._model.get_text_features(**inputs)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu().float()

    def compute_clip_embedding_for_image(self, image: Image.Image) -> torch.Tensor:
        """Returns normalized 1D float32 tensor on CPU."""
        self._load()
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embedding = self._model.get_image_features(**inputs)
        embedding = F.normalize(embedding, dim=-1)
        return embedding[0].cpu().float()


class CLIPAligner:
    """Finds the closest semantic class for a masked image region using CLIP."""

    def __init__(
        self,
        clip_embedder: CLIPEmbedder,
        cache: CachedCLIPAlignment,
        class_list: ClosedSetSemanticClassList,
        zero_out_background: bool = False,
    ) -> None:
        self.clip_embedder = clip_embedder
        self.cache = cache
        self.class_list = class_list
        self.zero_out_background = zero_out_background

    def find_closest_class(self, image: torch.Tensor, mask: torch.Tensor) -> int:
        """
        Args:
            image: CHW uint8 tensor
            mask:  HW bool tensor
        Returns:
            Integer class ID of the closest semantic class.
        """
        crop_pil = self._extract_masked_crop(image, mask)
        image_embedding = self.clip_embedder.compute_clip_embedding_for_image(crop_pil)

        class_names = self.cache.class_names()
        text_embeddings = torch.stack([self.cache.get_embedding(c) for c in class_names])  # (N, D)

        similarities = text_embeddings @ image_embedding  # (N,)
        best_name = class_names[int(similarities.argmax().item())]
        return self.class_list.class_to_id[best_name]

    def _extract_masked_crop(self, image: torch.Tensor, mask: torch.Tensor) -> Image.Image:
        """Crops the padded bounding box of the mask and zeros out non-mask pixels."""
        H, W = mask.shape
        rows = mask.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        cols = mask.any(dim=0).nonzero(as_tuple=False).squeeze(1)

        r_min, r_max = int(rows[0].item()), int(rows[-1].item())
        c_min, c_max = int(cols[0].item()), int(cols[-1].item())

        pad_r = max(1, int(0.15 * (r_max - r_min + 1)))
        pad_c = max(1, int(0.15 * (c_max - c_min + 1)))

        r0 = max(0, r_min - pad_r)
        r1 = min(H - 1, r_max + pad_r)
        c0 = max(0, c_min - pad_c)
        c1 = min(W - 1, c_max + pad_c)

        crop = image[:, r0:r1 + 1, c0:c1 + 1].clone()  # (C, h, w)
        if self.zero_out_background:
            crop[:, ~mask[r0:r1 + 1, c0:c1 + 1]] = 0

        pil = Image.fromarray(crop.permute(1, 2, 0).numpy())
        return pil
