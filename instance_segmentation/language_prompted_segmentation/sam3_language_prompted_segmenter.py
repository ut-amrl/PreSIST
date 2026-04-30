
from instance_segmentation.language_prompted_segmentation.abs_language_prompted_segmenter import AbsLanguagePromptedSegmenter, LanguagePromptedSegmentationResults


from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from utils.threading_utils import ForwardSerializeLock


def _mask_to_box_xyxy(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32)
    x0 = xs.min().item()
    y0 = ys.min().item()
    x1 = xs.max().item() + 1
    y1 = ys.max().item() + 1
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


class SAM3LanguagePromptedSegmenter(AbsLanguagePromptedSegmenter):
    """
    Raw SAM3 image model used via Sam3Processor.set_text_prompt for each prompt.

    Returns 0..K masks per prompt (flattened).
    """

    def __init__(
        self,
        *,
        name: str,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,

        score_thresh: float = 0.5, # optional -- default taken from typical SAM3 confidence threshold; set <=0 to disable
    ):
        super().__init__(name=name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.score_thresh = float(score_thresh)

        try:
            if checkpoint_path is not None:
                self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            else:
                self.model = build_sam3_image_model()
        except TypeError:
            self.model = build_sam3_image_model()

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model, confidence_threshold=self.score_thresh)

        # Defense-in-depth: PyTorch model + Sam3Processor with stateful
        # set_image / set_text_prompt sequence. Single-threaded today; the
        # lock keeps it correct under future cross-scene threading.
        self._forward_lock = ForwardSerializeLock(
            label=f"{type(self).__name__}.segment_from_prompts",
            advice="SAM3 segmentation cannot be parallelized across host "
                   "threads on one model; run from a single-threaded driver",
        )


    @torch.no_grad()
    def segment_from_prompts(self, image: torch.Tensor, prompts: List[str]) -> LanguagePromptedSegmentationResults:
        # Input validation, empty-result short-circuit, and PIL conversion run
        # outside the lock — they're pure CPU/Python and don't touch the
        # model. The lock is acquired only around the prompt loop where the
        # SAM3 processor is actually invoked.
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

        C, H, W = image.shape
        if len(prompts) == 0:
            return LanguagePromptedSegmentationResults(
                prompt_ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        # Convert to PIL once.
        img_np = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
        pil = Image.fromarray(img_np, mode="RGB")

        with self._forward_lock.acquire():
            return self._segment_from_prompts_locked(pil, prompts, H, W)

    def _segment_from_prompts_locked(
        self,
        pil: Image.Image,
        prompts: List[str],
        H: int,
        W: int,
    ) -> LanguagePromptedSegmentationResults:
        all_prompt_ids: List[int] = []
        all_masks: List[torch.Tensor] = []
        all_boxes: List[torch.Tensor] = []
        all_scores: List[float] = []

        for pid, prompt in enumerate(prompts):
            state = self.processor.set_image(pil)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)

            masks = out.get("masks", None)
            boxes = out.get("boxes", None)
            scores = out.get("scores", None)

            if masks is None or len(masks) == 0:
                continue

            # Normalize masks
            if isinstance(masks, torch.Tensor):
                masks_t = masks.detach().to("cpu").to(torch.bool)
            else:
                masks_t = torch.as_tensor(masks, device="cpu").to(torch.bool)

            K = masks_t.shape[0]

            # Normalize scores (optional)
            if isinstance(scores, torch.Tensor):
                scores_list = scores.detach().to("cpu").to(torch.float32).tolist()
            elif scores is not None:
                scores_list = torch.as_tensor(scores, device="cpu", dtype=torch.float32).tolist()
            else:
                scores_list = None

            # Normalize boxes (optional)
            if isinstance(boxes, torch.Tensor):
                boxes_t = boxes.detach().to("cpu").to(torch.float32)
            elif boxes is not None:
                boxes_t = torch.as_tensor(boxes, device="cpu", dtype=torch.float32)
            else:
                boxes_t = None

            for i in range(K):
                m = masks_t[i]
                if m.shape != (H, W):
                    m = m.reshape(H, W)




                s = float(scores_list[i]) if scores_list is not None else 0.0
                if self.score_thresh >= 0 and scores_list is not None and s < self.score_thresh:
                    continue

                if boxes_t is not None and boxes_t.ndim == 2 and boxes_t.shape[1] == 4:
                    b = boxes_t[i]
                else:
                    b = _mask_to_box_xyxy(m)

                all_prompt_ids.append(pid)
                all_masks.append(m)
                all_boxes.append(b)
                all_scores.append(s)

        if len(all_masks) == 0:
            return LanguagePromptedSegmentationResults(
                prompt_ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        masks_out = torch.stack(all_masks, dim=0)
        boxes_out = torch.stack(all_boxes, dim=0).to(torch.float32)
        scores_out = torch.tensor(all_scores, dtype=torch.float32)
        prompt_ids_out = torch.tensor(all_prompt_ids, dtype=torch.int64)

        return LanguagePromptedSegmentationResults(
            prompt_ids=prompt_ids_out,
            masks=masks_out,
            boxes=boxes_out,
            scores=scores_out,
        )
