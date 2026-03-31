"""
semantic_class_classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE LOCATION:
    dataset_generation/semantic_class_classifier.py

WHAT IT DOES:
    Given a list of semantic class labels and a list of image+mask pairs,
    finds the closest semantic class for each masked object using CLIP.

    This is the class-based baseline described in the evaluation plan:
    "Class-based, LLM — assign classes via CLIP alignment with COCO set"

INSTALL (add to Dockerfile after the wandb line):
    RUN pip install ftfy regex tqdm && \
        pip install git+https://github.com/openai/CLIP.git
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# COCO 80 CLASS LIST + FALLBACKS
# This is the "semantic class list" that CLIP compares each mask against.
# Add domain-specific classes here if needed (e.g. "bicycle rack", "trash bin")
# ─────────────────────────────────────────────────────────────────────────────
COCO_CLASS_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
    "object", "background",
]


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC BUCKET MAP
# Maps each class to a persistence tier.
# Used by the "dynamic buckets based" baselines in the evaluation plan.
# ─────────────────────────────────────────────────────────────────────────────
DYNAMIC_BUCKET_MAP = {
    "dynamic": [
        "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "bicycle", "motorcycle", "car", "bus",
        "train", "truck", "airplane", "boat",
    ],
    "semi_static": [
        "backpack", "handbag", "suitcase", "sports ball", "frisbee", "kite",
        "skateboard", "surfboard", "tennis racket", "skis", "snowboard",
        "baseball bat", "baseball glove", "umbrella", "tie", "cell phone",
        "laptop", "remote", "keyboard", "mouse", "bottle", "cup", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "wine glass", "fork", "knife",
        "spoon", "potted plant", "vase", "clock", "book", "teddy bear",
        "scissors", "hair drier", "toothbrush",
    ],
    "static": [
        "bench", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "chair", "couch", "bed", "dining table", "toilet",
        "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
        "object", "background",
    ],
}

# Reverse lookup: "chair" → "static", "person" → "dynamic"
CLASS_TO_BUCKET = {
    cls: bucket
    for bucket, classes in DYNAMIC_BUCKET_MAP.items()
    for cls in classes
}


class SemanticClassClassifier:
    """
    Assigns the closest semantic class label to each SAM segmentation mask
    using CLIP zero-shot image-text matching.

    How CLIP works:
        - CLIP has two encoders: one for images, one for text
        - Both produce vectors in the same shared space
        - You compare an image vector against text vectors for each class
        - Whichever class text vector is closest = predicted class
        - No training needed — this is zero-shot

    Usage:
        clf = SemanticClassClassifier(device="cuda")
        results = clf.classify_all_instances(image_np, masks)
        for r in results:
            print(r["predicted_class"], r["dynamic_bucket"])
    """

    def __init__(
        self,
        class_labels: list = None,
        model_name: str = "ViT-L/14",
        device: str = "cuda",
        confidence_threshold: float = 0.15,
        top_k: int = 3,
    ):
        """
        Args:
            class_labels:
                The semantic class list to match against.
                Defaults to COCO 80 + "object" + "background".
                This is exactly the "COCO set + Object + Background"
                described in the evaluation plan.

            model_name:
                Which CLIP model to use.
                "ViT-L/14" is the most accurate standard option.
                "ViT-B/32" is faster but less accurate.

            device:
                "cuda" or "cpu". Use "cuda" — same as all your
                other components (SAM2, DAM, InstructBLIP, Qwen).

            confidence_threshold:
                If CLIP's best match confidence is below this value,
                the instance is labeled "object" (fallback).
                Handles things outside COCO — common in surveillance scenes.

            top_k:
                How many top predictions to store. Doesn't affect
                predicted_class, just useful for debugging.
        """
        try:
            import clip
            self._clip = clip
        except ImportError:
            raise ImportError(
                "OpenAI CLIP not installed.\n"
                "Add to Dockerfile:\n"
                "  RUN pip install ftfy regex tqdm && \\\n"
                "      pip install git+https://github.com/openai/CLIP.git"
            )

        self.class_labels = class_labels or COCO_CLASS_LABELS
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = device

        # Load CLIP model and image preprocessor.
        # model      → the neural network (image + text encoders)
        # preprocess → resizes images to 224x224 and normalizes pixel values
        self.model, self.preprocess = self._clip.load(model_name, device=self.device)
        self.model.eval()

        # Encode all class label strings into text vectors RIGHT NOW at init.
        # This is the expensive part — we do it once and store the result.
        # Never re-encode per image.
        self.text_features = self._encode_class_labels()

        logger.info(
            f"SemanticClassClassifier ready — "
            f"{len(self.class_labels)} classes, "
            f"model={model_name}, device={device}"
        )

    def _encode_class_labels(self) -> torch.Tensor:
        """
        Converts every class label string into a CLIP text embedding vector.
        Called once at __init__, result stored as self.text_features.

        Uses "a photo of a {label}" prompt — standard CLIP zero-shot template
        from the original paper, consistently outperforms bare class names.

        Returns:
            Tensor shape (num_classes, 768) for ViT-L/14, unit-normalized.
        """
        prompts = [f"a photo of a {label}" for label in self.class_labels]

        # tokenize: converts strings → integer token IDs
        text_tokens = self._clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize to unit length so dot product = cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _get_masked_crop(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
    ) -> Image.Image:
        """
        Crops the bounding box of the mask out of the full image.
        Fills pixels outside the mask with neutral gray (128).

        Why gray: black/white resemble real object colors and confuse CLIP.
        Gray (128) is neutral — no COCO class looks like a pure gray blob.

        Args:
            image_np : (H, W, 3) uint8 RGB numpy array — the full image
            mask_np  : (H, W) binary numpy array — 1=object, 0=background

        Returns:
            PIL Image of the masked object on gray background.
        """
        rows, cols = np.where(mask_np > 0)

        if len(rows) == 0:
            logger.warning("Empty mask — returning gray placeholder.")
            return Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))

        # Bounding box of the mask region
        r1, r2 = int(rows.min()), int(rows.max())
        c1, c2 = int(cols.min()), int(cols.max())

        # Crop both the image and mask to the bounding box
        crop_img  = image_np[r1:r2 + 1, c1:c2 + 1].copy()  # (H', W', 3)
        crop_mask = mask_np[r1:r2 + 1, c1:c2 + 1]           # (H', W')

        # Gray background same size as the crop
        gray_bg = np.full_like(crop_img, 128)

        # crop_mask[:, :, None] expands (H', W') → (H', W', 1)
        # so it broadcasts across all 3 RGB channels
        result = np.where(crop_mask[:, :, None] > 0, crop_img, gray_bg)

        return Image.fromarray(result.astype(np.uint8))

    def classify_instance(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        instance_id: int = None,
    ) -> dict:
        """
        Finds the closest semantic class for one masked object observation.

        Steps:
          1. Crop + gray the mask region from the image
          2. Encode the crop through CLIP's image encoder → image vector
          3. Dot product against all pre-encoded class text vectors
          4. Softmax → probabilities → pick the highest

        Args:
            image_np    : (H, W, 3) full image numpy array
            mask_np     : (H, W) binary mask for this instance
            instance_id : integer ID for tracking

        Returns:
            {
                "instance_id":     int,
                "predicted_class": str,    e.g. "chair"
                "confidence":      float,  e.g. 0.82
                "dynamic_bucket":  str,    e.g. "static"
                "top_k":           [("chair", 0.82), ("couch", 0.11), ...]
            }
        """
        # Step 1: get the masked crop as a PIL image
        pil_crop = self._get_masked_crop(image_np, mask_np)

        # Step 2: preprocess (resize 224x224, normalize) + add batch dim
        # shape goes: (H,W,3) → (3,224,224) → (1,3,224,224)
        image_tensor = self.preprocess(pil_crop).unsqueeze(0).to(self.device)

        # Step 3: encode image → vector, normalize to unit length
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)  # (1, 768)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Step 4: dot product with all text vectors
        # (1, 768) @ (768, num_classes) → (num_classes,)
        similarities = (image_features @ self.text_features.T).squeeze(0)

        # Softmax → probabilities, sort descending, take top_k
        probs = similarities.softmax(dim=-1).cpu().numpy()
        top_indices = probs.argsort()[::-1][: self.top_k]
        top_results = [(self.class_labels[i], float(probs[i])) for i in top_indices]

        best_class, best_conf = top_results[0]

        # Fall back to "object" if not confident enough
        if best_conf < self.confidence_threshold:
            best_class = "object"

        # Map class → persistence tier
        bucket = CLASS_TO_BUCKET.get(best_class, "static")

        return {
            "instance_id":     instance_id,
            "predicted_class": best_class,
            "confidence":      best_conf,
            "dynamic_bucket":  bucket,
            "top_k":           top_results,
        }

    def classify_all_instances(
        self,
        image_np: np.ndarray,
        masks: list,
    ) -> list:
        """
        Given a list of images+masks, finds the closest semantic class
        for each object observation (mask). This is the main entry point.

        Args:
            image_np : (H, W, 3) uint8 RGB numpy array
            masks    : list of (H, W) binary numpy arrays from SAM2/SAM3

        Returns:
            List of result dicts, one per mask, each with:
            instance_id, predicted_class, confidence, dynamic_bucket, top_k
        """
        results = []
        for i, mask in enumerate(masks):
            result = self.classify_instance(image_np, mask, instance_id=i)
            results.append(result)
            logger.debug(
                f"  [{i:03d}] {result['predicted_class']:20s} "
                f"conf={result['confidence']:.2f}  "
                f"bucket={result['dynamic_bucket']}"
            )
        return results