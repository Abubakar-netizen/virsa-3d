import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    2D Semantic Segmentation Module for Cultural Heritage Complexes.
    Faithfully replicates Step 2 of Fadilah et al. (ISPRS 2026):
    Extracts per-class masks using concept/text prompts (e.g. 'roof', 'ground', 'vegetation')
    and outputs binary masks and RGBA masked images ready for 3DGS.
    """

    def __init__(
        self,
        classes: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "outputs/baseline/segmentation",
        use_neural_prompts: bool = True
    ):
        if classes is None:
            self.classes = [
                {"id": 0, "name": "background", "color": [0, 0, 0], "prompt": "sky background"},
                {"id": 1, "name": "ground", "color": [128, 64, 128], "prompt": "ground terrain pathway courtyard road soil"},
                {"id": 2, "name": "structure", "color": [70, 70, 70], "prompt": "historical temple roof architectural stone wall dome facade"},
                {"id": 3, "name": "vegetation", "color": [107, 142, 35], "prompt": "trees bushes vegetation foliage greenery plants"}
            ]
        else:
            self.classes = classes

        self.class_map = {c["id"]: c["name"] for c in self.classes}
        self.class_colors = {c["id"]: c["color"] for c in self.classes}
        self.output_dir = Path(output_dir)
        self.masks_dir = self.output_dir / "masks"
        self.rgba_dir = self.output_dir / "rgba"
        self.overlays_dir = self.output_dir / "overlays"

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.rgba_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)

        self.use_neural_prompts = use_neural_prompts
        self.model = None
        self.processor = None
        
        if self.use_neural_prompts:
            self._init_neural_model()

    def _init_neural_model(self):
        """Attempt to initialize zero-shot concept-prompted segmentation model (CLIPSeg)."""
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            logger.info("Loading CLIPSeg zero-shot text-prompted segmentation model...")
            self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model.eval()
            print("[SEGMENTER] Neural concept-prompt model (CLIPSeg) loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load neural model ({e}). Using calibrated CV heuristic proxy.")
            self.model = None
            self.processor = None

    def _neural_prompt_segmentation(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run text prompts on image using CLIPSeg to produce multi-class probability maps."""
        from PIL import Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        h, w = img_bgr.shape[:2]

        prompts = [c.get("prompt", c["name"]) for c in self.classes if c["id"] != 0]
        class_ids = [c["id"] for c in self.classes if c["id"] != 0]

        inputs = self.processor(
            text=prompts,
            images=[pil_img] * len(prompts),
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits: [num_prompts, 352, 352]
            preds = torch.sigmoid(outputs.logits)

        probs = np.zeros((h, w, len(self.classes)), dtype=np.float32)
        probs[:, :, 0] = 0.20  # background prior

        for idx, cid in enumerate(class_ids):
            pred_map = preds[idx].cpu().numpy()
            resized_pred = cv2.resize(pred_map, (w, h), interpolation=cv2.INTER_LINEAR)
            probs[:, :, cid] = resized_pred

        exp_p = np.exp(probs * 3.0)
        norm_probs = exp_p / np.sum(exp_p, axis=2, keepdims=True)
        return norm_probs

    def _extract_heuristic_semantic_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Calibrated feature-driven segmentation proxy for cultural heritage outdoor drone imagery.
        Computes HSV, ExG (Excess Green index for vegetation), texture gradients, and structural edges.
        """
        h, w = img_bgr.shape[:2]
        num_classes = len(self.classes)
        probs = np.zeros((h, w, num_classes), dtype=np.float32)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        # Excess Green Index (ExG = 2G - R - B)
        exg = 2.0 * g - r - b
        veg_score = np.clip(exg * 2.5 + (sat * 0.5), 0.0, 1.0)
        is_green_hue = ((hue >= 28) & (hue <= 90)).astype(np.float32)
        veg_prob = np.clip(veg_score * is_green_hue, 0.0, 1.0)

        # Structure / Temple detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)
        grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-6) * 3.0, 0.0, 1.0)

        stone_mask = ((hue >= 10) & (hue <= 35) & (sat >= 0.15) & (sat <= 0.65) & (val >= 0.25)) | \
                     ((sat <= 0.20) & (val >= 0.25) & (val <= 0.85))
        structure_prob = np.clip(0.6 * grad_norm + 0.4 * stone_mask.astype(np.float32) - 0.5 * veg_prob, 0.0, 1.0)

        # Ground detection
        ground_prob = np.clip((1.0 - grad_norm) * (1.0 - veg_prob) * (val > 0.15), 0.0, 1.0)

        probs[:, :, 1] = ground_prob
        probs[:, :, 2] = structure_prob
        probs[:, :, 3] = veg_prob
        probs[:, :, 0] = 0.15

        exp_p = np.exp(probs * 4.0)
        norm_probs = exp_p / np.sum(exp_p, axis=2, keepdims=True)
        return norm_probs

    def segment_image(
        self,
        image_path: str,
        image_id: str,
        add_view_noise: bool = False,
        noise_level: float = 0.08
    ) -> Dict[str, Any]:
        """Performs 2D semantic segmentation matching Step 2 of Fadilah et al."""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        h, w = img_bgr.shape[:2]

        if self.model is not None and self.processor is not None:
            try:
                probs = self._neural_prompt_segmentation(img_bgr)
            except Exception as e:
                logger.warning(f"Neural prompt segmentation failed ({e}), falling back to heuristic.")
                probs = self._extract_heuristic_semantic_features(img_bgr)
        else:
            probs = self._extract_heuristic_semantic_features(img_bgr)

        if add_view_noise:
            noise = np.random.normal(0, noise_level, probs.shape).astype(np.float32)
            noisy_probs = np.clip(probs + noise, 0.01, 0.99)
            norm_probs = noisy_probs / np.sum(noisy_probs, axis=2, keepdims=True)
        else:
            norm_probs = probs

        predicted_labels = np.argmax(norm_probs, axis=2).astype(np.uint8)

        per_class_paths: Dict[str, Dict[str, str]] = {}
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for c in self.classes:
            cid = c["id"]
            cname = c["name"]
            if cid == 0:
                continue

            binary_mask = (predicted_labels == cid).astype(np.uint8) * 255
            mask_filename = f"{image_id}_{cname}.png"
            mask_filepath = self.masks_dir / mask_filename
            cv2.imwrite(str(mask_filepath), binary_mask)

            # Create RGBA masked image as per Step 3 of paper
            rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = binary_mask
            rgba_filename = f"{image_id}_{cname}_rgba.png"
            rgba_filepath = self.rgba_dir / rgba_filename
            cv2.imwrite(str(rgba_filepath), rgba)

            per_class_paths[cname] = {
                "mask": str(mask_filepath).replace("\\", "/"),
                "rgba": str(rgba_filepath).replace("\\", "/")
            }

            color_bgr = c["color"][::-1]
            overlay[binary_mask > 0] = color_bgr

        blended = cv2.addWeighted(img_bgr, 0.65, overlay, 0.35, 0)
        overlay_filepath = self.overlays_dir / f"{image_id}_overlay.png"
        cv2.imwrite(str(overlay_filepath), blended)

        return {
            "image_id": image_id,
            "dimensions": [w, h],
            "labels": predicted_labels,
            "confidence_probabilities": norm_probs,
            "outputs": per_class_paths,
            "overlay_path": str(overlay_filepath).replace("\\", "/")
        }
