import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class Sam2HeritageSegmenter:
    """
    Genuine Meta SAM 2 (Segment Anything Model 2.1) Heritage Segmenter.
    Uses official 'facebook/sam2.1-hiera-tiny' foundation model weights to produce
    authentic curvilinear 2D semantic masks of Hindu Shahi stone temple shikharas,
    arches, and stone plinths from UAV drone imagery.
    Loads precomputed SAM 2 masks from data/processed/sam2_masks/ or runs live neural inference.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        cache_dir: str = "data/processed/sam2_masks",
        max_dimension: int = 1024,
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_dimension = max_dimension
        self.device = device
        self.model = None
        self.processor = None

    def _init_neural_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import Sam2Model, Sam2Processor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SAM2] Initializing Meta SAM 2.1 ({self.model_id}) on {self.device}...")
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        print(f"[SAM2] Foundation model loaded successfully ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params).")

    def _get_cache_path(self, filename: str) -> Tuple[Path, Path]:
        base_name = Path(filename).stem
        mask_file = self.cache_dir / f"{base_name}_sam2_mask.png"
        conf_file = self.cache_dir / f"{base_name}_sam2_conf.npy"
        return mask_file, conf_file

    def segment_image(
        self,
        img_path: str,
        category_meta: Dict[str, Any],
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment drone image using genuine Meta SAM 2.1 model:
          0: Sky / Distant Background
          1: Ground / Terrain (courtyard, plinth stone, rocky trail)
          2: Heritage Structure (Meta SAM 2.1 predicted curvilinear mask)
          3: Vegetation (cliff trees, bushes)

        Returns:
          label_map: [H, W] int32 (0 to 3)
          conf_map:  [H, W] float32 (confidence 0.0 to 1.0)
        """
        img_name = Path(img_path).name
        mask_file, conf_file = self._get_cache_path(img_name)

        if not force_recompute and mask_file.exists() and conf_file.exists():
            label_map = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE).astype(np.int32)
            conf_map = np.load(str(conf_file))
            return label_map, conf_map

        # Lazy load neural model if computing from scratch
        self._init_neural_model()
        import torch
        from PIL import Image

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        orig_h, orig_w = raw_bgr.shape[:2]
        scale = 1.0
        if max(orig_h, orig_w) > self.max_dimension:
            scale = self.max_dimension / float(max(orig_h, orig_w))
            nw, nh = int(orig_w * scale), int(orig_h * scale)
            small_bgr = cv2.resize(raw_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small_bgr = raw_bgr.copy()
            nh, nw = orig_h, orig_w

        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(small_rgb)

        # Viewpoint-Aware Point Prompts for SAM 2
        side = category_meta.get("side", "unknown")
        if "top" in side:
            pts = [[[[nw // 2, nh // 2], [nw // 2, int(nh * 0.40)], [nw // 2, int(nh * 0.60)]]]]
            lbls = [[[1, 1, 1]]]
        elif "bottom" in side:
            pts = [[[[nw // 2, int(nh * 0.35)], [nw // 2, int(nh * 0.55)]]]]
            lbls = [[[1, 1]]]
        else:
            pts = [[[[nw // 2, int(nh * 0.30)], [nw // 2, int(nh * 0.50)], [nw // 2, int(nh * 0.70)]]]]
            lbls = [[[1, 1, 1]]]

        inputs = self.processor(images=pil_img, input_points=pts, input_labels=lbls, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed_masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"]
        )

        scores = outputs.iou_scores[0, 0].cpu().numpy()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        structure_mask = processed_masks[0][0, 0, best_idx].cpu().numpy().astype(np.uint8)
        if structure_mask.shape != (nh, nw):
            structure_mask = cv2.resize(structure_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Vegetation isolation via Excess Green Index (ExG)
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        rgb_f = small_rgb.astype(np.float32) / 255.0
        exg = 2.0 * rgb_f[:, :, 1] - rgb_f[:, :, 0] - rgb_f[:, :, 2]
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        veg_mask = ((hue >= 25) & (hue <= 88) & (sat > 0.14) & (exg > 0.015) & (structure_mask == 0)).astype(np.uint8)

        # Sky isolation
        sky_mask = ((val > 0.68) & (sat < 0.22) & (np.arange(nh)[:, None] < nh * 0.38) & (structure_mask == 0) & (veg_mask == 0)).astype(np.uint8)

        # 4-Class Semantic Label Map
        label_map = np.ones((nh, nw), dtype=np.int32)
        label_map[sky_mask == 1] = 0
        label_map[veg_mask == 1] = 3
        label_map[structure_mask == 1] = 2

        conf_map = np.full((nh, nw), 0.80, dtype=np.float32)
        conf_map[structure_mask == 1] = float(np.clip(best_score, 0.70, 0.99))
        conf_map[veg_mask == 1] = 0.85
        conf_map[sky_mask == 1] = 0.92

        cv2.imwrite(str(mask_file), label_map.astype(np.uint8))
        np.save(str(conf_file), conf_map)

        return label_map, conf_map
