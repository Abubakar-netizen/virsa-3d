import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ContourSegmenter:
    """
    Contour-Accurate Multi-Class Semantic Segmentation Engine for Cultural Heritage Drone Imagery.
    Extracts authentic curvilinear silhouettes of Hindu Shahi stone temple shikharas,
    chaitya arches, stone masonry plinths, vegetation, and sky.
    Replaces coarse rectangular approximations with genuine contour extraction via
    GrabCut energy minimization (GMM graph-cuts) + Excess Green vegetation indexing + Canny edge refinement.
    """

    def __init__(self, cache_dir: str = "data/processed/contour_masks", max_dimension: int = 1280):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_dimension = max_dimension

    def _get_cache_path(self, filename: str) -> Path:
        base_name = Path(filename).stem
        return self.cache_dir / f"{base_name}_mask.png"

    def segment_image(
        self,
        img_path: str,
        category_meta: Dict[str, Any],
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment an image into 4 classes:
          0: Sky / Background
          1: Ground / Terrain (plinth, courtyard, cliff stone)
          2: Heritage Structure (temple masonry, shikhara, sanctum)
          3: Vegetation (bushes, cliff trees)

        Returns:
          label_map: [H, W] int32 array (0 to 3)
          conf_map:  [H, W] float32 array (confidence 0.0 to 1.0)
        """
        img_name = Path(img_path).name
        cache_file = self._get_cache_path(img_name)
        conf_file = self.cache_dir / f"{Path(img_name).stem}_conf.npy"

        if not force_recompute and cache_file.exists() and conf_file.exists():
            label_map = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE).astype(np.int32)
            conf_map = np.load(str(conf_file))
            return label_map, conf_map

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        orig_h, orig_w = img.shape[:2]
        scale = 1.0
        if max(orig_h, orig_w) > self.max_dimension:
            scale = self.max_dimension / float(max(orig_h, orig_w))
            nw, nh = int(orig_w * scale), int(orig_h * scale)
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small = img.copy()
            nh, nw = orig_h, orig_w

        # 1. Vegetation Isolation via Excess Green Index (ExG) and Hue
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = 2.0 * g - r - b
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        veg_mask = ((hue >= 25) & (hue <= 88) & (sat > 0.14) & (exg > 0.015)).astype(np.uint8)

        # 2. Geometric Prior Box adapted to camera viewpoint (side vs top vs front)
        side = category_meta.get("side", "unknown")
        temple_id = category_meta.get("temple", 1)

        if "top" in side:
            # Top-down nadir drone view
            pad_x = int(nw * 0.15)
            pad_y = int(nh * 0.15)
            rect = (pad_x, pad_y, nw - 2 * pad_x, nh - 2 * pad_y)
        elif "bottom" in side:
            # Low angle oblique looking up
            pad_x = int(nw * 0.12)
            pad_y = int(nh * 0.05)
            rect = (pad_x, pad_y, nw - 2 * pad_x, int(nh * 0.85))
        else:
            # Standard orbital facade view
            pad_x = int(nw * 0.12)
            pad_y = int(nh * 0.08)
            rect = (pad_x, pad_y, nw - 2 * pad_x, nh - 2 * pad_y)

        # 3. GrabCut Energy Minimization for Curvilinear Silhouette Extraction
        gc_mask = np.zeros((nh, nw), dtype=np.uint8)
        gc_mask.fill(cv2.GC_BGD)
        gc_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_FGD

        # Obvious sky corners in upper portion
        corner_w = int(nw * 0.20)
        corner_h = int(nh * 0.20)
        gc_mask[:corner_h, :corner_w] = cv2.GC_BGD
        gc_mask[:corner_h, nw - corner_w:] = cv2.GC_BGD

        # Mark confirmed vegetation as background relative to stone structure
        gc_mask[veg_mask == 1] = cv2.GC_BGD

        # Definite foreground prompt: high texture stone masonry in center core
        core_x = int(nw * 0.35)
        core_w = int(nw * 0.30)
        core_y = int(nh * 0.30)
        core_h = int(nh * 0.40)
        gc_mask[core_y:core_y + core_h, core_x:core_x + core_w] = cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(small, gc_mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)
            structure_binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except Exception as e:
            logger.warning(f"GrabCut failed on {img_name}: {e}. Falling back to color/edge thresholding.")
            # Fallback to stone color mask
            stone_hsv = ((hue >= 10) & (hue <= 38) & (sat >= 0.12) & (val >= 0.20) & (val <= 0.85))
            structure_binary = stone_hsv.astype(np.uint8)

        # Morphological refinement: close micro-cavities in eroded stone, remove isolated salt noise
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        structure_binary = cv2.morphologyEx(structure_binary, cv2.MORPH_CLOSE, k_close)
        structure_binary = cv2.morphologyEx(structure_binary, cv2.MORPH_OPEN, k_open)

        # 4. Synthesize 4-Class Semantic Label Map
        # 0: Sky/Distant Background
        # 1: Ground/Terrain (Courtyard, Plinth base, Surrounding rocky hills)
        # 2: Heritage Structure (Temple Shikhara, Facade, Sanctum)
        # 3: Vegetation
        label_map = np.ones((nh, nw), dtype=np.int32)  # Default: Ground (1)

        # Sky detection: high luminance, low saturation, upper vertical zone
        sky_mask = (val > 0.68) & (sat < 0.22) & (np.arange(nh)[:, None] < nh * 0.38) & (structure_binary == 0)
        label_map[sky_mask] = 0

        # Vegetation
        label_map[veg_mask == 1] = 3

        # Structure (curvilinear contour)
        label_map[structure_binary == 1] = 2

        # 5. Calibrated Confidence Map
        # Points deep inside structure have 0.95 confidence; boundary pixels near contour have 0.70
        dist_transform = cv2.distanceTransform(structure_binary, cv2.DIST_L2, 5)
        max_dist = dist_transform.max() if dist_transform.max() > 0 else 1.0
        conf_map = np.full((nh, nw), 0.80, dtype=np.float32)
        conf_map[structure_binary == 1] = 0.70 + 0.25 * (dist_transform[structure_binary == 1] / max_dist)
        conf_map[veg_mask == 1] = 0.85
        conf_map[sky_mask] = 0.90

        # Save to cache
        cv2.imwrite(str(cache_file), label_map.astype(np.uint8))
        np.save(str(conf_file), conf_map)

        return label_map, conf_map

    def get_or_create_label_map(self, img_path: str, category_meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        return self.segment_image(img_path, category_meta)

