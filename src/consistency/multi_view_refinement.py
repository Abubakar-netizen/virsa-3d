import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
from src.sfm.geometry import CameraView

class MultiViewConsistencyRefiner:
    """
    Proposed Virsa3D Multi-View Semantic Consistency and Label Refinement Engine.
    Resolves cross-view semantic label disagreements and suppresses 2D segmentation errors
    before propagating into 3D Gaussian Splatting representation.
    """

    def __init__(
        self,
        agreement_threshold: float = 0.60,
        min_views: int = 2,
        spatial_regularization: bool = True,
        output_dir: str = "outputs/proposed/refined_masks"
    ):
        self.agreement_threshold = agreement_threshold
        self.min_views = min_views
        self.spatial_regularization = spatial_regularization
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgba_dir = self.output_dir / "rgba"
        self.rgba_dir.mkdir(parents=True, exist_ok=True)

    def calculate_cross_view_agreement(
        self,
        point_3d: np.ndarray,
        camera_views: List[CameraView],
        view_label_maps: Dict[str, np.ndarray],
        view_confidence_maps: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Projects a 3D point across all overlapping camera views and calculates
        the semantic consensus class, agreement score, and observation count.
        """
        observed_labels = []
        observed_weights = []
        view_projections = {}

        for view in camera_views:
            coords, depth = view.project_point(point_3d)
            if coords is not None and depth > 0:
                u, v = int(round(coords[0])), int(round(coords[1]))
                lbl_map = view_label_maps.get(view.view_id)
                if lbl_map is not None and 0 <= v < lbl_map.shape[0] and 0 <= u < lbl_map.shape[1]:
                    lbl = int(lbl_map[v, u])
                    weight = 1.0
                    if view_confidence_maps and view.view_id in view_confidence_maps:
                        conf_map = view_confidence_maps[view.view_id]
                        weight = float(conf_map[v, u, lbl]) if conf_map.ndim == 3 else 1.0

                    observed_labels.append(lbl)
                    observed_weights.append(weight)
                    view_projections[view.view_id] = {"coords": (u, v), "label": lbl, "weight": weight}

        if len(observed_labels) < self.min_views:
            return {
                "consensus_label": observed_labels[0] if observed_labels else 0,
                "agreement_score": 1.0 if observed_labels else 0.0,
                "num_views": len(observed_labels),
                "is_consistent": True,
                "projections": view_projections
            }

        # Weighted label voting across overlapping views
        unique_labels = np.unique(observed_labels)
        total_weight = sum(observed_weights)
        label_scores = {}
        for l in unique_labels:
            score = sum(w for lbl, w in zip(observed_labels, observed_weights) if lbl == l)
            label_scores[l] = score / (total_weight + 1e-8)

        consensus_label = max(label_scores.keys(), key=lambda k: label_scores[k])
        agreement_score = label_scores[consensus_label]
        is_consistent = (agreement_score >= self.agreement_threshold)

        return {
            "consensus_label": int(consensus_label),
            "agreement_score": round(float(agreement_score), 4),
            "num_views": len(observed_labels),
            "is_consistent": is_consistent,
            "projections": view_projections
        }

    def refine_view_labels(
        self,
        target_view: CameraView,
        neighbor_views: List[CameraView],
        target_labels: np.ndarray,
        target_probs: np.ndarray,
        neighbor_labels: Dict[str, np.ndarray],
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refines the 2D semantic labels of target_view using multi-view consensus
        from overlapping neighbor views and spatial regularization.
        """
        h, w = target_labels.shape[:2]
        refined_labels = target_labels.copy()
        corrections_count = 0
        total_checked_pixels = 0

        # If a depth map or approximate surface depth is available:
        # Use homography / backprojection to query neighbors.
        # When dense depth is unavailable, evaluate on key structural feature points
        # and bilateral spatial-consensus filters.
        if depth_map is not None:
            # Dense cross-view reprojection refinement
            for v in range(0, h, 2):  # step by 2 for computational efficiency
                for u in range(0, w, 2):
                    z = depth_map[v, u]
                    if z <= 0.1:
                        continue
                    # Reconstruct 3D point in world coords from target view
                    # x_cam = K^-1 * [u, v, 1]^T * z
                    pt_cam = np.linalg.inv(target_view.K) @ np.array([u, v, 1.0]).reshape(3, 1) * z
                    pt_world = target_view.R.T @ (pt_cam - target_view.t)

                    consensus_res = self.calculate_cross_view_agreement(
                        pt_world.ravel(),
                        [target_view] + neighbor_views,
                        {**neighbor_labels, target_view.view_id: target_labels}
                    )

                    total_checked_pixels += 1
                    if consensus_res["num_views"] >= self.min_views and not consensus_res["is_consistent"]:
                        # Label conflict detected! Replace with multi-view consensus class
                        orig = target_labels[v, u]
                        refined_labels[v:v+2, u:u+2] = consensus_res["consensus_label"]
                        if orig != consensus_res["consensus_label"]:
                            corrections_count += 1

        # Spatial consistency regularization (CRF/Morphological smoothing)
        # Suppresses 3DGS floater-inducing single-pixel noise and unifies fragmented regions
        if self.spatial_regularization:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            for cid in np.unique(refined_labels):
                if cid == 0:
                    continue
                c_mask = (refined_labels == cid).astype(np.uint8)
                # Morphological closing fills small holes inside structural roofs
                closed = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
                # Morphological opening removes floating noise
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                refined_labels[opened > 0] = cid

        change_ratio = float(np.mean(refined_labels != target_labels))
        stats = {
            "view_id": target_view.view_id,
            "discrepancies_resolved": corrections_count,
            "label_difference_ratio": round(change_ratio, 4),
            "regularization_applied": self.spatial_regularization
        }

        return refined_labels, stats

    def export_refined_masks(
        self,
        image_id: str,
        img_bgr: np.ndarray,
        refined_labels: np.ndarray,
        classes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, str]]:
        """Exports refined binary masks and RGBA images ready for 3DGS reconstruction."""
        h, w = refined_labels.shape[:2]
        outputs = {}

        for c in classes:
            cid = c["id"]
            cname = c["name"]
            if cid == 0:
                continue

            binary_mask = (refined_labels == cid).astype(np.uint8) * 255
            mask_filename = f"{image_id}_{cname}_refined.png"
            mask_path = self.output_dir / mask_filename
            cv2.imwrite(str(mask_path), binary_mask)

            # Export RGBA masked output for 3DGS
            rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = binary_mask
            rgba_filename = f"{image_id}_{cname}_refined_rgba.png"
            rgba_path = self.rgba_dir / rgba_filename
            cv2.imwrite(str(rgba_path), rgba)

            outputs[cname] = {
                "mask": str(mask_path).replace("\\", "/"),
                "rgba": str(rgba_path).replace("\\", "/")
            }

        return outputs

