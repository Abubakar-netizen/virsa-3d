import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.dataset_loader import LabeledDatasetLoader

logger = logging.getLogger(__name__)

class RealSfMEngine:
    """
    Genuine Structure-from-Motion (SfM) Feature Extraction & Stereo Triangulation Engine.
    Uses OpenCV SIFT, Lowe's ratio test matching, RANSAC Essential Matrix recovery,
    and ray triangulation from calibrated pinhole camera models.
    """

    def __init__(
        self,
        max_features: int = 4000,
        ratio_thresh: float = 0.75,
        reproj_thresh_px: float = 4.0,
        working_dimension: int = 1920
    ):
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.reproj_thresh_px = reproj_thresh_px
        self.working_dimension = working_dimension
        self.sift = cv2.SIFT_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def load_and_scale_image(self, img_path: str) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Load image and downscale to working dimension for consistent SfM processing."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        h, w = img.shape[:2]
        orig_shape = (w, h)

        scale = 1.0
        if max(w, h) > self.working_dimension:
            scale = self.working_dimension / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img, scale, orig_shape

    def compute_intrinsic_matrix(self, exif_meta: Dict[str, Any], img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute Pinhole Camera Intrinsic Matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        derived from real EXIF focal length (DJI FC8482 sensor: ~6.4mm x 4.8mm equivalent).
        """
        w, h = img_shape
        cx = w / 2.0
        cy = h / 2.0

        f_35mm = exif_meta.get("focal_length_35mm")
        f_mm = exif_meta.get("focal_length_mm")

        if f_35mm and f_35mm > 0:
            # 35mm film format width is 36.0 mm
            fx = (f_35mm / 36.0) * w
            fy = fx
        elif f_mm and f_mm > 0:
            # DJI Mavic 3 sensor width ~ 17.3 mm (4/3 inch) or 6.4mm depending on lens
            sensor_w_mm = 17.3
            fx = (f_mm / sensor_w_mm) * w
            fy = fx
        else:
            # Standard drone field of view fallback (~84 degrees FOV)
            fx = w * 1.2
            fy = w * 1.2

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return K

    def extract_features(self, img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract SIFT keypoints and 128-dimensional descriptors."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = self.sift.detectAndCompute(gray, None)
        return kps, descs

    def match_features(self, descs1: np.ndarray, descs2: np.ndarray) -> List[cv2.DMatch]:
        """Match features with Lowe's ratio test."""
        if descs1 is None or descs2 is None or len(descs1) < 10 or len(descs2) < 10:
            return []
        matches = self.matcher.knnMatch(descs1, descs2, k=2)
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
        return good

    def triangulate_image_pair(
        self,
        img_path1: str,
        img_path2: str,
        exif1: Dict[str, Any],
        exif2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate Essential Matrix, recover relative camera pose (R, t),
        and triangulate 3D tie points from matched correspondences.
        """
        img1, scale1, (w1, h1) = self.load_and_scale_image(img_path1)
        img2, scale2, (w2, h2) = self.load_and_scale_image(img_path2)

        K1 = self.compute_intrinsic_matrix(exif1, (img1.shape[1], img1.shape[0]))
        K2 = self.compute_intrinsic_matrix(exif2, (img2.shape[1], img2.shape[0]))

        kps1, descs1 = self.extract_features(img1)
        kps2, descs2 = self.extract_features(img2)
        matches = self.match_features(descs1, descs2)

        if len(matches) < 20:
            return {"success": False, "reason": "Insufficient matched correspondences", "points_3d": np.zeros((0, 3))}

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        # Essential Matrix via RANSAC
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None or mask is None:
            return {"success": False, "reason": "RANSAC Essential Matrix failed", "points_3d": np.zeros((0, 3))}

        inliers = mask.ravel() == 1
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if len(pts1_in) < 15:
            return {"success": False, "reason": "Too few RANSAC inliers", "points_3d": np.zeros((0, 3))}

        # Recover relative Pose R, t
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K1)

        # Projection matrices
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t))

        # Triangulation
        pts4d_h = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
        pts3d = (pts4d_h[:3] / pts4d_h[3]).T

        # Filter points in front of both cameras (positive Z depth)
        valid_z = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 200.0)
        # Transform to second camera coordinate frame
        pts3d_cam2 = (R @ pts3d.T + t).T
        valid_z = valid_z & (pts3d_cam2[:, 2] > 0.1)

        valid_pts = pts3d[valid_z]
        valid_pts1_2d = pts1_in[valid_z]
        valid_pts2_2d = pts2_in[valid_z]

        # Extract RGB color for each 3D point from img1
        colors = []
        for p2d in valid_pts1_2d:
            u, v = int(round(p2d[0])), int(round(p2d[1]))
            u = np.clip(u, 0, img1.shape[1] - 1)
            v = np.clip(v, 0, img1.shape[0] - 1)
            b, g, r = img1[v, u]
            colors.append([r / 255.0, g / 255.0, b / 255.0])
        colors = np.array(colors, dtype=np.float32)

        return {
            "success": True,
            "inlier_count": int(np.sum(inliers)),
            "valid_triangulated_count": len(valid_pts),
            "points_3d": valid_pts,
            "pts1_2d": valid_pts1_2d,
            "pts2_2d": valid_pts2_2d,
            "colors": colors,
            "R": R,
            "t": t,
            "K1": K1,
            "K2": K2,
            "P1": P1,
            "P2": P2
        }

if __name__ == "__main__":
    loader = LabeledDatasetLoader()
    print(f"Loaded {len(loader.items)} items.")
    engine = RealSfMEngine()

    # Test triangulation on an adjacent pair of real Temple 1 Front images
    t1_fronts = [it for it in loader.items if it["category_name"] == "Temple 1 Front"]
    if len(t1_fronts) >= 2:
        img1 = t1_fronts[0]
        img2 = t1_fronts[1]
        print(f"Testing real SfM triangulation between:")
        print(f"  Image 1: {img1['filename']}")
        print(f"  Image 2: {img2['filename']}")

        res = engine.triangulate_image_pair(img1["filepath"], img2["filepath"], img1["exif"], img2["exif"])
        print("SfM Triangulation Result:")
        print(f"  Success:             {res['success']}")
        if res["success"]:
            print(f"  RANSAC Inliers:      {res['inlier_count']}")
            print(f"  Triangulated Points: {res['valid_triangulated_count']}")
            print(f"  Point bounds:        X=[{res['points_3d'][:,0].min():.2f}, {res['points_3d'][:,0].max():.2f}], Z=[{res['points_3d'][:,2].min():.2f}, {res['points_3d'][:,2].max():.2f}]")

