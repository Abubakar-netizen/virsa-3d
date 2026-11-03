import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2
from pathlib import Path

class CameraView:
    """Represents a single calibrated UAV camera station."""
    def __init__(
        self,
        view_id: str,
        image_path: str,
        image_size: Tuple[int, int],  # (width, height)
        K: np.ndarray,                # 3x3 Intrinsic matrix
        R: np.ndarray,                # 3x3 Rotation matrix
        t: np.ndarray                 # 3x1 Translation vector
    ):
        self.view_id = view_id
        self.image_path = image_path
        self.width, self.height = image_size
        self.K = K
        self.R = R
        self.t = t.reshape(3, 1)
        self.P = K @ np.hstack([R, self.t])  # 3x4 Camera Projection Matrix
        self.center = -R.T @ self.t

    def project_point(self, point_3d: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Projects a 3D world coordinate [X, Y, Z] onto the 2D image plane."""
        pt_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float64).reshape(4, 1)
        projected = self.P @ pt_h
        depth = float(projected[2, 0])

        if depth <= 1e-4:
            return None, depth

        u = float(projected[0, 0] / depth)
        v = float(projected[1, 0] / depth)

        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v), depth
        return None, depth

    def is_visible(self, point_3d: np.ndarray) -> bool:
        coords, depth = self.project_point(point_3d)
        return coords is not None and depth > 0


class RealSfMTriangulator:
    """
    Structure-from-Motion (SfM) Triangulation Engine.
    Extracts real 3D point cloud and architectural geometry of the historical temple
    directly from overlapping UAV drone imagery pairs.
    """

    def __init__(self, max_features: int = 1500):
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def triangulate_dataset(
        self,
        camera_views: List[CameraView],
        step_stride: int = 1,
        max_dist_pairs: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts real 3D tie points and true sampled RGB colors from overlapping drone views.
        Returns:
            positions: [N, 3] float32 world coordinates
            colors: [N, 3] uint8 RGB colors
            point_observations: list of dicts mapping each point to observed views [(view_idx, u, v)]
        """
        all_points = []
        all_colors = []
        point_observations = []
        n_views = len(camera_views)

        # Pre-extract features for all views
        print(f"[SfM] Detecting visual keypoints across {n_views} drone images...")
        features_cache = []
        images_cache = []

        for idx, view in enumerate(camera_views):
            img_bgr = cv2.imread(view.image_path)
            if img_bgr is None:
                features_cache.append(([], None))
                images_cache.append(None)
                continue
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            kps, des = self.detector.detectAndCompute(gray, None)
            features_cache.append((kps, des))
            images_cache.append(img_bgr)

        # Match adjacent and overlapping orbital views
        print(f"[SfM] Triangulating real 3D architectural tie-points from overlapping pairs...")
        for i in range(0, n_views, step_stride):
            kps1, des1 = features_cache[i]
            img1 = images_cache[i]
            if des1 is None or len(kps1) < 20 or img1 is None:
                continue

            view1 = camera_views[i]

            for offset in range(1, max_dist_pairs + 1):
                j = (i + offset) % n_views
                kps2, des2 = features_cache[j]
                img2 = images_cache[j]
                if des2 is None or len(kps2) < 20 or img2 is None:
                    continue

                view2 = camera_views[j]

                # Match features
                matches = self.matcher.match(des1, des2)
                if len(matches) < 25:
                    continue

                # Sort by descriptor distance
                matches = sorted(matches, key=lambda x: x.distance)[:400]

                pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

                # RANSAC filtering
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.0, 0.99)
                if mask is None or F is None:
                    continue

                mask = mask.ravel().astype(bool)
                pts1_good = pts1[mask]
                pts2_good = pts2[mask]

                if len(pts1_good) < 15:
                    continue

                # Triangulate points using projection matrices
                pts1_h = pts1_good.T
                pts2_h = pts2_good.T
                pts4d = cv2.triangulatePoints(view1.P, view2.P, pts1_h, pts2_h)
                pts4d /= (pts4d[3, :] + 1e-8)
                pts3d = pts4d[:3, :].T

                # Depth and bounds filtering (keep realistic points in front of cameras)
                for pt_idx, pt in enumerate(pts3d):
                    # Check depth in both cameras
                    coords1, d1 = view1.project_point(pt)
                    coords2, d2 = view2.project_point(pt)

                    if coords1 is not None and coords2 is not None and 1.0 < d1 < 45.0 and 1.0 < d2 < 45.0:
                        # Sample true color from image 1
                        u, v = int(round(coords1[0])), int(round(coords1[1]))
                        if 0 <= v < img1.shape[0] and 0 <= u < img1.shape[1]:
                            b, g, r = img1[v, u]
                            all_points.append(pt)
                            all_colors.append([r, g, b])
                            point_observations.append([
                                (view1.view_id, u, v),
                                (view2.view_id, int(round(coords2[0])), int(round(coords2[1])))
                            ])

        if len(all_points) == 0:
            print("[SfM WARNING] No points triangulated. Returning default cloud.")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), []

        positions = np.array(all_points, dtype=np.float32)
        colors = np.array(all_colors, dtype=np.uint8)

        # Spatial decimation / voxel filtering to remove duplicates and distribute evenly
        voxel_size = 0.15  # 15cm spatial binning
        voxel_idx = np.floor(positions / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_idx, axis=0, return_index=True)

        positions = positions[unique_indices]
        colors = colors[unique_indices]
        filtered_observations = [point_observations[idx] for idx in unique_indices]

        print(f"[SfM SUCCESS] Successfully triangulated {len(positions)} REAL 3D architectural tie-points!")
        return positions, colors, filtered_observations
