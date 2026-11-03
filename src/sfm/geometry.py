import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2

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
        # Optical center in world coordinates: C = -R^T * t
        self.center = -R.T @ self.t

    def project_point(self, point_3d: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Projects a 3D world coordinate [X, Y, Z] onto the 2D image plane.
        Returns: ((u, v), depth) if in front of camera, else (None, depth).
        """
        pt_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float64).reshape(4, 1)
        projected = self.P @ pt_h
        depth = float(projected[2, 0])

        if depth <= 1e-4:
            return None, depth  # Behind camera

        u = float(projected[0, 0] / depth)
        v = float(projected[1, 0] / depth)

        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v), depth
        return None, depth

    def is_visible(self, point_3d: np.ndarray) -> bool:
        coords, depth = self.project_point(point_3d)
        return coords is not None and depth > 0


class MultiViewGeometry:
    """
    Handles Multi-View Geometry, Feature Matching, Sparse Tie-Point Triangulation,
    and Cross-View Camera Calibration.
    """

    def __init__(self, min_matches: int = 15):
        self.min_matches = min_matches
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, img_bgr: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_views(
        self,
        des1: np.ndarray,
        des2: np.ndarray
    ) -> List[cv2.DMatch]:
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    @staticmethod
    def estimate_relative_pose(
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Estimates Essential Matrix, R and t between two calibrated views.
        """
        if len(pts1) < 8 or len(pts2) < 8:
            return None, None, np.zeros((0,), dtype=bool)

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return None, None, np.zeros((0,), dtype=bool)

        mask = mask.ravel().astype(bool)
        if np.sum(mask) < 5:
            return None, None, mask

        _, R, t, mask_pose = cv2.recoverPose(E, pts1[mask], pts2[mask], K)
        return R, t, mask

    @staticmethod
    def triangulate_points(
        P1: np.ndarray,
        P2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> np.ndarray:
        """
        Triangulates 3D points from 2D correspondences across two views.
        Returns: [N, 3] array of 3D coordinates.
        """
        if len(pts1) == 0 or len(pts2) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        pts1_h = pts1.T.astype(np.float64)
        pts2_h = pts2.T.astype(np.float64)

        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        pts4d /= (pts4d[3, :] + 1e-8)  # normalize homogeneous coordinate
        return pts4d[:3, :].T

