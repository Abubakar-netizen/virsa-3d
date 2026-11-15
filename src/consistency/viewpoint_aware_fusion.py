import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial import cKDTree

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class ViewpointAwareFusionEngine:
    """
    Viewpoint-Aware, Normal-Guided, and Occlusion-Culled 2D-to-3D Label Fusion Engine.
    Implements:
      - B1: Direct Lifting (Fadilah et al. 2026)
      - B2: Unweighted Majority Voting (Pellis et al. 2022)
      - Proposed Virsa3D:
          * 3D Surface Normals via PCA Eigen-decomposition (Viewing Angle Filtering)
          * Scale-Adaptive Depth-Aware Occlusion Culling via 2D Z-Buffering
          * Mask Boundary Confidence Weighting
          * Normalized Shannon Entropy Multi-View Uncertainty Quantification
    """

    def __init__(
        self,
        num_classes: int = 4,
        z_grid_step: int = 32,
        occlusion_tol_m: float = 0.50,
        occlusion_tol_ratio: float = 0.35
    ):
        self.num_classes = num_classes  # 0: Sky/Background, 1: Ground/Terrain, 2: Heritage Structure, 3: Vegetation
        self.z_grid_step = z_grid_step
        self.occlusion_tol_m = occlusion_tol_m
        self.occlusion_tol_ratio = occlusion_tol_ratio

    @staticmethod
    def compute_point_cloud_normals_pca(
        points: np.ndarray,
        k_neighbors: int = 20,
        reference_cam_centers: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute genuine 3D surface normal unit vectors for each point via
        Local Principal Component Analysis (PCA) on K-Nearest Neighbors.
        Covariance C = (1/k) * sum((p_j - p_mean)(p_j - p_mean)^T).
        The normal is the eigenvector corresponding to the smallest eigenvalue.
        Orients normals towards the observing camera cluster so n . (C - X) > 0.
        """
        n_pts = len(points)
        if n_pts < k_neighbors:
            return np.tile(np.array([0.0, 0.0, -1.0]), (n_pts, 1))

        tree = cKDTree(points)
        dists, indices = tree.query(points, k=k_neighbors)

        normals = np.zeros((n_pts, 3), dtype=np.float64)

        # Reference viewpoint center for outward orientation
        if reference_cam_centers is not None and len(reference_cam_centers) > 0:
            cam_cluster_center = np.mean(reference_cam_centers, axis=0)
        else:
            cam_cluster_center = np.array([0.0, 0.0, 0.0])

        for i in range(n_pts):
            nbr_indices = indices[i]
            nbr_pts = points[nbr_indices]
            nbr_mean = np.mean(nbr_pts, axis=0)

            # 3x3 Covariance matrix
            diff = nbr_pts - nbr_mean
            cov = (diff.T @ diff) / float(k_neighbors)

            # Eigen-decomposition (Hermitian/Symmetric matrix)
            eigenvals, eigenvecs = np.linalg.eigh(cov)

            # Smallest eigenvalue corresponds to index 0
            n_vec = eigenvecs[:, 0]
            norm_val = np.linalg.norm(n_vec)
            if norm_val > 1e-6:
                n_vec = n_vec / norm_val
            else:
                n_vec = np.array([0.0, 0.0, -1.0])

            # Consistent outward orientation: normal points towards the observing camera positions
            view_dir = cam_cluster_center - points[i]
            if np.dot(n_vec, view_dir) < 0:
                n_vec = -n_vec

            normals[i] = n_vec

        return normals

    def build_camera_z_buffer(self, points: np.ndarray, cam: Dict[str, Any]) -> np.ndarray:
        """
        Construct a discrete 2D minimum-depth buffer (Z-buffer) for a camera view.
        Rasterizes all 3D points projected into the camera frustum onto a spatial grid.
        """
        w, h = cam["width"], cam["height"]
        gw = int(np.ceil(w / float(self.z_grid_step)))
        gh = int(np.ceil(h / float(self.z_grid_step)))

        z_buf = np.full((gh, gw), np.inf, dtype=np.float32)

        R, t, K = cam["R"], cam["t"].flatten(), cam["K"]
        pts_cam = (R @ points.T).T + t
        zs = pts_cam[:, 2]

        valid = zs > 0.05
        if not np.any(valid):
            return z_buf

        p_homo = (K @ pts_cam[valid].T).T
        us = p_homo[:, 0] / zs[valid]
        vs = p_homo[:, 1] / zs[valid]

        in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        valid_u = us[in_bounds]
        valid_v = vs[in_bounds]
        valid_z = zs[valid][in_bounds]

        gu = np.clip((valid_u / self.z_grid_step).astype(np.int32), 0, gw - 1)
        gv = np.clip((valid_v / self.z_grid_step).astype(np.int32), 0, gh - 1)

        for u_idx, v_idx, z_val in zip(gu, gv, valid_z):
            if z_val < z_buf[v_idx, u_idx]:
                z_buf[v_idx, u_idx] = z_val

        return z_buf

    def project_point_to_camera(
        self,
        pt3d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        img_w: int,
        img_h: int
    ) -> Tuple[bool, float, float, float]:
        """
        Project 3D point into camera view.
        Returns (is_in_frustum, u_px, v_px, depth_z).
        """
        pt_cam = R @ pt3d + t.flatten()
        z = pt_cam[2]
        if z <= 0.05:  # Behind camera plane
            return False, 0.0, 0.0, z

        p_homo = K @ pt_cam
        u = p_homo[0] / z
        v = p_homo[1] / z

        if 0 <= u < img_w and 0 <= v < img_h:
            return True, u, v, z
        return False, u, v, z

    def run_direct_lifting(
        self,
        points: np.ndarray,
        cameras: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Baseline 1 (Fadilah et al. 2026):
        Direct lifting: assign label from the single closest camera view in frustum.
        """
        n_pts = len(points)
        pred_labels = np.zeros(n_pts, dtype=np.int32)

        for i in range(n_pts):
            pt = points[i]
            best_dist = float('inf')
            best_label = 0

            for cam in cameras:
                vis, u, v, depth = self.project_point_to_camera(
                    pt, cam["R"], cam["t"], cam["K"], cam["width"], cam["height"]
                )
                if vis and depth < best_dist:
                    best_dist = depth
                    ui = int(np.clip(round(u), 0, cam["width"] - 1))
                    vi = int(np.clip(round(v), 0, cam["height"] - 1))
                    best_label = cam["label_map"][vi, ui]

            pred_labels[i] = best_label
        return pred_labels

    def run_unweighted_majority_voting(
        self,
        points: np.ndarray,
        cameras: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Baseline 2 (Pellis et al. 2022):
        Unweighted majority voting across all camera rays seeing the 3D point in frustum.
        """
        n_pts = len(points)
        pred_labels = np.zeros(n_pts, dtype=np.int32)

        for i in range(n_pts):
            pt = points[i]
            votes = np.zeros(self.num_classes, dtype=np.float32)

            for cam in cameras:
                vis, u, v, depth = self.project_point_to_camera(
                    pt, cam["R"], cam["t"], cam["K"], cam["width"], cam["height"]
                )
                if vis:
                    ui = int(np.clip(round(u), 0, cam["width"] - 1))
                    vi = int(np.clip(round(v), 0, cam["height"] - 1))
                    lbl = cam["label_map"][vi, ui]
                    if 0 <= lbl < self.num_classes:
                        votes[lbl] += 1.0

            if np.sum(votes) > 0:
                pred_labels[i] = np.argmax(votes)
            else:
                pred_labels[i] = 0

        return pred_labels

    def run_viewpoint_aware_fusion(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        cameras: List[Dict[str, Any]],
        use_occlusion_culling: bool = True,
        use_normal_weighting: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Proposed Virsa3D Method:
        Integrates:
          1. 3D Surface Normal Cosine Weighting (Grazing angle and backface suppression)
          2. Scale-Adaptive Depth-Aware Z-Buffer Occlusion Culling
          3. Distance Attenuation Weighting (1 / d)
          4. 2D Mask Boundary Confidence
          5. Normalized Shannon Entropy Multi-View Uncertainty
        """
        n_pts = len(points)
        pred_labels = np.zeros(n_pts, dtype=np.int32)
        uncertainties = np.zeros(n_pts, dtype=np.float32)
        conflicts_resolved = 0
        total_rays_evaluated = 0
        occluded_rays_culled = 0
        backfaces_culled = 0

        # Extract camera centers
        cam_centers = [-cam["R"].T @ cam["t"].flatten() for cam in cameras]

        # Compute normals if not supplied
        if normals is None and use_normal_weighting:
            normals = self.compute_point_cloud_normals_pca(points, k_neighbors=20, reference_cam_centers=cam_centers)

        # Precompute Z-buffers for all cameras
        z_buffers = []
        if use_occlusion_culling:
            for cam in cameras:
                z_buffers.append(self.build_camera_z_buffer(points, cam))

        log_k = np.log(self.num_classes) if self.num_classes > 1 else 1.0

        for i in range(n_pts):
            pt = points[i]
            n_vec = normals[i] if (normals is not None and use_normal_weighting) else None
            weighted_probs = np.zeros(self.num_classes, dtype=np.float64)
            raw_labels_seen = []

            for cam_idx, cam in enumerate(cameras):
                vis, u, v, depth = self.project_point_to_camera(
                    pt, cam["R"], cam["t"], cam["K"], cam["width"], cam["height"]
                )
                if not vis:
                    continue

                total_rays_evaluated += 1

                # 1. Depth-Aware Occlusion Culling (Scale-Adaptive Z-Buffer Check)
                if use_occlusion_culling and len(z_buffers) > cam_idx:
                    z_buf = z_buffers[cam_idx]
                    gu = int(np.clip(u / self.z_grid_step, 0, z_buf.shape[1] - 1))
                    gv = int(np.clip(v / self.z_grid_step, 0, z_buf.shape[0] - 1))
                    min_depth = z_buf[gv, gu]

                    # Scale-adaptive tolerance: 35% of local depth or 0.40 units minimum
                    tol = max(0.40, min_depth * self.occlusion_tol_ratio)
                    if depth > min_depth + tol:
                        occluded_rays_culled += 1
                        continue  # Point is occluded behind foreground surface

                ui = int(np.clip(round(u), 0, cam["width"] - 1))
                vi = int(np.clip(round(v), 0, cam["height"] - 1))
                lbl = cam["label_map"][vi, ui]

                # 2. Viewing Angle & Normal Alignment (cos theta)
                if n_vec is not None:
                    cam_center = cam_centers[cam_idx]
                    ray_dir = cam_center - pt
                    ray_norm = np.linalg.norm(ray_dir)
                    if ray_norm > 1e-5:
                        ray_dir /= ray_norm
                    else:
                        ray_dir = np.array([0.0, 0.0, -1.0])

                    cos_theta = float(np.dot(n_vec, ray_dir))
                    if cos_theta <= -0.10:
                        backfaces_culled += 1
                        continue  # Back-facing surface, invisible to this camera view

                    # Soft grazing angle attenuation: max(0.05, cos theta)^1.2
                    angle_weight = np.power(max(0.05, cos_theta), 1.2)
                else:
                    angle_weight = 1.0

                # 3. Distance Attenuation (1 / max(1.0, depth))
                dist_weight = 1.0 / max(1.0, depth)

                # 4. 2D Mask Confidence
                conf = cam.get("conf_map", None)
                c_val = float(conf[vi, ui]) if conf is not None else 1.0

                weight = angle_weight * dist_weight * c_val

                if 0 <= lbl < self.num_classes:
                    weighted_probs[lbl] += weight
                    raw_labels_seen.append(lbl)

            sum_w = np.sum(weighted_probs)
            if sum_w > 0:
                p = weighted_probs / sum_w
                best_class = int(np.argmax(p))
                pred_labels[i] = best_class

                p_safe = p[p > 0]
                entropy = -np.sum(p_safe * np.log(p_safe)) / log_k
                uncertainties[i] = float(np.clip(entropy, 0.0, 1.0))

                unique_labels = set(raw_labels_seen)
                if len(unique_labels) > 1:
                    conflicts_resolved += 1
            else:
                pred_labels[i] = 0
                uncertainties[i] = 1.0

        telemetry = {
            "total_points": n_pts,
            "conflicts_resolved": conflicts_resolved,
            "total_rays_evaluated": total_rays_evaluated,
            "occluded_rays_culled": occluded_rays_culled,
            "backfaces_culled": backfaces_culled,
            "mean_uncertainty": float(np.mean(uncertainties)),
            "high_confidence_ratio": float(np.mean(uncertainties < 0.25))
        }

        return pred_labels, uncertainties, telemetry
