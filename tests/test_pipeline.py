import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import SemanticMetrics
from src.sfm.geometry import CameraView, MultiViewGeometry
from src.consistency.multi_view_refinement import MultiViewConsistencyRefiner
from src.reconstruction.splatter import GaussianScene

def test_metrics_binary():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    m = SemanticMetrics.compute_binary_metrics(y_true, y_pred)
    assert m["tp"] == 2
    assert m["fn"] == 1
    assert m["fp"] == 0
    assert abs(m["iou"] - 2.0 / 3.0) < 1e-3
    assert abs(m["precision"] - 1.0) < 1e-3

def test_voxelize_point_cloud():
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [1.1, 1.1, 1.1]
    ])
    labels = np.array([1, 1, 2])
    voxels = SemanticMetrics.voxelize_point_cloud(points, labels, voxel_size=0.5)
    assert len(voxels) == 2
    assert voxels[(0, 0, 0)] == 1
    assert voxels[(2, 2, 2)] == 2

def test_camera_projection():
    K = np.eye(3) * 500.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0
    K[2, 2] = 1.0
    R = np.eye(3)
    t = np.array([0, 0, 0])

    cam = CameraView(view_id="test_cam", image_path="", image_size=(640, 480), K=K, R=R, t=t)
    pt3d = np.array([0.0, 0.0, 10.0])
    coords, depth = cam.project_point(pt3d)
    assert coords is not None
    assert abs(coords[0] - 320.0) < 1e-3
    assert abs(coords[1] - 240.0) < 1e-3
    assert abs(depth - 10.0) < 1e-3

def test_floater_pruning():
    positions = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [100.0, 100.0, 100.0]])
    colors = np.zeros((3, 3), dtype=np.uint8)
    opacities = np.array([[0.9], [0.9], [0.05]])
    scales = np.ones((3, 3))
    quats = np.zeros((3, 4))
    quats[:, 0] = 1.0
    labels = np.array([1, 1, 1])

    scene = GaussianScene(positions, colors, opacities, scales, quats, labels)
    pruned = scene.prune_floaters(min_opacity=0.2, k_neighbors=2)
    # The point with opacity 0.05 should be eliminated
    assert pruned.num_points <= 2

if __name__ == "__main__":
    test_metrics_binary()
    test_voxelize_point_cloud()
    test_camera_projection()
    test_floater_pruning()
    print("All unit tests passed successfully!")
