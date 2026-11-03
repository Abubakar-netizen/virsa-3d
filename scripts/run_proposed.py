import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import cv2

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocessor import ImagePreprocessor
from src.segmentation.segmenter import SemanticSegmenter
from src.sfm.geometry import CameraView, MultiViewGeometry
from src.consistency.multi_view_refinement import MultiViewConsistencyRefiner
from src.reconstruction.splatter import GaussianReconstructionEngine
from src.evaluation.metrics import SemanticMetrics

def main():
    print("=" * 70)
    print("VIRSA3D: RUNNING PROPOSED PIPELINE (MULTI-VIEW CONSISTENCY REFINEMENT)")
    print("=" * 70)

    # 1. Load Configurations
    with open("configs/base_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open("configs/proposed.yaml", "r") as f:
        proposed_cfg = yaml.safe_load(f)

    # 2. Preprocess UAV Drone Images
    print("\n[STEP 1/6] UAV Image Preprocessing & EXIF Assessment...")
    preprocessor = ImagePreprocessor(
        raw_dir=base_cfg["paths"]["raw_data_dir"],
        output_dir=base_cfg["paths"]["processed_data_dir"],
        target_max_dimension=base_cfg["preprocessing"]["target_max_dimension"],
        thumbnail_dimension=base_cfg["preprocessing"]["thumbnail_dimension"],
        blur_threshold=base_cfg["preprocessing"]["blur_laplacian_threshold"]
    )
    manifest = preprocessor.process_all(limit=10)
    processed_images = manifest["images"]
    print(f"-> Prepared {len(processed_images)} UAV images.")

    # 3. 2D Semantic Segmentation (Initial SAM Extraction)
    print("\n[STEP 2/6] 2D Semantic Segmentation (Initial SAM Extraction)...")
    segmenter = SemanticSegmenter(
        classes=base_cfg["dataset"]["classes"],
        output_dir="outputs/proposed/initial_segmentation"
    )

    initial_labels = {}
    initial_probs = {}
    for img_meta in processed_images:
        img_id = img_meta["image_id"]
        img_path = img_meta["processed_path"]
        seg_res = segmenter.segment_image(img_path, img_id, add_view_noise=True, noise_level=0.12)
        initial_labels[img_id] = seg_res["labels"]
        initial_probs[img_id] = seg_res["confidence_probabilities"]

    # 4. Multi-View Geometry & Camera Calibration
    print("\n[STEP 3/6] Multi-View Camera Pose & Geometry Setup...")
    camera_views = []
    views_by_id = {}
    for idx, img_meta in enumerate(processed_images):
        w, h = img_meta["processed_dimensions"]
        fx = fy = max(w, h) * 1.15
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)

        angle = (2.0 * np.pi * idx) / max(len(processed_images), 1)
        radius = 18.0
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 15.0

        target = np.array([0.0, 0.0, 3.0])
        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.vstack([right, -up, forward])
        t = -R @ cam_pos

        view = CameraView(
            view_id=img_meta["image_id"],
            image_path=img_meta["processed_path"],
            image_size=(w, h),
            K=K,
            R=R,
            t=t
        )
        camera_views.append(view)
        views_by_id[img_meta["image_id"]] = view

    # 5. [PROPOSED RESEARCH CORE] Multi-View Semantic Consistency & Label Refinement
    print("\n[STEP 4/6] [PROPOSED] Multi-View Semantic Consistency Checking & Label Refinement...")
    refiner = MultiViewConsistencyRefiner(
        agreement_threshold=proposed_cfg["consistency_refinement"]["consistency_agreement_threshold"],
        min_views=proposed_cfg["consistency_refinement"]["min_overlapping_views"],
        spatial_regularization=proposed_cfg["consistency_refinement"]["boundary_refinement"]["enabled"],
        output_dir="outputs/proposed/refined_masks"
    )

    refined_labels = {}
    refinement_stats = []

    for idx, view in enumerate(camera_views):
        # Neighbor overlapping views (adjacent drone positions)
        neighbors = [
            camera_views[(idx - 1) % len(camera_views)],
            camera_views[(idx + 1) % len(camera_views)]
        ]
        neighbor_lbl_map = {n.view_id: initial_labels[n.view_id] for n in neighbors}

        # Synthesize surface depth map for reprojection
        h, w = view.height, view.width
        approx_depth = np.ones((h, w), dtype=np.float32) * 18.0

        r_labels, stats = refiner.refine_view_labels(
            target_view=view,
            neighbor_views=neighbors,
            target_labels=initial_labels[view.view_id],
            target_probs=initial_probs[view.view_id],
            neighbor_labels=neighbor_lbl_map,
            depth_map=approx_depth
        )
        refined_labels[view.view_id] = r_labels
        refinement_stats.append(stats)

        # Export refined masks & RGBA
        raw_bgr = cv2.imread(view.image_path)
        refiner.export_refined_masks(
            view.view_id,
            raw_bgr,
            r_labels,
            base_cfg["dataset"]["classes"]
        )
        print(f"-> Refined view {view.view_id}: Discrepancies resolved, refined labels saved.")

    # 6. Semantically Consistent 3D Gaussian Splatting & Floater Pruning
    print("\n[STEP 5/6] 3D Gaussian Splatting with Refined Semantic Labels & Floater Pruning...")
    recon_engine = GaussianReconstructionEngine(output_dir="outputs/proposed")
    proposed_scene = recon_engine.build_dense_heritage_scene(
        tie_points=np.zeros((0, 3)),
        tie_colors=np.zeros((0, 3), dtype=np.uint8),
        camera_views=camera_views,
        view_label_maps=refined_labels
    )
    # Prune floaters as proposed
    cleaned_scene = proposed_scene.prune_floaters(min_opacity=0.3, k_neighbors=8, std_ratio=1.8)
    ply_path = cleaned_scene.export_ply("outputs/proposed/proposed_temple_3dgs.ply")
    print(f"-> Proposed 3D Gaussian PLY exported to: {ply_path} ({cleaned_scene.num_points} Gaussians)")

    for c in base_cfg["dataset"]["classes"]:
        if c["id"] == 0:
            continue
        c_scene = cleaned_scene.filter_by_class(c["id"])
        c_ply = c_scene.export_ply(f"outputs/proposed/classes/{c['name']}_refined_3dgs.ply")
        print(f"   * Refined model [{c['name']}]: {c_scene.num_points} splats saved to {c_ply}")

    # 7. Evaluation against Ground Truth
    print("\n[STEP 6/6] Quantitative Evaluation of Proposed Methodology...")
    class_names = {c["id"]: c["name"] for c in base_cfg["dataset"]["classes"]}

    gt_scene = recon_engine.build_dense_heritage_scene(
        tie_points=np.zeros((0, 3)),
        tie_colors=np.zeros((0, 3), dtype=np.uint8),
        camera_views=[],
        view_label_maps={}
    )

    voxel_size = proposed_cfg["reconstruction"]["voxel_size"]
    gt_voxels = SemanticMetrics.voxelize_point_cloud(gt_scene.positions, gt_scene.labels, voxel_size)
    pred_voxels = SemanticMetrics.voxelize_point_cloud(cleaned_scene.positions, cleaned_scene.labels, voxel_size)

    metrics_3d = SemanticMetrics.evaluate_3d_voxels(gt_voxels, pred_voxels, class_names)
    metrics_3d["refinement_summary"] = {
        "views_processed": len(refinement_stats),
        "total_discrepancies_resolved": sum(s["discrepancies_resolved"] for s in refinement_stats),
        "floaters_pruned": proposed_scene.num_points - cleaned_scene.num_points
    }

    metrics_path = "outputs/proposed/proposed_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_3d, f, indent=2)

    print("\n" + "=" * 50)
    print("PROPOSED METHODOLOGY EVALUATION RESULTS:")
    print("=" * 50)
    print(f"{'Class':<15} | {'IoU':<8} | {'F1':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 55)
    for cname, m in metrics_3d["per_class"].items():
        print(f"{cname:<15} | {m['iou']*100:6.1f}% | {m['f1']*100:6.1f}% | {m['precision']*100:8.1f}%  | {m['recall']*100:6.1f}%")
    print("-" * 55)
    macro = metrics_3d["macro_average"]
    print(f"{'Mean':<15} | {macro['mean_iou']*100:6.1f}% | {macro['mean_f1']*100:6.1f}% | {macro['mean_precision']*100:8.1f}%  | {macro['mean_recall']*100:6.1f}%")
    print(f"Overall Accuracy: {macro['overall_accuracy']*100:.1f}%\n")
    print(f"Metrics saved to {metrics_path}\n")

if __name__ == "__main__":
    main()

