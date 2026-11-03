import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class SemanticMetrics:
    """
    Computes 2D and 3D segmentation metrics matching the formal definitions
    in Fadilah et al. (ISPRS 2026) and Betsas et al. (2025).
    Includes: IoU, F1-score (Dice), Precision, Recall, and Overall Accuracy.
    """

    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute binary segmentation metrics (IoU, F1, Precision, Recall).
        y_true, y_pred: boolean or binary 1D/2D arrays.
        """
        y_true = (y_true > 0).astype(bool)
        y_pred = (y_pred > 0).astype(bool)

        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        tn = np.logical_and(~y_true, ~y_pred).sum()

        intersection = tp
        union = tp + fp + fn

        iou = float(intersection / union) if union > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }

    @classmethod
    def evaluate_2d_multiclass(
        cls,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        class_names: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Evaluate 2D multiclass segmentation across all specified classes.
        Returns per-class metrics and macro mean.
        """
        per_class = {}
        ious = []
        f1s = []
        precisions = []
        recalls = []

        for class_id, name in class_names.items():
            if class_id == 0:  # Skip background in mean IoU calculation if desired
                continue
            gt_mask = (ground_truth == class_id)
            pred_mask = (prediction == class_id)
            metrics = cls.compute_binary_metrics(gt_mask, pred_mask)
            per_class[name] = metrics
            ious.append(metrics["iou"])
            f1s.append(metrics["f1"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

        mean_metrics = {
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "mean_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "mean_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0
        }

        return {
            "per_class": per_class,
            "macro_average": mean_metrics
        }

    @staticmethod
    def voxelize_point_cloud(
        points: np.ndarray,
        labels: np.ndarray,
        voxel_size: float = 0.5
    ) -> Dict[Tuple[int, int, int], int]:
        """
        Discretize 3D points into a voxel grid of size voxel_size (default 0.5m x 0.5m x 0.5m).
        Each voxel is assigned the majority label of the points it contains.
        """
        if len(points) == 0:
            return {}

        voxel_indices = np.floor(points / voxel_size).astype(int)
        voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}

        for idx, coord in enumerate(voxel_indices):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(int(labels[idx]))

        majority_voxels: Dict[Tuple[int, int, int], int] = {}
        for key, lbls in voxel_dict.items():
            counts = np.bincount(lbls)
            majority_voxels[key] = int(np.argmax(counts))

        return majority_voxels

    @classmethod
    def evaluate_3d_voxels(
        cls,
        gt_voxels: Dict[Tuple[int, int, int], int],
        pred_voxels: Dict[Tuple[int, int, int], int],
        class_names: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Evaluate 3D semantic segmentation on voxel grids matching Table 2 of Fadilah et al.
        """
        all_voxel_keys = set(gt_voxels.keys()).union(set(pred_voxels.keys()))
        if not all_voxel_keys:
            return {"per_class": {}, "macro_average": {}, "overall_accuracy": 0.0}

        sorted_keys = sorted(list(all_voxel_keys))
        gt_array = np.array([gt_voxels.get(k, 0) for k in sorted_keys])
        pred_array = np.array([pred_voxels.get(k, 0) for k in sorted_keys])

        total_correct = np.sum(gt_array == pred_array)
        overall_accuracy = float(total_correct / len(gt_array))

        per_class = {}
        ious = []
        f1s = []
        precisions = []
        recalls = []

        for class_id, name in class_names.items():
            if class_id == 0:
                continue
            gt_m = (gt_array == class_id)
            pred_m = (pred_array == class_id)
            m = cls.compute_binary_metrics(gt_m, pred_m)
            per_class[name] = m
            ious.append(m["iou"])
            f1s.append(m["f1"])
            precisions.append(m["precision"])
            recalls.append(m["recall"])

        macro_avg = {
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "mean_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "mean_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "overall_accuracy": round(overall_accuracy, 4)
        }

        return {
            "per_class": per_class,
            "macro_average": macro_avg,
            "total_evaluated_voxels": len(sorted_keys)
        }

