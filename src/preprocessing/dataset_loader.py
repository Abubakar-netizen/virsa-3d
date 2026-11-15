import os
import sys
import glob
import json
import logging
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.preprocessing.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    't1_b': {'id': 1, 'name': 'Temple 1 Back', 'temple': 1, 'side': 'back'},
    't1_f': {'id': 2, 'name': 'Temple 1 Front', 'temple': 1, 'side': 'front'},
    't1_fb': {'id': 3, 'name': 'Temple 1 Front Bottom', 'temple': 1, 'side': 'front_bottom'},
    't1_ft': {'id': 4, 'name': 'Temple 1 Front Top', 'temple': 1, 'side': 'front_top'},
    't1_lb': {'id': 5, 'name': 'Temple 1 Left Bottom', 'temple': 1, 'side': 'left_bottom'},
    't1_lt': {'id': 6, 'name': 'Temple 1 Left Top', 'temple': 1, 'side': 'left_top'},
    't1_l':  {'id': 7, 'name': 'Temple 1 Left', 'temple': 1, 'side': 'left'},
    't1_r': {'id': 8, 'name': 'Temple 1 Right', 'temple': 1, 'side': 'right'},
    't1_rb': {'id': 9, 'name': 'Temple 1 Right Bottom', 'temple': 1, 'side': 'right_bottom'},
    't1_rt': {'id': 10, 'name': 'Temple 1 Right Top', 'temple': 1, 'side': 'right_top'},
    't1_tb': {'id': 11, 'name': 'Temple 1 Top Back', 'temple': 1, 'side': 'top_back'},
    't1_tl': {'id': 12, 'name': 'Temple 1 Top Left', 'temple': 1, 'side': 'top_left'},
    't2_top': {'id': 13, 'name': 'Temple 2 Top View', 'temple': 2, 'side': 'top'},
    't2_b': {'id': 14, 'name': 'Temple 2 Bottom/Back', 'temple': 2, 'side': 'bottom_back'},
    't2_f': {'id': 15, 'name': 'Temple 2 Front', 'temple': 2, 'side': 'front'},
    't2_l': {'id': 16, 'name': 'Temple 2 Left', 'temple': 2, 'side': 'left'},
    't2_r': {'id': 17, 'name': 'Temple 2 Right', 'temple': 2, 'side': 'right'},
}

class LabeledDatasetLoader:
    """
    Robust Loader and Stratified K-Fold Cross-Validation Partitioner
    for the Amb Temple Complex Labeled Drone & Ground Imagery.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.preprocessor = ImagePreprocessor(raw_dir=str(self.data_dir))
        self.items = []
        self._load_dataset()

    def _resolve_category(self, filename: str) -> Dict[str, Any]:
        fn = filename.lower().replace("-", "_")
        for prefix in sorted(CATEGORY_MAP.keys(), key=lambda x: -len(x)):
            if fn.startswith(prefix):
                return CATEGORY_MAP[prefix]
        return {'id': 0, 'name': 'Uncategorized', 'temple': 0, 'side': 'unknown'}

    def _load_dataset(self):
        exts = ('.jpg', '.jpeg', '.png')
        all_files = [f for f in self.data_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
        all_files.sort(key=lambda p: p.name)

        for p in all_files:
            cat = self._resolve_category(p.name)
            meta = self.preprocessor.extract_exif(p)
            self.items.append({
                "filename": p.name,
                "filepath": str(p),
                "category_id": cat["id"],
                "category_name": cat["name"],
                "temple": cat["temple"],
                "side": cat["side"],
                "size_bytes": p.stat().st_size,
                "exif": meta
            })

    def get_summary(self) -> Dict[str, Any]:
        counts = {}
        for it in self.items:
            c = it["category_name"]
            counts[c] = counts.get(c, 0) + 1
        return {
            "total_images": len(self.items),
            "categories": counts,
            "temple1_images": sum(1 for it in self.items if it["temple"] == 1),
            "temple2_images": sum(1 for it in self.items if it["temple"] == 2)
        }

    def generate_stratified_kfold(self, n_splits: int = 5, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate reproducible Stratified K-Fold partitions ensuring balanced representation
        of all temple facades across training and validation/test folds.
        """
        X = np.arange(len(self.items))
        # Group minor categories to ensure at least n_splits samples per group for sklearn
        y = np.array([it["category_id"] for it in self.items])
        
        # Merge singletons into broader temple facade groups for valid cross-validation
        y_stratified = []
        for it in self.items:
            cid = it["category_id"]
            if cid in [9, 10, 11, 12]: # rare t1 corners -> group into t1 right/top
                y_stratified.append(8)
            elif cid in [3, 4]: # t1 fb/ft -> group into t1 front
                y_stratified.append(2)
            elif cid in [13]: # t2 top (3 items) -> group into t2
                y_stratified.append(14)
            else:
                y_stratified.append(cid)
        y_stratified = np.array(y_stratified)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_stratified), 1):
            train_files = [self.items[i]["filename"] for i in train_idx]
            test_files = [self.items[i]["filename"] for i in test_idx]

            folds.append({
                "fold": fold_idx,
                "train_count": len(train_files),
                "test_count": len(test_files),
                "train_filenames": train_files,
                "test_filenames": test_files
            })

        return folds

if __name__ == "__main__":
    loader = LabeledDatasetLoader()
    summary = loader.get_summary()
    print("Dataset Summary:")
    print(json.dumps(summary, indent=2))

    folds = loader.generate_stratified_kfold(n_splits=5)
    print(f"\nGenerated {len(folds)}-Fold Stratified Cross-Validation Folds:")
    for f in folds:
        print(f"  Fold {f['fold']}: Train {f['train_count']} images | Test {f['test_count']} images")

    out_path = Path("data/processed/kfold_splits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump({"summary": summary, "folds": folds}, fp, indent=2)
    print(f"\nSaved K-Fold split definitions to {out_path}")
