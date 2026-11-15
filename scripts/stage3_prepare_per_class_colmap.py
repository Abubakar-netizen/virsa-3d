"""
VIRSA3D: STAGE 3 — PER-CLASS DATA PREPARATION FOR 3D GAUSSIAN SPLATTING
Reference: Widiatmoko A. Fadilah et al., 2026 (Stage 3 Breakdown)

Generates independent per-class COLMAP workspaces:
  - colmap_workspaces/structure/
  - colmap_workspaces/ground/
  - colmap_workspaces/vegetation/

Each workspace contains:
  1. images/: Authentic RGBA PNGs (Alpha channel = class segmentation mask)
  2. sparse/0/: Exact camera poses (cameras.txt, images.txt, points3D.txt)
     matching Stage 1 Metashape photogrammetry.
"""

import os
import sys
import re
import time
import shutil
from pathlib import Path
import cv2
import numpy as np

def prepare_per_class_workspaces():
    print("=" * 80)
    print("VIRSA3D: STAGE 3 — PER-CLASS COLMAP WORKSPACE GENERATION")
    print("Classes: Structure (Class 2), Ground (Class 1), Vegetation (Class 3)")
    print("Target: 146 Registered UAV/Ground Cameras")
    print("=" * 80)

    # Source paths
    sparse_src = Path("stage_1_photogrammetry/aligned_cameras/sparse/0")
    if not sparse_src.exists():
        raise FileNotFoundError(f"Stage 1 COLMAP export not found at {sparse_src}")

    mask_dir = Path("data/processed/sam2_masks")
    data_dir = Path("data")

    # Read cameras.txt, images.txt, points3D.txt
    with open(sparse_src / "cameras.txt", "r") as f:
        cameras_content = f.read()

    with open(sparse_src / "images.txt", "r") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    with open(sparse_src / "points3D.txt", "r") as f:
        points3d_content = f.read()

    # Index raw data photos
    print("[1/4] Indexing raw input photos in data/...")
    data_photos = {}
    for p in data_dir.rglob("*.*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"] and "mask" not in p.name.lower():
            data_photos[p.stem.lower()] = p

    # Parse images.txt: header vs data lines
    header_lines = [l for l in raw_lines if l.startswith("#")]
    data_lines = [l for l in raw_lines if not l.startswith("#")]

    image_records = []
    # Every image consists of exactly TWO lines:
    # 1. Meta: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # 2. Points: X Y POINT3D_ID ...
    for i in range(0, len(data_lines), 2):
        meta_line = data_lines[i]
        points_line = data_lines[i + 1] if i + 1 < len(data_lines) else ""
        parts = meta_line.split()
        img_id = parts[0]
        qw, qx, qy, qz = parts[1:5]
        tx, ty, tz = parts[5:8]
        cam_id = parts[8]
        raw_name = " ".join(parts[9:])

        image_records.append({
            "id": img_id,
            "pose": [qw, qx, qy, qz, tx, ty, tz],
            "cam_id": cam_id,
            "name": raw_name,
            "points_line": points_line
        })

    print(f"-> Parsed exactly {len(image_records)} camera poses from images.txt.")

    # Match each image record to its raw photo and SAM2 mask
    print("[2/4] Mapping camera poses to raw photos and Meta SAM 2.1 masks...")
    matched_data = []
    for rec in image_records:
        raw_name = rec["name"]
        base = re.sub(r'_\d+\.jpg$', '', raw_name, flags=re.IGNORECASE)
        clean = re.sub(r'[\s\-_]*copy.*', '', base, flags=re.IGNORECASE)
        stem = Path(clean).stem

        # Match photo
        photo_path = data_photos.get(stem.lower())
        if not photo_path:
            photo_path = data_photos.get(Path(raw_name).stem.lower())

        # Match mask
        mask_path = mask_dir / f"{stem}_sam2_mask.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{Path(raw_name).stem}_sam2_mask.png"

        if not photo_path or not photo_path.exists():
            raise FileNotFoundError(f"Could not locate raw photo for camera {raw_name}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Could not locate SAM2 mask for camera {raw_name}")

        out_png_name = f"{Path(raw_name).stem}.png"
        matched_data.append({
            "rec": rec,
            "photo_path": photo_path,
            "mask_path": mask_path,
            "out_png_name": out_png_name
        })

    print(f"-> All {len(matched_data)} cameras successfully paired with photos and masks.")

    # Classes configuration: name -> mask_value
    classes = {
        "structure": 2,
        "ground": 1,
        "vegetation": 3
    }

    workspaces_root = Path("colmap_workspaces")
    workspaces_root.mkdir(parents=True, exist_ok=True)

    # Step 3: Setup directories and write per-class sparse/0 files
    print("\n[3/4] Initializing per-class COLMAP sparse/0 coordinate systems...")
    for cls_name in classes.keys():
        ws_dir = workspaces_root / cls_name
        ws_img = ws_dir / "images"
        ws_sparse = ws_dir / "sparse" / "0"
        ws_img.mkdir(parents=True, exist_ok=True)
        ws_sparse.mkdir(parents=True, exist_ok=True)

        # Write cameras.txt
        with open(ws_sparse / "cameras.txt", "w") as f:
            f.write(cameras_content)

        # Write points3D.txt
        with open(ws_sparse / "points3D.txt", "w") as f:
            f.write(points3d_content)

        # Write updated images.txt (referencing .png files)
        with open(ws_sparse / "images.txt", "w") as f:
            for h in header_lines:
                f.write(h + "\n")
            for item in matched_data:
                rec = item["rec"]
                pose_str = " ".join(rec["pose"])
                png_name = item["out_png_name"]
                f.write(f"{rec['id']} {pose_str} {rec['cam_id']} {png_name}\n")
                f.write(f"{rec['points_line']}\n")

        print(f"-> Workspace initialized: {ws_sparse} (cameras.txt, images.txt, points3D.txt)")

    # Step 4: Generate authentic RGBA PNG images
    print("\n[4/4] Generating authentic RGBA masked images per class...")
    total_imgs = len(matched_data)
    t0_start = time.time()

    for idx, item in enumerate(matched_data, start=1):
        t0 = time.time()
        # Read BGR photo
        bgr = cv2.imread(str(item["photo_path"]))
        h, w = bgr.shape[:2]

        # Read 2D semantic mask
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # Save per-class RGBA
        for cls_name, cls_val in classes.items():
            ws_img_dir = workspaces_root / cls_name / "images"
            out_path = ws_img_dir / item["out_png_name"]

            # Binary alpha channel: 255 where class matches, 0 elsewhere
            alpha = np.where(mask_resized == cls_val, 255, 0).astype(np.uint8)

            # Masked RGB: black where alpha is 0
            bgr_masked = bgr.copy()
            bgr_masked[alpha == 0] = 0

            # 4-channel BGRA
            bgra = np.dstack([bgr_masked, alpha])

            # Write PNG with fast compression level 1
            cv2.imwrite(str(out_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        dt = time.time() - t0
        if idx % 10 == 0 or idx == total_imgs:
            print(f"[{idx}/{total_imgs}] Rendered RGBA for {item['out_png_name']} ({w}x{h}) in {dt:.2f}s")

    total_time = time.time() - t0_start
    print("=" * 80)
    print(f"STAGE 3 PER-CLASS WORKSPACES SUCCESSFULLY CREATED in {total_time:.2f}s!")
    for cls_name in classes.keys():
        count = len(list((workspaces_root / cls_name / "images").glob("*.png")))
        print(f"-> {cls_name}: {count} RGBA images ready in colmap_workspaces/{cls_name}/")
    print("=" * 80)

if __name__ == "__main__":
    prepare_per_class_workspaces()
