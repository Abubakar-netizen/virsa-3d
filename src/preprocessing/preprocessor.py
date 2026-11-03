import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ExifTags
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    UAV Imagery Preprocessing and Quality Assessment Engine.
    Handles EXIF extraction (GPS, focal length, altitude), Laplacian blur detection,
    exposure validation, and downscaled multi-resolution caching.
    """

    def __init__(
        self,
        raw_dir: str = "data",
        output_dir: str = "data/processed",
        target_max_dimension: int = 1920,
        thumbnail_dimension: int = 512,
        blur_threshold: float = 100.0,
        supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.thumbnails_dir = self.output_dir / "thumbnails"
        
        self.target_max_dimension = target_max_dimension
        self.thumbnail_dimension = thumbnail_dimension
        self.blur_threshold = blur_threshold
        self.supported_extensions = supported_extensions

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def _convert_dms_to_decimal(self, dms, ref: str) -> Optional[float]:
        """Convert Degrees, Minutes, Seconds tuple to decimal degrees."""
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return round(decimal, 7)
        except Exception:
            return None

    def extract_exif(self, image_path: Path) -> Dict[str, Any]:
        """Extract camera, GPS and capture settings from image EXIF."""
        metadata: Dict[str, Any] = {
            "make": "Unknown",
            "model": "Unknown",
            "focal_length_mm": None,
            "focal_length_35mm": None,
            "iso": None,
            "exposure_time": None,
            "gps": {
                "latitude": None,
                "longitude": None,
                "altitude_m": None
            }
        }

        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return metadata

                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make":
                        metadata["make"] = str(value).strip()
                    elif tag_name == "Model":
                        metadata["model"] = str(value).strip()
                    elif tag_name == "FocalLength":
                        metadata["focal_length_mm"] = float(value)
                    elif tag_name == "FocalLengthIn35mmFilm":
                        metadata["focal_length_35mm"] = float(value)
                    elif tag_name == "ISOSpeedRatings":
                        metadata["iso"] = int(value)
                    elif tag_name == "ExposureTime":
                        metadata["exposure_time"] = float(value)

                # Modern Pillow GPS IFD extraction
                try:
                    gps_ifd = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
                    if gps_ifd:
                        gps_tags = {}
                        for t, val in gps_ifd.items():
                            sub_tag = ExifTags.GPSTAGS.get(t, str(t))
                            gps_tags[sub_tag] = val

                        lat = gps_tags.get("GPSLatitude")
                        lat_ref = gps_tags.get("GPSLatitudeRef", "N")
                        lon = gps_tags.get("GPSLongitude")
                        lon_ref = gps_tags.get("GPSLongitudeRef", "E")
                        alt = gps_tags.get("GPSAltitude")

                        if lat:
                            metadata["gps"]["latitude"] = self._convert_dms_to_decimal(lat, lat_ref)
                        if lon:
                            metadata["gps"]["longitude"] = self._convert_dms_to_decimal(lon, lon_ref)
                        if alt is not None:
                            try:
                                metadata["gps"]["altitude_m"] = round(float(alt), 2)
                            except Exception:
                                pass
                except Exception as ex_gps:
                    logger.debug(f"GPS IFD extraction skipped: {ex_gps}")
        except Exception as e:
            logger.warning(f"Error reading EXIF from {image_path.name}: {e}")

        return metadata

    def assess_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Compute Laplacian variance (blur score) and mean brightness."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))

        is_sharp = laplacian_var >= self.blur_threshold
        return {
            "blur_laplacian_variance": round(laplacian_var, 2),
            "is_sharp": is_sharp,
            "mean_brightness": round(mean_brightness, 2),
            "quality_status": "acceptable" if is_sharp else "low_sharpness"
        }

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image: extract EXIF, evaluate blur, downscale and save cache."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        quality_info = self.assess_quality(img_bgr)
        exif_info = self.extract_exif(image_path)

        # Compute scaling factor for working resolution
        max_dim = max(orig_h, orig_w)
        scale = 1.0
        if max_dim > self.target_max_dimension:
            scale = self.target_max_dimension / float(max_dim)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_bgr = img_bgr
            new_w, new_h = orig_w, orig_h

        # Compute thumbnail
        thumb_scale = self.thumbnail_dimension / float(max(new_w, new_h))
        thumb_w = int(new_w * thumb_scale)
        thumb_h = int(new_h * thumb_scale)
        thumb_bgr = cv2.resize(resized_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Save processed working image and thumbnail
        stem = image_path.stem
        # Clean extension if duplicated like .JPG.jpeg
        clean_stem = stem.replace(".JPG", "").replace(".jpg", "")
        out_filename = f"{clean_stem}.jpg"
        
        proc_img_path = self.images_dir / out_filename
        thumb_img_path = self.thumbnails_dir / out_filename

        cv2.imwrite(str(proc_img_path), resized_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(thumb_img_path), thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return {
            "image_id": clean_stem,
            "original_filename": image_path.name,
            "original_dimensions": [orig_w, orig_h],
            "processed_dimensions": [new_w, new_h],
            "scale_factor": round(scale, 4),
            "processed_path": str(proc_img_path).replace("\\", "/"),
            "thumbnail_path": str(thumb_img_path).replace("\\", "/"),
            "quality": quality_info,
            "exif": exif_info
        }

    def process_all(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Scan raw directory, process all valid images, and save dataset_manifest.json."""
        all_files = sorted([
            p for p in self.raw_dir.iterdir()
            if p.suffix.lower() in self.supported_extensions
        ])

        if limit:
            all_files = all_files[:limit]

        logger.info(f"Processing {len(all_files)} images from {self.raw_dir}")
        manifest_items: List[Dict[str, Any]] = []

        for p in all_files:
            try:
                item_info = self.process_image(p)
                manifest_items.append(item_info)
                print(f"[PREPROCESS] Processed {p.name} -> {item_info['processed_dimensions']} (Blur score: {item_info['quality']['blur_laplacian_variance']})")
            except Exception as e:
                print(f"[PREPROCESS ERROR] Skipping {p.name}: {e}")

        manifest = {
            "total_images": len(manifest_items),
            "target_dimension": self.target_max_dimension,
            "images": manifest_items
        }

        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[PREPROCESS COMPLETE] Manifest written to {manifest_path}")
        return manifest
