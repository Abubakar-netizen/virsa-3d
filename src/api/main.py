"""
VIRSA3D: BACKEND REST API ENGINE (FASTAPI)
Digital Heritage Platform for Amb Hindu Shahi Temple Complex
Serves:
1. High-Fidelity 3D Textured Digital Twin (Development Portion)
2. 3D Gaussian Splatting & Photogrammetric Research Artifacts (Research Portion)
3. Structural Damage Detection & AI Conservation Metrology
4. Archaeological GIS and Camera Calibration Telemetry
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Virsa3D Digital Heritage & AI Conservation API",
    description="Backend engine for the Amb Hindu Shahi Temple 3D Digital Twin, 3DGS Semantic Visualizer, and Damage Metrology",
    version="2.0.0"
)

# Enable CORS for web development and external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
WEB_DIR = BASE_DIR / "src" / "web"
STATIC_DIR = WEB_DIR / "static"

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "Virsa3D Cultural Heritage Reconstruction & Conservation Platform",
        "version": "2.0.0",
        "monument": "Amb Hindu Shahi Temple Complex, Salt Range, Punjab, Pakistan"
    }

@app.get("/api/site-info")
def get_site_info():
    """Returns archaeological site metadata, GPS coordinates, and historical context."""
    return {
        "site_name": "Historical Amb Hindu Shahi Temple Complex",
        "local_name": "امب مندر (Amb Mandir)",
        "region": "Salt Range, Khushab District, Punjab, Pakistan",
        "period": "c. 7th – 9th Century CE (Hindu Shahi Dynasty)",
        "architectural_style": "Kashmir-Gandharan / Hindu Shahi Stone Ashlar Masonry",
        "monument_height_m": 13.5,
        "features": [
            "Curvilinear stone shikhara (superstructure tower) with fluted ornamental pilasters",
            "Trifoliate pedimented sanctum entrance with decorated lintels and side jambs",
            "Dressed Kanja limestone / sandstone ashlar dry-joint masonry",
            "Elevated central sanctum with circumambulatory plinth terrace",
            "Rectangular beam socket holes historically supporting wooden portico balconies"
        ],
        "gps": {
            "latitude": 32.5084334,
            "longitude": 71.9363274,
            "altitude_asl_m": 822.5,
            "coordinate_system": "WGS 84 / UTM Zone 43N"
        },
        "drone_survey": {
            "platform": "DJI FC8482 (Zenmuse 48.8 MP Sensor)",
            "total_images": 146,
            "calibrated_cameras": 146,
            "alignment_rate": "100.0%",
            "gsd_mm_per_pixel": 2.10,
            "rms_reprojection_error_px": 0.455,
            "trajectory_protocol": "3-Tier Cross-View (Horizontal Ground, 45° Oblique, Nadir Overhead)"
        },
        "project_team": {
            "title": "Virsa3D: Viewpoint-Aware Multi-View 3D Semantic Reconstruction & Gaussian Splatting for Cultural Heritage",
            "institution": "FAST-NUCES, Chiniot-Faisalabad Campus",
            "members": [
                {"name": "Abubakar Siddique", "roll": "23F-3003"},
                {"name": "Mubashra Afzal", "roll": "23F-3078"},
                {"name": "Zia Ur Rehman", "roll": "23F-0547"}
            ],
            "supervisor": "Mr. Umar Iqbal"
        }
    }

@app.get("/api/metrics/research")
def get_research_metrics():
    """Returns verified quantitative benchmarks matching the research paper 1:1."""
    return {
        "stage1_photogrammetry": {
            "total_cameras": 146,
            "aligned_cameras": 146,
            "sparse_tie_points": 266078,
            "dense_point_cloud": 9765311,
            "reprojection_error_rms": 0.455,
            "gsd_mm_per_pixel": 2.10
        },
        "stage1_1_classified_reference": {
            "total_points": 9765311,
            "classes": {
                "structure": {"code": 6, "points": 5528931, "percentage": 56.61},
                "ground": {"code": 2, "points": 2658007, "percentage": 27.22},
                "vegetation": {"code": 5, "points": 228056, "percentage": 2.34},
                "context": {"code": 0, "points": 1350317, "percentage": 13.83}
            }
        },
        "stage2_2d_segmentation": {
            "model": "Meta SAM 2.1",
            "test_split_frames": 20,
            "classes": {
                "structure": {"precision": 95.12, "recall": 92.60, "f1": 93.85, "iou": 88.42},
                "ground": {"precision": 94.30, "recall": 90.88, "f1": 92.56, "iou": 86.15},
                "vegetation": {"precision": 91.55, "recall": 87.89, "f1": 89.68, "iou": 81.30}
            },
            "macro_average": {"precision": 93.66, "recall": 90.46, "f1": 92.05, "iou": 85.29}
        },
        "stage4_3dgs_photometric": {
            "structure": {"gaussians": 60000, "train_psnr": 25.12, "test_psnr": 23.85, "ssim": 0.862},
            "ground": {"gaussians": 60000, "train_psnr": 16.40, "test_psnr": 14.80, "ssim": 0.741},
            "vegetation": {"gaussians": 60000, "train_psnr": 19.85, "test_psnr": 18.44, "ssim": 0.795}
        },
        "stage5_3d_spatial_voxels": {
            "voxel_grid_size_m": 0.5,
            "classes": {
                "structure": {"ref_voxels": 15511, "3dgs_voxels": 9921, "tp": 9886, "precision": 99.65, "recall": 63.74, "f1": 77.74, "iou": 63.59},
                "ground": {"ref_voxels": 12912, "3dgs_voxels": 9188, "tp": 9165, "precision": 99.75, "recall": 70.98, "f1": 82.94, "iou": 70.85},
                "vegetation": {"ref_voxels": 2917, "3dgs_voxels": 2762, "tp": 2734, "precision": 98.99, "recall": 93.73, "f1": 96.28, "iou": 92.84}
            },
            "macro_precision": 99.46,
            "macro_iou": 75.76
        },
        "stage5_m3c2_geometric_accuracy": {
            "mean_distance_cm": 5.18,
            "median_distance_cm": 4.54,
            "std_deviation_cm": 5.22,
            "within_tolerance_6cm_pct": 70.44,
            "lod_95_cm": 2.84
        },
        "viewpoint_ablation_novelty": {
            "config_a_nadir": {"cameras": 3, "coverage_pct": 77.28, "data_voids_pct": 22.72, "mean_views": 2.8},
            "config_b_nadir_oblique": {"cameras": 41, "coverage_pct": 99.72, "data_voids_pct": 0.28, "mean_views": 5.8},
            "config_c_virsa3d_full": {"cameras": 146, "coverage_pct": 100.00, "data_voids_pct": 0.00, "mean_views": 48.5}
        }
    }

@app.get("/api/damage/inspection")
def get_damage_inspection():
    """Returns architectural damage anomalies detected on the 3D temple mesh."""
    return {
        "site": "Amb Hindu Shahi Temple",
        "health_score": 78,
        "structural_status": "Stable with Localized Vulnerabilities",
        "conservation_priority": "Immediate Joint Stabilization & Lintel Grouting",
        "last_survey_date": "2026-11-15",
        "total_anomalies_logged": 5,
        "defects": [
            {
                "id": "DEF-01",
                "title": "Collapsed Lantern / Amalaka Summit",
                "category": "Structural Void / Superstructure Failure",
                "severity": "Critical",
                "location_description": "Shikhara Tower Apex (Z = +13.5m)",
                "coordinates": {"x": 0.0, "y": 0.2, "z": 6.8},
                "dimensions": "Diameter ~2.4m, Depth ~1.1m",
                "probable_cause": "Historical seismic shaking combined with missing tie timbers",
                "conservation_action": "Install weather-resistant titanium internal tie anchors and non-invasive protective cap"
            },
            {
                "id": "DEF-02",
                "title": "Fractured Portico Entrance Lintel",
                "category": "Masonry Shear Crack",
                "severity": "Critical",
                "location_description": "Main Sanctum Entrance Beam",
                "coordinates": {"x": -0.85, "y": -2.4, "z": 0.95},
                "dimensions": "Length ~1.45m, Width 3-8mm",
                "probable_cause": "Differential overburden load from upper pediment pedimentation",
                "conservation_action": "Hydraulic lime micro-grouting and external carbon-fiber rod reinforcement"
            },
            {
                "id": "DEF-03",
                "title": "Western Plinth Salt Sub-efflorescence & Spalling",
                "category": "Material Deterioration / Spalling",
                "severity": "Moderate",
                "location_description": "Basal Plinth Molding (Western Facade)",
                "coordinates": {"x": -2.8, "y": -0.5, "z": -1.8},
                "dimensions": "Surface area ~3.2 sq. meters, Relief loss 15-40mm",
                "probable_cause": "Capillary groundwater rise carrying Salt Range halite minerals",
                "conservation_action": "Cellulose poultice desalination followed by sacrificial lime rendering"
            },
            {
                "id": "DEF-04",
                "title": "North-East Facade Joint Leaching & Mortar Loss",
                "category": "Joint Erosion / Mortar Depletion",
                "severity": "Moderate",
                "location_description": "Upper Shikhara Horizontal Masonry Courses",
                "coordinates": {"x": 1.9, "y": 1.4, "z": 4.1},
                "dimensions": "Depth of open joints: 40-90mm",
                "probable_cause": "Intense seasonal monsoon rainwater washing through dry ashlar joints",
                "conservation_action": "Deep pointing with natural hydraulic lime (NHL 3.5) mortar"
            },
            {
                "id": "DEF-05",
                "title": "Bio-Crust & Lichen Colonization",
                "category": "Biological Growth",
                "severity": "Minor",
                "location_description": "Shaded Northern Pilaster Recesses",
                "coordinates": {"x": 0.4, "y": 2.6, "z": 2.2},
                "dimensions": "Area ~4.5 sq. meters",
                "probable_cause": "High relative humidity in shaded vertical flutes during winter",
                "conservation_action": "Biocidal wash (quaternary ammonium compound) with soft bristle brush"
            }
        ]
    }

@app.get("/api/models/{model_name}")
def get_model(model_name: str):
    """Serves 3D models for real-time web rendering."""
    model_name_clean = model_name.lower().replace("-", "_")

    model_mapping = {
        # 1. High-Fidelity 3D Textured Digital Twin (Development Portion)
        "real_web_glb": STATIC_DIR / "amb_temple_web.glb",
        "real_master_glb": STATIC_DIR / "amb_temple_textured_model.glb",
        "real_web_obj": STATIC_DIR / "amb_temple_web.obj",
        "real_web_mtl": STATIC_DIR / "amb_temple_web.mtl",
        "real_textured_obj": OUTPUTS_DIR / "textured_mesh" / "amb_temple_textured_model.obj",
        "real_textured_mtl": STATIC_DIR / "amb_temple_textured_model.mtl",
        "real_texture_jpg": STATIC_DIR / "amb_temple_textured_model.jpg",

        # 2. Research 3DGS Models (Research Portion)
        "research_3dgs_composite": STATIC_DIR / "amb_temple_research_3dgs.ply",
        "3dgs_structure": STATIC_DIR / "3dgs_structure.ply",
        "3dgs_ground": STATIC_DIR / "3dgs_ground.ply",
        "3dgs_vegetation": STATIC_DIR / "3dgs_vegetation.ply",
        
        # 3. Classified Reference Point Cloud (ASPRS Standards)
        "classified_reference_ply": BASE_DIR / "data" / "ground_truth" / "amb_temple_3d_reference_classified.ply",

        # Legacy fallbacks
        "proposed": STATIC_DIR / "amb_temple_web.glb",
        "baseline": OUTPUTS_DIR / "baseline" / "models" / "baseline_scene.ply"
    }

    target_file = model_mapping.get(model_name_clean)
    if not target_file or not target_file.exists():
        # Search static dir directly
        candidate = STATIC_DIR / model_name
        if candidate.exists():
            target_file = candidate
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found on server")

    # Determine media type
    media_types = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".ply": "application/octet-stream",
        ".jpg": "image/jpeg",
        ".png": "image/png"
    }
    mtype = media_types.get(target_file.suffix.lower(), "application/octet-stream")

    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type=mtype
    )

# Mount web frontend
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
