import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Virsa3D Digital Heritage API",
    description="Backend API for 3D Gaussian Splatting Visualizer and Archaeological Heritage Metadata",
    version="1.0.0"
)

# Enable CORS for local web development
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

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Virsa3D Heritage Reconstruction Platform"}

@app.get("/api/site-info")
def get_site_info():
    """Returns archaeological site metadata, GPS coordinates, and historical context."""
    manifest_path = BASE_DIR / "data" / "processed" / "dataset_manifest.json"
    gps_info = {"latitude": 32.5084334, "longitude": 71.9363274, "altitude_min_m": 802.68, "altitude_max_m": 842.39}
    total_imgs = 72
    drone_model = "DJI FC8482"

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                total_imgs = data.get("total_images", total_imgs)
                if data.get("images"):
                    first_exif = data["images"][0].get("exif", {})
                    drone_model = f"{first_exif.get('make', 'DJI')} {first_exif.get('model', 'FC8482')}"
                    last_exif = data["images"][-1].get("exif", {})
                    gps_info["altitude_min_m"] = first_exif.get("gps", {}).get("altitude_m", 802.68)
                    gps_info["altitude_max_m"] = last_exif.get("gps", {}).get("altitude_m", 842.39)
        except Exception:
            pass

    return {
        "site_name": "Historical Hindu Shahi Archaeological Temple Complex",
        "region": "Salt Range, Punjab, Pakistan",
        "period": "c. 7th – 10th Century CE",
        "architectural_style": "Kashmir-Gandharan / Hindu Shahi Stone Masonry Architecture",
        "features": [
            "Curvilinear shikhara (superstructure tower)",
            "Trifoliate arched entrance with pedimented niches",
            "Dressed Kanja / sandstone ashlar masonry",
            "Elevated central sanctum with circumambulatory path"
        ],
        "gps": gps_info,
        "drone_survey": {
            "platform": drone_model,
            "total_images": total_imgs,
            "flight_type": "Oblique & Orbital Multi-View Cross-Angle",
            "normalized_resolution": "1920 x 1440 px"
        },
        "project_team": {
            "title": "Virsa3D: AI-Assisted 3D Reconstruction and Digital Preservation",
            "institution": "FAST-NUCES, Chiniot-Faisalabad Campus",
            "members": ["Abubakar Siddique (23F-3003)", "Mubashra Afzal (23F-3078)", "Zia Ur Rehman (23F-0547)"],
            "supervisor": "Mr. Umar Iqbal"
        }
    }

@app.get("/api/metrics/comparison")
def get_metrics_comparison():
    """Returns side-by-side evaluation metrics between Baseline and Proposed pipelines."""
    b_path = OUTPUTS_DIR / "baseline" / "baseline_metrics.json"
    p_path = OUTPUTS_DIR / "proposed" / "proposed_metrics.json"

    baseline_data = {}
    proposed_data = {}

    if b_path.exists():
        with open(b_path, "r") as f:
            baseline_data = json.load(f)
    if p_path.exists():
        with open(p_path, "r") as f:
            proposed_data = json.load(f)

    return {
        "baseline": baseline_data,
        "proposed": proposed_data,
        "classes": ["ground", "structure", "vegetation"],
        "voxel_size_m": 0.5
    }

@app.get("/api/models/{model_name}")
def get_model(model_name: str):
    """Serves binary PLY 3D models for rendering in browser."""
    candidates = {
        "baseline": [
            OUTPUTS_DIR / "baseline" / "models" / "baseline_scene.ply",
            OUTPUTS_DIR / "baseline" / "baseline_temple_3dgs.ply",
        ],
        "proposed": [
            OUTPUTS_DIR / "proposed" / "models" / "proposed_scene.ply",
            OUTPUTS_DIR / "proposed" / "proposed_temple_3dgs.ply",
        ],
        "structure": [
            OUTPUTS_DIR / "proposed" / "classes" / "structure_refined_3dgs.ply",
        ],
        "ground": [
            OUTPUTS_DIR / "proposed" / "classes" / "ground_refined_3dgs.ply",
        ],
        "vegetation": [
            OUTPUTS_DIR / "proposed" / "classes" / "vegetation_refined_3dgs.ply",
        ],
        "mesh": [
            OUTPUTS_DIR / "proposed" / "models" / "amb_temple_textured_mesh.obj",
        ],
        "mtl": [
            OUTPUTS_DIR / "proposed" / "models" / "amb_temple_textured_mesh.mtl",
        ],
        "texture": [
            OUTPUTS_DIR / "proposed" / "models" / "amb_temple_texture_atlas.png",
        ]
    }

    file_paths = candidates.get(model_name.lower())
    if not file_paths:
        raise HTTPException(status_code=404, detail=f"Model key '{model_name}' invalid")

    target_file = None
    for p in file_paths:
        if p.exists():
            target_file = p
            break

    if not target_file:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' file not found")

    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type="application/octet-stream"
    )

# Mount web frontend
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

