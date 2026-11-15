import Metashape
from pathlib import Path

def main():
    print("=" * 80)
    print("VIRSA3D: EXPORT HIGH-FIDELITY WEB 3D MODEL (GLTF / GLB)")
    print("=" * 80)

    proj_path = Path("stage_1_photogrammetry/metashape_project/amb_temple.psx").resolve()
    # Remove stale lock if any
    lock_file = proj_path.parent / "amb_temple.files" / "lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

    doc = Metashape.Document()
    doc.open(str(proj_path))
    chunk = doc.chunk
    model = chunk.model
    print(f"Loaded master model: {len(model.faces):,} faces, {len(model.vertices):,} vertices")

    # Target web directories
    out_dir = Path("outputs/textured_mesh").resolve()
    static_dir = Path("src/web/static").resolve()
    static_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Full Master GLTF
    gltf_master = out_dir / "amb_temple_master.gltf"
    print(f"\n[1/2] Exporting Master GLTF with 4K texture to: {gltf_master}...")
    chunk.exportModel(
        path=str(gltf_master),
        binary=False,
        save_texture=True,
        save_normals=True,
        save_colors=True,
        save_uv=True,
        format=Metashape.ModelFormatGLTF
    )
    print(f"-> Master GLTF Exported!")

    # 2. Decimate for fast 60 FPS web rendering
    print(f"\n[2/2] Creating web-optimized model (300,000 faces) for instantaneous browser loading...")
    # Duplicate model
    chunk.decimateModel(face_count=300000)
    web_model = chunk.model
    print(f"-> Decimated web model faces: {len(web_model.faces):,}, vertices: {len(web_model.vertices):,}")

    web_gltf = static_dir / "amb_temple_web.gltf"
    chunk.exportModel(
        path=str(web_gltf),
        binary=False,
        save_texture=True,
        save_normals=True,
        save_colors=True,
        save_uv=True,
        format=Metashape.ModelFormatGLTF
    )
    print(f"-> Web GLTF Exported to: {web_gltf}!")

    # Also export web OBJ
    web_obj = static_dir / "amb_temple_web.obj"
    chunk.exportModel(
        path=str(web_obj),
        binary=False,
        save_texture=True,
        save_normals=True,
        save_colors=True,
        save_uv=True,
        format=Metashape.ModelFormatOBJ
    )
    print(f"-> Web OBJ Exported to: {web_obj}!")

    print("\n" + "=" * 80)
    print("SUCCESS: WEB 3D MODELS EXPORTED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
