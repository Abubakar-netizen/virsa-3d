import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("=" * 70)
    print("VIRSA3D: LAUNCHING INTERACTIVE 3D HERITAGE WEB PLATFORM")
    print("FastAPI Backend + Three.js 3D Gaussian Viewer")
    print("=" * 70)
    print("\n[INFO] Server running at: http://localhost:8000")
    print("[INFO] Press CTRL+C to stop the server.\n")

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
