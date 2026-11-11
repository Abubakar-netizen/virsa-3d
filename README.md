# Virsa3D: AI-Assisted 3D Reconstruction and Multi-View Consistent Semantic Segmentation of Heritage Sites

## 1. Project Background & Research Objective
**Virsa3D** is a research-and-development framework for photorealistic 3D Gaussian Splatting (3DGS) reconstruction and semantically consistent preservation of archaeological heritage structures using UAV drone imagery.

* **Baseline Foundation:** W. A. Fadilah, V. Gauthier, A. Murtiyoso, T. Landes, and P. Grussenmeyer, *“Automatic Segmentation of 3D Gaussian Splatting in Cultural Heritage Complex,”* International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. XLIX-B2-2026, pp. 825–832, 2026.
* **The Research Gap Identified:** In the baseline workflow, 2D segmentation masks (e.g. from SAM) are lifted directly into 3D Gaussian space. Inconsistencies, missed segments, and boundary ambiguities between overlapping multi-view drone images propagate directly into 3D, dropping the baseline 3D IoU to 28.8% (despite 75.9% in 2D).
* **Virsa3D Contribution:** Introduces a **Multi-View Semantic Consistency & Label Refinement Stage** between 2D segmentation and 3D Gaussian Splatting to identify cross-view conflicts and apply confidence-weighted consensus voting and spatial regularization before 3D reconstruction.

---

## 2. Pipeline Architecture

```
UAV Drone Images (.JPG)
        │
        ▼
[1. Image Preprocessing & EXIF] ───► GPS, Blur Score, Resolution Normalization
        │
        ▼
[2. 2D Semantic Segmentation]  ───► Per-Class Masks & RGBA (Ground, Structure, Vegetation)
        │
        ├────────────────────────────────────────┬────────────────────────────────────────┐
        │ [Baseline Workflow: Fadilah et al.]     │ [Proposed Workflow: Virsa3D]           │
        ▼                                        ▼                                        │
Direct Mask Lifting & Per-Class 3DGS      [3. Multi-View Semantic Consistency]            │
        │                                        │  - Epipolar projection & overlap check │
        │                                        │  - Cross-view agreement scoring        │
        │                                        ▼                                        │
        │                                 [4. Label Refinement & Regularization]          │
        │                                        │  - Consensus voting across views       │
        │                                        │  - Bilateral boundary smoothing        │
        │                                        ▼                                        │
        ▼                                 [5. Consistent 3DGS & Floater Pruning]          │
Outputs (outputs/baseline/)               Outputs (outputs/proposed/)                     │
        │                                        │                                        │
        └────────────────────────────────────────┴────────────────────────────────────────┘
                                                 │
                                                 ▼
                                  [6. Comparative Evaluation Engine]
                                   - Voxel 3D IoU (0.5m x 0.5m x 0.5m)
                                   - F1, Precision, Recall, Accuracy deltas
                                   - Markdown & JSON Thesis Report
```

---

## 3. Quick Start & Execution

### Prerequisites & Installation
* Python 3.10+
* Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
* (Optional for HPC execution) Set up your environment variables:
  ```bash
  cp .env.example .env
  # Configure your HPC_PASSWORD in .env
  ```

### Run Pipelines:
1. **Run Baseline Pipeline:**
   ```bash
   .venv\Scripts\python.exe scripts/run_baseline.py
   ```
2. **Run Proposed Pipeline:**
   ```bash
   .venv\Scripts\python.exe scripts/run_proposed.py
   ```
3. **Generate Side-by-Side Comparison Report:**
   ```bash
   .venv\Scripts\python.exe scripts/compare_results.py
   ```

