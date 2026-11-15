# Virsa3D: Official Stage 5 Quantitative Multi-Modal Evaluation Report

## 1. 2D Semantic Segmentation Evaluation (Meta SAM 2.1 on 20 Held-Out Frames)
| Semantic Class | Precision (%) | Recall (%) | F1-Score (%) | 2D IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Structure** | 95.12% | 92.60% | 93.85% | **88.42%** |
| **Ground** | 94.30% | 90.88% | 92.56% | **86.15%** |
| **Vegetation** | 91.55% | 87.89% | 89.68% | **81.30%** |
| **Macro Average** | **93.66%** | **90.46%** | **92.03%** | **85.29%** |

## 2. 3D Spatial Voxel Comparison (0.5m Spatial Grid against 9.76M Reference Cloud)
| Class | Reference Voxels | 3DGS Voxels | Precision (%) | Recall (%) | F1-Score (%) | Voxel IoU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Structure** | 15,511 | 9,921 | 99.65% | 63.74% | **77.74%** | **63.59%** |
| **Ground** | 12,912 | 9,188 | 99.75% | 70.98% | **82.94%** | **70.85%** |
| **Vegetation** | 2,917 | 2,762 | 98.99% | 93.73% | **96.28%** | **92.84%** |

## 3. Cloud-to-Cloud Geometric Distance (M3C2 via Lague et al., 2013)
- Mean Distance Error: **5.18 cm**
- Standard Deviation: **5.66 cm**
- Median Distance Error: **4.53 cm**
- Within Tolerance (+/- 6cm): **70.55% of Gaussians**
- Level of Detection (LoD_95%): **+/- 2.84 cm**

## 4. Photometric Rendering Fidelity (Held-Out Test Views)
- Structure Test PSNR: **23.85 dB** (Training: 25.12 dB)
- Structure Test SSIM: **0.862**
- Ground Test PSNR: **14.80 dB** (Training: 16.40 dB)
- Vegetation Test PSNR: **18.44 dB** (Training: 19.85 dB)
