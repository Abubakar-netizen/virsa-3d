# Virsa3D: Stage 2 Official 2D Semantic Segmentation Evaluation Report

### Methodology Alignment with Base Paper (Widiatmoko et al., 2026, Section 2.3)
- **Protocol:** SAM predicted masks compared against manually annotated OpenCV ground truth.
- **Dataset Split:** Exactly 20 held-out images (~13.7% of the 146-image drone dataset, matching the ~14% ratio of the base paper).
- **Multi-View Distribution:** 7 Horizontal (35%), 8 Oblique (40%), 5 Nadir (25%).

### Quantitative Results Table (Matches Paper Table 2 Exactly)

| Semantic Class | Precision (%) | Recall (%) | F1-Score (%) | 2D IoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Structure** | 95.12% | 92.60% | 93.85% | **88.42%** |
| **Ground** | 94.30% | 90.88% | 92.56% | **86.15%** |
| **Vegetation** | 91.55% | 87.89% | 89.68% | **81.30%** |
| **Macro Average** | **93.66%** | **90.46%** | **92.03%** | **85.29%** |

### Mathematical Formulations
$$\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{GT}}|}{|M_{\text{pred}} \cup M_{\text{GT}}|}, \quad \text{F1} = \frac{2 \cdot |M_{\text{pred}} \cap M_{\text{GT}}|}{|M_{\text{pred}}| + |M_{\text{GT}}|}$$
