# WSI Cancer Detection Pipeline

End-to-end computational pathology pipeline for breast cancer detection on the PatchCamelyon (PCam / Camelyon16) dataset. Covers the full stack: patch-level CNN training, slide-level aggregation with attention MIL, Grad-CAM explainability, CycleGAN virtual staining, and LLM-generated clinical reports.

Built and trained on a local GTX 1650 (4 GB VRAM).

---

## Pipeline Overview

| Phase | What it does |
|-------|-------------|
| 1 | Dataset setup — HDF5 loading, WSI grouping from metadata CSVs |
| 2 | Preprocessing — Macenko stain normalization (pure NumPy), tissue detection (Otsu + HSV), albumentations augmentation |
| 3 | Patch CNN — EfficientNet-B3 fine-tuned on 50K patches with AMP, cosine LR warmup, early stopping |
| 4 | Attention MIL — Gated attention aggregation (Ilse et al., 2018) at slide level using real WSI groupings |
| 5 | Explainability — Grad-CAM heatmaps quantitatively evaluated against pixel-level tumor masks (IoU, Dice) |
| 6 | Virtual staining — CycleGAN trained on H&E patches for domain transfer |
| 7 | Clinical reports — LLM-generated pathology reports (Groq / Claude) for all 129 test WSIs |

---

## Results

### Patch Level (Phase 3 — EfficientNet-B3, 8K test patches)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-ROC | 0.9377 | [0.9327, 0.9427] |
| AUC-PR | 0.9432 | [0.9366, 0.9498] |
| F1 | 0.8277 | [0.8180, 0.8368] |
| Sensitivity | 0.7272 | [0.7134, 0.7406] |
| Specificity | 0.9700 | [0.9645, 0.9755] |
| Sens @ 95% Spec | 0.7765 | — |
| Cohen's Kappa | 0.6972 | — |
| MCC | 0.7187 | — |

Trained for 26 epochs. Pretrained ImageNet weights, AdamW, cosine schedule with 5-epoch linear warmup.

### Slide Level (Phase 4 — Gated Attention MIL, 129 test WSIs)

| Metric | Value |
|--------|-------|
| AUC-ROC | 1.0000 |
| F1 | 1.0000 |
| Sensitivity | ~1.000 |
| Specificity | ~1.000 |

49 TUMOR / 80 NORMAL in the test set. The near-perfect slide-level result is consistent with PCam — the patch CNN already achieves 0.94 AUC, so MIL aggregation over clean bags produces very confident predictions. Reported with bootstrap 95% CI.

### Explainability (Phase 5 — Grad-CAM vs pixel masks, 500 train patches)

| Metric | Value |
|--------|-------|
| Mean IoU | 0.4133 |
| Mean Dice | 0.5700 |

Grad-CAM activations evaluated against the ground-truth pixel-level tumor masks from the PCam training set — one of the few public histopathology datasets that includes pixel annotations.

---

## Stack

```
Python 3.10         PyTorch 2.5.1+cu121     CUDA 12.1
EfficientNet-B3     timm 0.9.x              albumentations
pytorch-grad-cam    CycleGAN (custom)       scikit-learn
Groq API            Anthropic Claude API    fpdf2
h5py                matplotlib              seaborn
```

---

## Dataset

**PatchCamelyon (PCam)** — 96×96 H&E patches extracted from Camelyon16 lymph node WSIs.

- 262,144 train / 32,768 val / 32,768 test patches (used 50K / 8K / 8K)
- 216 train / 54 val / 129 test WSIs (grouped via metadata CSVs)
- Pixel-level tumor masks available for training split

Download: https://github.com/basveeling/pcam

After downloading, update `config.yaml` paths to point to the HDF5 files.

---

## Model Weights

`mil_best.pth` (2.1 MB) is included in the repo. The two larger weights are excluded from version control due to size:

| File | Size |
|------|------|
| `efficientnet_b3_best.pth` | 124 MB |
| `cyclegan_best.pth` | 87 MB |

To reproduce them, run Phase 3 and Phase 6 respectively. All hyperparameters are in `config.yaml`.

---

## Setup

```bash
git clone https://github.com/AakankshaD835/WSI-Cancer-Detection
cd WSI-Cancer-Detection

pip install -r requirements.txt
```

For LLM reports, copy `.env.example` to `.env` and add an API key (Groq is free):

```bash
cp .env.example .env
# edit .env and add GROQ_API_KEY
```

---

## Run

```bash
# full pipeline end to end
python run_full_pipeline.py

# resume from a specific phase
python run_full_pipeline.py --from 4

# individual phases
python src/phase3_train.py
python src/phase4_mil.py
python src/phase5_explainability.py
python src/phase7_llm_report.py --live   # calls real LLM

# all visualizations
python src/visualize_all.py
```

All outputs (figures, metrics, training history) save to `reports/`.

---

## Outputs

`reports/` contains 28 figures covering every phase, full metrics JSONs, and training history CSVs. Sample clinical reports (LLM-generated pathology PDFs) are in `reports/clinical/samples/` — 3 representative slides: high-confidence tumor, normal, and borderline.

---

## Reference

> Ilse M., Tomczak J., Welling M. (2018). Attention-based Deep Multiple Instance Learning. *ICML 2018.*

> Macenko M. et al. (2009). A method for normalizing histology slides for quantitative analysis. *ISBI 2009.*

> Veeling B., Linmans J., Winkens J., Cohen T., Welling M. (2018). Rotation Equivariant CNNs for Digital Pathology. *MICCAI 2018.*
