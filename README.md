# Failure Analysis for Medical Image AI Models

A runnable baseline for **multi-label medical image classification** and failure analysis on chest X-rays. It includes:

* **Training** (DenseNet-121, ≤15 epochs)
* **Evaluation**: AUROC, AUPRC, F1@0.5, Sensitivity@95% specificity
* **Stress tests**: Gaussian noise/blur, JPEG artifacts, brightness–contrast (3 severities)
* **Slicing**: per-group metrics for any CSV column (e.g., `view`, `sex`, `site`, `num_findings`)
* **Calibration**: reliability diagram, ECE, temperature scaling (+ MC-Dropout, simple TTA variance)
* **Selective prediction**: risk–coverage curve
* **Validation CLI** to run the suite on a new CSV/folder

> The code is dataset-agnostic. You provide a CSV listing image paths and 2–4 binary label columns.

---

## Contents

* [Environment](#environment)
* [Dataset Setup (NIH ChestXray14)](#dataset-setup-nih-chestxray14)
* [CSV Format](#csv-format)
* [Quickstart](#quickstart)
* [Stress Test Severities](#stress-test-severities)
* [Core Commands](#core-commands)
* [Slicing (Domain Shift & Complexity)](#slicing-domain-shift--complexity)
* [Calibration & Selective Prediction](#calibration--selective-prediction)
* [Case Studies](#case-studies)
* [Outputs](#outputs)
* [Results Snapshot](#results-snapshot)
* [Repo Layout](#repo-layout)
* [Troubleshooting](#troubleshooting)
* [License & Disclaimer](#license--disclaimer)
* [Citations](#citations)

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

* Python 3.9
* PyTorch + Torchvision (ImageNet weights for DenseNet-121)
* Albumentations, scikit-learn, pandas, matplotlib, tabulate
* Reproducibility seed: **1337**
* GPU recommended (CPU works but is slower)

---

## Dataset Setup (NIH ChestXray14)

**Images (Hugging Face, Git-LFS):**

```bash
git lfs install
git clone https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset data/hf_cxr14
# images end up under: data/hf_cxr14/images/images/*.png
```

**Create the CSV used in this repo (3 labels):**

```bash
python make_cxr14_csv.py \
  --hf-root data/hf_cxr14 \
  --out data/chestxray14_three_labels.csv \
  --labels Cardiomegaly Edema Effusion
```

This writes `image_path`, `split` (train/val/test), and the three binary label columns.
Paths look like: `data/hf_cxr14/images/images/00000001_000.png`.

---

## CSV Format

`image_path` may be relative or absolute. `split` is optional (use `--auto-split` if absent).

```csv
image_path,split,Cardiomegaly,Edema,Effusion,view,sex,num_findings
data/hf_cxr14/images/images/00000001_000.png,train,0,1,0,PA,M,1
data/hf_cxr14/images/images/00000022_000.png,val,  0,0,1,PA,F,1
data/hf_cxr14/images/images/00012345_000.png,test,1,0,0,AP,M,2
```

To add splits automatically:

```bash
python main.py train \
  --csv data/your.csv --img-root . --labels Cardiomegaly Edema Effusion \
  --auto-split 0.7 0.1 0.2 --outdir outputs/baseline
```

---

## Quickstart

```bash
# Train baseline (10 epochs)
python main.py train \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/baseline_cxr14

# Evaluate on clean test set
python main.py eval \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion --split test \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/clean_eval

# Stress tests (3 severities × 4 corruptions)
python main.py stress \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion --split test \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/stress

# Calibration (temperature scaling) + reliability plots
python main.py calibrate \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion \
  --val-split val --test-split test \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/calib

# Selective prediction (risk–coverage curve)
python main.py selective \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion --split test \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/selective
```

---

## Stress Test Severities

| Corruption                | Sev-1 | Sev-2 | Sev-3 |
| ------------------------- | :---: | :---: | :---: |
| Gaussian noise (variance) |   10  |   25  |   50  |
| Gaussian blur (kernel)    |  3×3  |  5×5  |  9×9  |
| JPEG quality (q)          |   70  |   50  |   30  |
| Bright/Contrast (± limit) |  0.15 |  0.30 |  0.45 |

Matching the implementation in `main.py`.

---

## Core Commands

* **Training recipe**: DenseNet-121 (ImageNet init), final 3-way linear head; BCEWithLogits + class `pos_weight`; AdamW (1e-4); cosine LR; batch 16; input (224×224); dropout 0.2.
* **Best model**: `best.pt` (chosen by validation **macro AUROC**).

---

## Slicing (Domain Shift & Complexity)

Compute metrics for each subgroup by passing `--group-col`. Examples:

```bash
# View slice: PA vs AP
python main.py eval \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion --split test \
  --group-col view \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/slices_view

# Case complexity: 1 vs 2+ findings (assuming 'num_findings' column)
python main.py eval \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --labels Cardiomegaly Edema Effusion --split test \
  --group-col num_findings_bin \
  --weights outputs/baseline_cxr14/best.pt \
  --outdir outputs/slices_complexity
```

---

## Calibration & Selective Prediction

* **Temperature scaling** is fitted on the validation split and applied to logits at test time.
  Result in our runs: **ECE 0.101 → 0.025**; AUROC/AUPRC unchanged.
* **Selective prediction** writes `risk_coverage_*.csv` (micro risk across coverage levels) to help set an abstention policy.

---

## Case Studies

Generate the figures used in the paper:

```bash
# Requires predictions CSV (created via evaluate/calibrate pipelines)
python make_case_studies.py \
  --csv data/chestxray14_three_labels.csv --img-root . \
  --weights outputs/baseline_cxr14/best.pt \
  --preds outputs/case_studies/predictions_test.csv \
  --outdir outputs/case_studies
```

Outputs:

* `case_clean_pa.png` — clean PA with Effusion present (model positive at 0.5)
* `case_corrupted.png` — same study with severity-3 corruption (blur 9×9 or JPEG q=30), reduced confidence
* `case_ap_miss.png` — AP portable false negative (domain shift)

---

## Outputs

Each subcommand writes JSON/CSV + PNGs under `--outdir`:

* `best.pt` — checkpoint with best val macro AUROC
* `clean_eval/metrics_test.json` — clean macro/micro/per-class metrics
* `stress/*.json` and `stress_summary.csv` — corruption metrics and summary
* `calib/calibration_summary.json` + `reliability_*.png`
* `selective/risk_coverage_*.csv` + plot
* `group_metrics_*.json` — per-group slice metrics when `--group-col` is set
* `taxonomy_evidence/` — small dataset evidence (montage, brightness histograms, threshold sweeps) via `run_evidence.py`

---

## Results Snapshot

Clean test (macro, 3 labels):
**AUROC 0.896 · AUPRC 0.322 · F1@0.5 0.263 · Sens@95% 0.496**

Calibration (temperature scaling): **ECE 0.101 → 0.025** (AUROC/AUPRC unchanged).

Corruptions: performance degrades monotonically with severity; blur and JPEG are most harmful; brightness–contrast has milder impact. PA ≫ AP for sensitivity at the fixed operating point; multi-pathology cases are harder than single-finding.

---

## Repo Layout

```
main.py                    # train/eval/stress/calibrate/selective/valtool
make_cxr14_csv.py          # build CSV for NIH CXR14 (3 labels)
make_case_studies.py       # export case study images
run_evidence.py            # taxonomy/evidence figures + stats
tools/                     # helpers (data checks, etc.)
data/                      # CSVs (you add), image roots (you add)
outputs/                   # results written here by commands above
```

---

## Troubleshooting

* **Albumentations “size must be tuple”**
  Pass `(image_size, image_size)` to `RandomResizedCrop` (already fixed in `main.py`).

* **`tabulate` missing (for pandas `to_markdown`)**
  `pip install tabulate`.

* **Git-LFS smudge/partial files**
  `git lfs pull` inside `data/hf_cxr14`.

* **GaussianBlur creates “black” result**
  Don’t set `sigma_limit=0` with large kernels; use defaults (implemented).

* **Predictions CSV lacks metadata (e.g., `view`)**
  Join predictions with the original CSV on `image_path` when you need to filter by metadata.

---

## License & Disclaimer

Code in this repository is for research/education. **Not for clinical use.**
Include a permissive license (MIT/Apache-2.0) if you plan to share derivatives.

---

## Citations

If you use this starter, please cite the underlying datasets and libraries:

```bibtex
@inproceedings{Wang_2017_ChestXray8,
  title = {ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  author = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  booktitle = {CVPR},
  year = {2017}
}

@article{Huang_2017_DenseNet,
  title = {Densely Connected Convolutional Networks},
  author = {Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q.},
  journal = {CVPR},
  year = {2017}
}

@misc{Albumentations,
  title = {Albumentations: fast and flexible image augmentations},
  author = {Buslaev, Alexander et al.},
  year = {2020},
  howpublished = {\url{https://github.com/albumentations-team/albumentations}}
}

