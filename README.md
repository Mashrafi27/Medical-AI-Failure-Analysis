# CV8502 — Failure Analysis of Medical AI Systems (Starter)

This repo is a **coding starter** for the assignment. It gives you a runnable baseline for a **multi-label medical image classification** task with:
- Training (DenseNet121) for ≤15 epochs
- Evaluation with AUROC, AUPRC, sensitivity @95% specificity, F1
- **Stress tests** (noise/blur/JPEG/brightness-contrast) at 3 severities
- **Slicing** by any CSV metadata column (e.g., `site`)
- **Calibration** (reliability diagram, ECE, temperature scaling) and **uncertainty** (MC-Dropout, simple TTA variance)
- **Selective prediction** (risk–coverage curve)
- A small **validation CLI** for running the failure suite on a new folder/CSV

> It’s dataset-agnostic. Point it at a CSV that lists image paths and 2–4 binary pathology columns. See `data/dataset_template.csv` for the expected format.

---

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Put your CSV and images in place (see template file + README below)

# 3) Train baseline (example with 3 labels)
python main.py train   --csv data/your_labels.csv   --img-root /path/to/images   --labels Cardiomegaly Edema Effusion   --epochs 10 --batch-size 16 --lr 1e-4   --outdir outputs/baseline

# 4) Evaluate on clean test set
python main.py eval   --csv data/your_labels.csv --img-root /path/to/images   --labels Cardiomegaly Edema Effusion   --split test --weights outputs/baseline/best.pt --outdir outputs/clean_eval

# 5) Stress tests (3 severities × 4 corruptions)
python main.py stress   --csv data/your_labels.csv --img-root /path/to/images   --labels Cardiomegaly Edema Effusion   --split test --weights outputs/baseline/best.pt --outdir outputs/stress

# 6) Calibration + reliability
python main.py calibrate   --csv data/your_labels.csv --img-root /path/to/images   --labels Cardiomegaly Edema Effusion   --val-split val --test-split test   --weights outputs/baseline/best.pt --outdir outputs/calib

# 7) Selective prediction (risk–coverage)
python main.py selective   --csv data/your_labels.csv --img-root /path/to/images   --labels Cardiomegaly Edema Effusion   --split test --weights outputs/baseline/best.pt --outdir outputs/selective
```

> Repro: set `--seed 1337` on any command to fix randomness.

---

## CSV format

`image_path` is relative or absolute; `split` is optional (else use `--auto-split`). Each label column is 0/1.

```csv
image_path,split,Cardiomegaly,Edema,Effusion,site
patient0001.png,train,0,1,0,A
patient0002.png,train,1,0,0,A
patient1025.png,val,0,0,1,B
patient2042.png,test,1,1,0,B
```

If you don’t have a `split` column, add `--auto-split 0.7 0.1 0.2` during training; it will write back a new CSV with the splits added.

---

## Outputs

Each subcommand writes JSON/CSV metrics and figures (PNG) under `--outdir`. The best model checkpoint is `best.pt` (picked by mean AUROC on the validation split).

---

## Notes

- Keep runs light (≤15 epochs). Log wall-clock to report.
- Avoid PHI; ensure images are de-identified.
- Slicing: any CSV column can be used as `--group-col site` to report per-group metrics.
- For segmentation tasks (alternative to classification), you can still re-use the stress, calibration, and selective logic—swap the model/dataset. This starter focuses on classification to get you moving quickly.
