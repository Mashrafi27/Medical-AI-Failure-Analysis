# tools/eval_summaries.py
import os, json, numpy as np, pandas as pd
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_metrics_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def threshold_sweep_from_model(csv_path: str, img_root: str, labels: List[str], weights: str,
                               image_size: int = 224, batch_size: int = 16, split: str = "test",
                               out_csv: str = None) -> pd.DataFrame:
    """
    Recomputes probabilities on the chosen split and evaluates micro-F1 across thresholds.
    Uses the dataset/model utilities from main.py (no duplication).
    """
    import sys, torch
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from main import MultiLabelImageDataset, build_model, predict, get_transforms, get_device
    from torch.utils.data import DataLoader

    device = get_device()
    tf = get_transforms(image_size, "val")
    ds = MultiLabelImageDataset(csv_path, img_root, labels, split=split, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=len(labels), drop_rate=0.2, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))

    logits, lab, _ = predict(model, dl, device)
    probs = 1/(1+np.exp(-logits))

    from sklearn.metrics import f1_score, roc_auc_score
    ths = np.round(np.linspace(0.1, 0.9, 17), 2)
    rows = []
    for t in ths:
        yhat = (probs >= t).astype(int)
        f1_micro = f1_score(lab.ravel(), yhat.ravel(), zero_division=0)
        try:
            auroc_micro = roc_auc_score(lab.ravel(), probs.ravel())
        except Exception:
            auroc_micro = float("nan")
        rows.append({"threshold": float(t), "f1_micro": float(f1_micro), "auroc_micro": float(auroc_micro)})
    df = pd.DataFrame(rows)
    if out_csv: df.to_csv(out_csv, index=False)
    return df

def plot_threshold_sweep(df: pd.DataFrame, out_png: str, title: str = "F1 vs Threshold"):
    plt.figure()
    plt.plot(df["threshold"], df["f1_micro"])
    plt.xlabel("threshold"); plt.ylabel("F1 (micro)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def write_markdown_summary(metrics_path: str, calib_summary_path: str, sweep_csv: str, out_md: str):
    m = load_metrics_json(metrics_path)
    calib = None
    if os.path.exists(calib_summary_path):
        try:
            calib = load_metrics_json(calib_summary_path)
        except Exception:
            calib = None

    lines = []
    lines.append("# Task A — Model & Evaluation Evidence\n")
    macro = m["metrics"]["macro"] if "metrics" in m else m["macro"]
    lines.append("## Clean test metrics (macro)\n")
    lines.append(f"- AUROC: **{macro['auroc']:.3f}**")
    lines.append(f"- AUPRC: **{macro['auprc']:.3f}**")
    lines.append(f"- F1@0.5: **{macro['f1@0.5']:.3f}**")
    lines.append(f"- Sensitivity@95%Spec: **{macro['sens@95spec']:.3f}**")

    if calib is not None:
        lines.append("\n## Calibration\n")
        lines.append(f"- ECE (uncalibrated): **{calib['baseline']['ece']:.3f}**")
        lines.append(f"- ECE (temperature-scaled, T={calib['temp_scaling']['T']:.2f}): **{calib['temp_scaling']['ece']:.3f}**")
        lines.append("Figures to include: `reliability_uncal.png`, `reliability_temp_scaled.png`.")

    if os.path.exists(sweep_csv):
        df = pd.read_csv(sweep_csv)
        best = df.iloc[df["f1_micro"].argmax()]
        lines.append("\n## Threshold sensitivity\n")
        lines.append(f"- Best F1_micro **{best['f1_micro']:.3f}** at threshold **{best['threshold']:.2f}**")
        lines.append("Figure: `f1_threshold_sweep.png`")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
