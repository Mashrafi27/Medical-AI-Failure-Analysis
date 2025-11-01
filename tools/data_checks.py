# tools/data_checks.py
import os, numpy as np, pandas as pd
from typing import List, Dict
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column.")
    return df

def class_prevalence(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """Overall + per-split prevalence (mean of 0/1)."""
    out = []
    if "split" in df.columns:
        groups = df.groupby("split")
    else:
        groups = [("all", df)]
    for split, d in groups:
        row = {"split": split}
        for c in labels:
            row[c] = d[c].mean()
        out.append(row)
    return pd.DataFrame(out)

def pixel_stats(df: pd.DataFrame, img_root: str, n_samples: int = 400, seed: int = 1337) -> Dict:
    """Brightness/contrast proxies across a sample: per-image mean & std (grayscale)."""
    rng = np.random.RandomState(seed)
    paths = df["image_path"].values
    idx = rng.choice(len(paths), size=min(n_samples, len(paths)), replace=False)
    means, stds = [], []
    for i in idx:
        p = paths[i]
        p = p if os.path.isabs(p) else os.path.join(img_root, p)
        try:
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        except Exception:
            continue
        means.append(arr.mean()); stds.append(arr.std())
    return {
        "count": len(means),
        "mean_of_means": float(np.mean(means)), "std_of_means": float(np.std(means)),
        "mean_of_stds": float(np.mean(stds)), "std_of_stds": float(np.std(stds)),
        "per_image_means": means, "per_image_stds": stds,
    }

def save_histograms(stats: Dict, outdir: str, title_prefix: str = "brightness"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(); plt.hist(stats["per_image_means"], bins=30)
    plt.xlabel("mean intensity"); plt.ylabel("count"); plt.title(f"{title_prefix}: per-image mean")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "brightness_means.png"), dpi=200); plt.close()
    plt.figure(); plt.hist(stats["per_image_stds"], bins=30)
    plt.xlabel("std intensity"); plt.ylabel("count"); plt.title(f"{title_prefix}: per-image std (contrast proxy)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "brightness_stds.png"), dpi=200); plt.close()

def montage_random(df: pd.DataFrame, img_root: str, out_path: str, n: int = 16, seed: int = 1337):
    """Grid of random images for visual evidence (under/over-exposure, markers, artifacts)."""
    from PIL import Image
    rng = np.random.RandomState(seed)
    sel = df.sample(n=min(n, len(df)), random_state=seed)
    rows = min(4, int(np.sqrt(n))); cols = max(1, int(np.ceil(n/rows)))
    canvas = Image.new("L", (256*cols, 256*rows))
    k = 0
    for _, r in sel.iterrows():
        p = r["image_path"]; p = p if os.path.isabs(p) else os.path.join(img_root, p)
        try:
            im = Image.open(p).convert("L").resize((256,256))
        except Exception:
            continue
        canvas.paste(im, ((k % cols)*256, (k // cols)*256)); k += 1
        if k >= rows*cols: break
    canvas.convert("RGB").save(out_path)

def write_markdown_summary(prevalence_df: pd.DataFrame, stats: Dict, out_md: str, labels: List[str]):
    lines = []
    lines.append("# Task A — Data Evidence\n")
    lines.append("## Class prevalence (fraction of positives)\n")
    lines.append(prevalence_df.to_markdown(index=False))
    lines.append("\n## Brightness/contrast proxies (random sample)\n")
    lines.append(f"- Images sampled: **{stats['count']}**")
    lines.append(f"- Mean of per-image means: **{stats['mean_of_means']:.3f}** (spread **{stats['std_of_means']:.3f}**)")
    lines.append(f"- Mean of per-image stds: **{stats['mean_of_stds']:.3f}** (spread **{stats['std_of_stds']:.3f}**)")
    lines.append("\nFigures saved: `brightness_means.png`, `brightness_stds.png`, `train_montage.png`")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
