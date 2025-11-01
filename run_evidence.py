#!/usr/bin/env python
# run_evidence.py
# Generates Task-A artifacts (tables + figures + mini-markdown summaries)
import os, argparse, pandas as pd
from tools.data_checks import load_csv, class_prevalence, pixel_stats, save_histograms, montage_random, write_markdown_summary as write_data_md
from tools.eval_summaries import threshold_sweep_from_model, plot_threshold_sweep, write_markdown_summary as write_eval_md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--weights", required=True, help="Trained checkpoint (e.g., outputs/baseline_cxr14/best.pt)")
    ap.add_argument("--outdir", default="outputs/taxonomy_evidence")
    ap.add_argument("--n-sample", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- DATA EVIDENCE ---
    df = load_csv(args.csv)
    prev = class_prevalence(df, args.labels)
    stats = pixel_stats(df, args.img_root, n_samples=args.n_sample, seed=1337)
    prev.to_csv(os.path.join(args.outdir, "prevalence.csv"), index=False)
    save_histograms(stats, args.outdir, title_prefix="brightness")
    montage_random(df.query("split=='train'") if "split" in df.columns else df, args.img_root,
                   os.path.join(args.outdir, "train_montage.png"), n=16, seed=1337)
    write_data_md(prev, stats, os.path.join(args.outdir, "01_data_evidence.md"), args.labels)

    # --- MODEL/EVAL EVIDENCE ---
    sweep_csv = os.path.join(args.outdir, "threshold_sweep.csv")
    df_sweep = threshold_sweep_from_model(args.csv, args.img_root, args.labels, args.weights,
                                          split="test", out_csv=sweep_csv)
    plot_threshold_sweep(df_sweep, os.path.join(args.outdir, "f1_threshold_sweep.png"))

    # assume you already ran: main.py eval -> outputs/clean_eval/metrics_test.json
    metrics_path = os.path.join("outputs", "clean_eval", "metrics_test.json")
    calib_path   = os.path.join("outputs", "calib", "calibration_summary.json")  # optional
    write_eval_md(metrics_path, calib_path, sweep_csv, os.path.join(args.outdir, "02_model_eval_evidence.md"))

    # index
    with open(os.path.join(args.outdir, "INDEX.md"), "w") as f:
        f.write("# Task A — Evidence bundle\n\n")
        f.write("- 01_data_evidence.md\n- 02_model_eval_evidence.md\n")
        f.write("- Tables: prevalence.csv, threshold_sweep.csv\n")
        f.write("- Figures: brightness_means.png, brightness_stds.png, train_montage.png, f1_threshold_sweep.png\n")

if __name__ == "__main__":
    main()
