#!/usr/bin/env python
# -*- coding: utf-8 -*-
# make_case_studies.py  — plain images, no overlays

import os, cv2, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

LABELS = ["Cardiomegaly","Edema","Effusion"]

# ---------- IO helpers ----------
def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_png(rgb, path):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ---------- model / transforms ----------
def build_model(weights, device):
    m = densenet121(weights=DenseNet121_Weights.DEFAULT, drop_rate=0.2)
    m.classifier = nn.Linear(m.classifier.in_features, 3)
    m.load_state_dict(torch.load(weights, map_location=device), strict=True)
    return m.to(device).eval()

def tf_eval():
    return A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(224, 224),
        A.Normalize(), ToTensorV2()
    ])

@torch.no_grad()
def predict_single(model, img_rgb, device):
    x = tf_eval()(image=img_rgb)["image"].unsqueeze(0).to(device)
    p = torch.sigmoid(model(x)).cpu().numpy()[0]
    return p

# ---------- data plumbing ----------
def merge_view_and_labels(df_preds, main_csv):
    meta = pd.read_csv(main_csv)[["image_path", "view"] + LABELS]
    m = df_preds.merge(meta, on="image_path", how="left", suffixes=("", "_meta"))
    if m["view"].isna().any():
        # fallback merge by filename only (handles different prefixes)
        m["fn"] = m["image_path"].apply(os.path.basename)
        meta2 = meta.copy(); meta2["fn"] = meta2["image_path"].apply(os.path.basename)
        m = m.drop(columns=["view"]).merge(meta2[["fn", "view"]], on="fn", how="left").drop(columns=["fn"])
    # ensure label columns present
    for c in LABELS:
        if c not in m.columns and f"{c}_meta" in m.columns:
            m[c] = m[f"{c}_meta"]
    return m

# ---------- main ----------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # Load rows (reuse preds if provided; otherwise compute)
    if args.preds and os.path.exists(args.preds):
        rows = merge_view_and_labels(pd.read_csv(args.preds), args.csv)
        need_model_for_clean_probs = not all(f"prob_{c}" in rows.columns for c in LABELS)
    else:
        df = pd.read_csv(args.csv).query("split=='test'").copy()
        rows = df[["image_path", "view"] + LABELS].copy()
        need_model_for_clean_probs = True

    # Compute missing probabilities (one pass over test)
    if need_model_for_clean_probs:
        model = build_model(args.weights, device)
        probs = []
        for _, r in rows.iterrows():
            pth = r["image_path"]
            if not os.path.isabs(pth):
                pth = os.path.join(args.img_root, pth)
            probs.append(predict_single(model, read_img(pth), device))
        probs = np.vstack(probs)
        for j, lab in enumerate(LABELS):
            rows[f"prob_{lab}"] = probs[:, j]

    # --------- Case 1: AP false negative (Effusion) ---------
    has_view = "view" in rows.columns and rows["view"].notna().any()
    cand = rows[(rows["Effusion"] == 1) & (rows["prob_Effusion"] < 0.5)]
    if has_view:
        ap_cand = cand[cand["view"].astype(str).str.upper() == "AP"]
        if len(ap_cand) > 0:
            cand = ap_cand
    ap_row = cand.sort_values("prob_Effusion").iloc[0]
    ap_path = ap_row["image_path"]
    if not os.path.isabs(ap_path):
        ap_path = os.path.join(args.img_root, ap_path)
    ap_img = read_img(ap_path)
    write_png(ap_img, os.path.join(args.outdir, "case_ap_miss.png"))

    # --------- Case 2: clean PA TP that degrades ---------
    pool = rows[(rows["Effusion"] == 1) & (rows["prob_Effusion"] >= 0.90)]
    if has_view and len(pool) > 0:
        pool = pool[pool["view"].astype(str).str.upper() == "PA"] if len(pool) > 0 else pool
    if len(pool) == 0:
        pool = rows[rows["Effusion"] == 1].sort_values("prob_Effusion", ascending=False)
    row = pool.iloc[0]
    clean_path = row["image_path"]
    if not os.path.isabs(clean_path):
        clean_path = os.path.join(args.img_root, clean_path)
    clean_img = read_img(clean_path)
    write_png(clean_img, os.path.join(args.outdir, "case_clean_pa.png"))

    # corrupt same image (try blur 9x9, else JPEG q=30), no overlays
    model = build_model(args.weights, device)  # only needed for this single scoring
    # blur = A.Compose([A.GaussianBlur(blur_limit=(9, 9), sigma_limit=0, always_apply=True)])
    # bl_img = blur(image=clean_img)["image"]
    # bl_p = predict_single(model, bl_img, device)
    # cor_img = bl_img
    # if bl_p[2] >= 0.5:
    #     jpeg = A.Compose([A.ImageCompression(quality_lower=30, quality_upper=30, always_apply=True)])
    #     cor_img = jpeg(image=clean_img)["image"]
    # write_png(cor_img, os.path.join(args.outdir, "case_corrupted.png"))

    cor_blur = cv2.GaussianBlur(clean_img, (9, 9), 1.0)

    # (b) strong JPEG re-encode (q=30)
    ok, enc = cv2.imencode(
        ".jpg",
        cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 30],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    cor_jpeg = cv2.cvtColor(jpeg_bgr, cv2.COLOR_BGR2RGB)

    # Pick the corruption that hurts Effusion the most (ideally flips below 0.5)
    p_blur  = predict_single(model, cor_blur, device)
    p_jpeg  = predict_single(model, cor_jpeg, device)
    # Prefer the one with lower Effusion probability; fall back to blur on ties
    use_jpeg = p_jpeg[2] < p_blur[2]
    cor_img  = cor_jpeg if use_jpeg else cor_blur

    write_png(cor_img, os.path.join(args.outdir, "case_corrupted.png"))

    print("Wrote:",
          os.path.join(args.outdir, "case_clean_pa.png"),
          os.path.join(args.outdir, "case_corrupted.png"),
          os.path.join(args.outdir, "case_ap_miss.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="main dataset CSV (has view + labels)")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--preds", default=None, help="optional predictions_test.csv to reuse probs")
    args = ap.parse_args()
    main(args)
