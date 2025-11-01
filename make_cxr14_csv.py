import os, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(".")
IMG_DIR  = ROOT/"data/hf_cxr14/images/images/"
META_DIR = ROOT/"data/hf_cxr14/data"
CSV_IN   = META_DIR/"Data_Entry_2017_v2020.csv"
CSV_OUT  = ROOT/"data/chestxray14_three_labels.csv"

labels = ["Cardiomegaly","Edema","Effusion"]

# ---------- Load metadata ----------
df = pd.read_csv(CSV_IN)
print("Loaded NIH metadata:", df.shape)

# keep only rows whose image exists
df = df[df["Image Index"].apply(lambda f: (IMG_DIR/f).exists())].copy()

# ---------- Create binary label columns ----------
def has(lbls, name): 
    parts = str(lbls).split("|")
    return 1 if name in parts else 0

for c in labels:
    df[c] = df["Finding Labels"].apply(lambda s, c=c: has(s, c))

# relative image paths
df["image_path"] = df["Image Index"].apply(lambda f: str(Path("data/hf_cxr14/images/images/")/f))

# ---------- Simple 70/10/20 patient-wise split ----------
rng = np.random.RandomState(1337)
pats = df["Patient ID"].unique()
rng.shuffle(pats)
n = len(pats)
tr = set(pats[:int(0.7*n)])
va = set(pats[int(0.7*n):int(0.8*n)])
te = set(pats[int(0.8*n):])

def assign(pid): 
    if pid in tr: return "train"
    elif pid in va: return "val"
    else: return "test"

df["split"] = df["Patient ID"].map(assign)

# ---------- Select columns and save ----------
cols = ["image_path","split"] + labels + ["Patient ID","Patient Age","Patient Gender","View Position"]
out = df[cols].rename(columns={
    "Patient ID":"patient_id",
    "Patient Age":"age",
    "Patient Gender":"sex",
    "View Position":"view"
})
out.to_csv(CSV_OUT, index=False)
print(f"✅ Wrote {CSV_OUT} with {len(out)} rows.")
