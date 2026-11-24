"""Tahap 1: bersihkan dataset + split 70/15/15 stratified.

- buang duplikat persis (MD5)
- buang near-duplicate dalam kelas (dHash <= 2, konservatif)
- simpan manifest train/val/test ke output/_data/
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OUT  # noqa: E402

ROOT = Path(r"C:\Users\Wahyu Pratomo\Downloads\Indonesia-Medicinal-Plant-dataset\Indonesia-Medicinal-Plant-Dataset")
NEAR_DUP_MAX = 2


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def dhash(p, size=8):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


print("=== TAHAP 0-1: cek dataset + pembersihan ===", flush=True)
rows = []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() == ".jpg":
            rows.append({"path": str(f), "label": d.name})
df = pd.DataFrame(rows)
print(f"file terbaca: {len(df):,} | folder: {df.label.nunique()}", flush=True)

labels = sorted(df.label.unique())
idx = {c: i for i, c in enumerate(labels)}
df["y"] = df.label.map(idx)
(DATA / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")

# --- duplikat persis ---
print("hitung MD5...", flush=True)
df["md5"] = [md5(p) for p in df.path]
n0 = len(df)
df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
exact_removed = n0 - len(df)
print(f"duplikat persis dibuang: {exact_removed}", flush=True)

# --- near duplicate dalam kelas ---
print("hitung dHash...", flush=True)
df["hash"] = [dhash(p) for p in df.path]
keep = np.ones(len(df), dtype=bool)
by_class = defaultdict(list)
for i, lab in enumerate(df.label.values):
    by_class[lab].append(i)
near_removed = 0
for lab, ids in by_class.items():
    kept_idx = []
    for i in ids:
        dup = False
        for j in kept_idx:
            if int((df.hash.values[i] != df.hash.values[j]).sum()) <= NEAR_DUP_MAX:
                dup = True
                break
        if dup:
            keep[i] = False
            near_removed += 1
        else:
            kept_idx.append(i)
df = df[keep].drop(columns=["hash"]).reset_index(drop=True)
print(f"near-duplicate dibuang: {near_removed}", flush=True)

# --- split stratified ---
tr, rest = train_test_split(df, test_size=0.30, random_state=42, stratify=df.y)
va, te = train_test_split(rest, test_size=0.50, random_state=42, stratify=rest.y)
for name, part in (("train", tr), ("val", va), ("test", te)):
    part[["path", "label", "y"]].to_csv(DATA / f"{name}.csv", index=False)
    print(f"{name:5} {len(part):5,} gambar", flush=True)

report = pd.DataFrame({
    "item": ["file awal", "duplikat persis dibuang", "near-duplicate dibuang",
             "file bersih", "train", "val", "test", "kelas"],
    "nilai": [n0, exact_removed, near_removed, len(df), len(tr), len(va), len(te), len(labels)],
})
report.to_csv(DATA / "dataset_report.csv", index=False)
print(report.to_string(index=False))
print("DONE TAHAP 0-1", flush=True)
