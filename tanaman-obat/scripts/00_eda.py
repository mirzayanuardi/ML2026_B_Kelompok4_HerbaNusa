"""Tahap 0: EDA pada data MENTAH (sebelum pembersihan, sebelum preprocessing).

Tidak ada resize, normalisasi, atau augmentasi di sini. Semua angka dihitung
dari file gambar apa adanya (128x128, RGB).

Output: output/eda/figures/*.png + tabel ringkas di notebook.
"""
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, show_fig, tampil  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Wahyu Pratomo\Downloads\Indonesia-Medicinal-Plant-dataset\Indonesia-Medicinal-Plant-Dataset")
NEAR_DUP_MAX = 2

D = OUT / "eda"
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- baca folder mentah + hitung statistik tiap gambar ---
paths, kelas, md5s, lebar, tinggi, mode, bright, kontras, kb, hashes = [], [], [], [], [], [], [], [], [], []
for _d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for _f in sorted(_d.iterdir()):
        if not (_f.is_file() and _f.suffix.lower() == ".jpg"):
            continue
        _raw = _f.read_bytes()
        with Image.open(_f) as im:
            _w, _h = im.size
            _mode = im.mode
            _gray = np.asarray(im.convert("L"), dtype=np.uint8)
            _small = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        paths.append(str(_f)); kelas.append(_d.name)
        md5s.append(hashlib.md5(_raw).hexdigest())
        lebar.append(_w); tinggi.append(_h); mode.append(_mode)
        bright.append(float(_gray.mean())); kontras.append(float(_gray.std()))
        kb.append(len(_raw) / 1024)
        hashes.append((_small[:, 1:] > _small[:, :-1]).flatten())

df = pd.DataFrame({
    "path": paths, "label": kelas, "md5": md5s, "lebar": lebar, "tinggi": tinggi,
    "mode": mode, "brightness": bright, "contrast": kontras, "ukuran_kb": kb,
})
print(f"gambar mentah: {len(df):,} | kelas: {df.label.nunique()}", flush=True)

# --- cek duplikat (ditemukan di EDA, dibersihkan di tahap berikutnya) ---
mask = ~df.duplicated("md5").values
dup_persis = int((~mask).sum())
sisa = df[mask].reset_index(drop=True)
h = [hashes[k] for k in np.where(mask)[0]]
keep = np.ones(len(sisa), dtype=bool)
for _lab in sisa.label.unique():
    idxs = np.where(sisa.label.values == _lab)[0]
    kept = []
    for i in idxs:
        if any(int((h[i] != h[j]).sum()) <= NEAR_DUP_MAX for j in kept):
            keep[i] = False
        else:
            kept.append(i)
dup_mirip = int((~keep).sum())
print(f"duplikat persis: {dup_persis} | near-duplicate (setelah duplikat persis): {dup_mirip}", flush=True)

# --- tabel informasi dataset ---
tampil("Informasi dataset (mentah)", pd.DataFrame({
    "Item": ["Jumlah gambar", "Jumlah kelas", "Resolusi", "Mode warna",
             "Format file", "Ukuran file (rata-rata)", "Gambar per kelas"],
    "Nilai": [f"{len(df):,}", f"{df.label.nunique()}",
              f"{int(df.lebar.min())}x{int(df.tinggi.min())}",
              ", ".join(sorted(df['mode'].unique())), "JPG",
              f"{df.ukuran_kb.mean():.1f} KB (min {df.ukuran_kb.min():.1f}, maks {df.ukuran_kb.max():.1f})",
              f"{int(df.label.value_counts().min())}-{int(df.label.value_counts().max())}"],
}))

# --- cek missing dan duplikat ---
tampil("Pemeriksaan kualitas data", pd.DataFrame({
    "Pemeriksaan": ["Sel kosong", "File tidak ditemukan", "Duplikat persis (MD5)",
                    "Near-duplicate (dHash)"],
    "Hasil": [int(df[["path", "label"]].isna().sum().sum()),
              int(sum(not Path(p).exists() for p in df.path)), dup_persis, dup_mirip],
}))

# --- grafik distribusi kelas ---
jumlah = df.label.value_counts().sort_values()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(range(len(jumlah)), jumlah.values, color="#2e7d32", width=1.0)
ax[0].axhline(jumlah.mean(), color="#c62828", ls="--", lw=1,
              label=f"rata-rata {jumlah.mean():.1f}")
ax[0].set_title("Jumlah gambar per kelas (data mentah, urut naik)")
ax[0].set_xlabel("kelas (diurutkan)"); ax[0].set_ylabel("jumlah gambar")
ax[0].grid(axis="y", alpha=.3); ax[0].legend(fontsize=8)
ax[1].hist(jumlah.values, bins=12, color="#1565c0")
ax[1].set_title("Sebaran jumlah gambar antar kelas")
ax[1].set_xlabel("gambar per kelas"); ax[1].set_ylabel("jumlah kelas")
ax[1].grid(alpha=.3)
show_fig(fig, FIG / "distribusi_kelas.png", dpi=130)

# --- tabel sebaran kelas ---
tampil("Sebaran jumlah gambar per kelas", pd.DataFrame({
    "Keterangan": ["Jumlah kelas", "Total gambar", "Min per kelas", "Maks per kelas",
                   "Rata-rata per kelas"],
    "Nilai": [f"{df.label.nunique()}", f"{len(df):,}", int(jumlah.min()), int(jumlah.max()),
              f"{jumlah.mean():.1f}"],
}))

# --- grafik distribusi fitur gambar ---
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(df.brightness, bins=40, color="#1565c0")
ax[0].set_title("Brightness (0-255)"); ax[0].set_ylabel("jumlah gambar")
ax[1].hist(df.contrast, bins=40, color="#6a1b9a")
ax[1].set_title("Contrast (std piksel)")
ax[2].hist(df.ukuran_kb, bins=40, color="#e65100")
ax[2].set_title("Ukuran file (KB)")
for a in ax:
    a.grid(alpha=.3)
show_fig(fig, FIG / "distribusi_fitur.png", dpi=130)

# --- tabel statistik fitur ---
tampil("Statistik fitur gambar",
       df[["brightness", "contrast", "ukuran_kb"]].describe().round(2).T.reset_index()
       .rename(columns={"index": "Fitur"}))

# --- outlier ---
def hitung_outlier(kolom):
    q1, q3 = kolom.quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((kolom < lo) | (kolom > hi)).sum()), lo, hi
outlier = {k: hitung_outlier(df[k]) for k in ("brightness", "contrast", "ukuran_kb")}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, k in zip(ax, ("brightness", "contrast", "ukuran_kb")):
    a.boxplot(df[k])
    a.set_title(k)
show_fig(fig, FIG / "outlier.png", dpi=130)

# --- tabel outlier ---
tampil("Pemeriksaan outlier (aturan 1.5 x IQR)", pd.DataFrame({
    "Fitur": list(outlier),
    "Jumlah outlier": [v[0] for v in outlier.values()],
    "Persen": [f"{100 * v[0] / len(df):.1f}%" for v in outlier.values()],
    "Batas bawah": [round(v[1], 1) for v in outlier.values()],
    "Batas atas": [round(v[2], 1) for v in outlier.values()],
}))

# --- korelasi ---
corr = df[["brightness", "contrast", "ukuran_kb", "lebar", "tinggi"]].corr()
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Korelasi antar fitur gambar")
show_fig(fig, FIG / "korelasi_fitur.png", dpi=130)

# --- contoh gambar ---
rng = np.random.default_rng(7)
pilih = rng.choice(df.label.nunique(), 20, replace=False)
nama_kelas = sorted(df.label.unique())
fig, axes = plt.subplots(4, 5, figsize=(11, 8.6))
for a, k in zip(axes.ravel(), pilih):
    nama = nama_kelas[k]
    contoh = df[df.label == nama].sample(1, random_state=int(k) + 1).path.iloc[0]
    with Image.open(contoh) as im:
        a.imshow(im.convert("RGB"))
    a.set_title(nama[:30], fontsize=7)
    a.axis("off")
fig.suptitle("Contoh gambar dari 20 kelas berbeda", fontsize=11)
show_fig(fig, FIG / "contoh_gambar.png", dpi=100)

print("DONE EDA", flush=True)
