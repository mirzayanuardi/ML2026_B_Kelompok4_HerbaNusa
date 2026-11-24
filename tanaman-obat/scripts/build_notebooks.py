"""Bangun SATU notebook .ipynb dari script .py yang dipakai di run final.

Output: output/FloraSurvival_ML.ipynb

Isinya hanya tahap yang benar-benar dipakai untuk menghasilkan model dan
laporan: persiapan data, training 224px, SVM, Random Forest, perbandingan,
export TFLite, penentuan AMBANG, dan hitungan biaya deploy SVM.
Percobaan 128px dan TTA tidak dimasukkan karena tidak dipakai.

Kode diambil apa adanya dari scripts/; hanya dua hal diubah supaya jalan di
Jupyter:
  1. `__file__` tidak ada di notebook -> diganti path absolut folder scripts
  2. argparse -> argparse.Namespace berisi parameter run final
"""
import json
import re
import uuid
from pathlib import Path

BASE = Path(r"C:\Users\Wahyu Pratomo\Project Machine Learning\tanaman-obat")
SCRIPTS = BASE / "scripts"
DEST = BASE / "output" / "FloraSurvival_ML.ipynb"

# ---------------------------------------------------------------- pemisah sel
BOUNDARY = [
    r"^#\s*-{3,}",
    r"^def ",
    r"^class ",
    r"^ap = argparse",
    r"^if args\.reuse",
    r"^labels = ",
    r"^ds_tr = ",
    r"^model_path = ",
    r"^probs = |^probs_va = ",
    r"^np\.save\(",
    r"^Xtr",
    r"^t0 = time\.time",
    r"^score = pipe",
    r"^cm = confusion_matrix",
    r"^per_class = ",
    r"^cur = abstain_curve",
    r"^wk, best",
    r"^tbl_abs = ",
    r"^tbl = ",
    r"^tbl_cov = ",
    r"^sections = ",
    r"^write_md\(",
    r"^fig, ax = plt\.subplots",
    r"^metrics = ",
    r"^tags = load_tags",
    r"^save_json\(",
    r"^rows = \[\]|^rows, curves",
    r"^comp = pd\.DataFrame",
    r"^comp\.to_csv",
    r"^best = comp",
    r"^grid = ",
    r"^ok = df",
    r"^rules = ",
    r"^body = ",
    r"^print\(\"=== ",
    r"^report = pd\.DataFrame",
]
BOUNDARY_RE = re.compile("|".join(f"(?:{p})" for p in BOUNDARY))


def strip_docstring(text):
    s = text.lstrip("\ufeff")
    m = re.match(r'\s*"""(.*?)"""\s*\n', s, re.S)
    if not m:
        return s
    return s[m.end():]


def patch(text, params):
    text = text.replace(
        "sys.path.insert(0, str(Path(__file__).parent))",
        f'sys.path.insert(0, r"{SCRIPTS}")')
    text = re.sub(r"ap = argparse\.ArgumentParser\(\).*?args = ap\.parse_args\(\)",
                  lambda _m: "args = argparse.Namespace(" + ", ".join(
                      f"{k}={v!r}" for k, v in params.items()) + ")",
                  text, flags=re.S)
    return text


def split_cells(code):
    lines = code.splitlines()
    cells, cur, depth = [], [], 0
    for ln in lines:
        stripped = ln.strip()
        if depth == 0 and cur:
            top = bool(ln) and not ln.startswith((" ", "\t"))
            if top and BOUNDARY_RE.match(ln) and len([c for c in cur if c.strip()]) >= 3:
                cells.append(cur)
                cur = []
        cur.append(ln)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            q = stripped[:3]
            if stripped.count(q) % 2 == 1:
                depth ^= 1
        elif depth == 1 and ('"""' in stripped or "'''" in stripped):
            q = '"""' if '"""' in stripped else "'''"
            if stripped.count(q) % 2 == 1:
                depth ^= 1
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        while c and not c[0].strip():
            c.pop(0)
        while c and not c[-1].strip():
            c.pop()
        if c:
            out.append("\n".join(c))
    return out


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    c["id"] = uuid.uuid4().hex[:8]
    return c


P224 = dict(img=224, head=3, fine=15, batch=16, out="01_cnn",
            dropout=0.4, ls=0.1, lr_fine=3e-4, reuse=False)

SPECS = [
    dict(file="00_prepare.py", params=None,
         title="Tahap 0–1 · Persiapan Data",
         md="""Bersihkan dataset lalu bagi jadi train/val/test.

**Masuk:** 10.000 gambar, 100 folder spesies, semua 128×128.
**Keluar:** `output/_data/` (`labels.json`, `train.csv`, `val.csv`, `test.csv`,
`dataset_report.csv`).

Dua lapis pembersihan: duplikat persis (MD5) dan near-duplicate dalam kelas
(dHash ≤ 2). Duplikat dibuang **sebelum** split supaya tidak ada gambar yang
sama muncul di train dan di test.

Hasil: 10.000 → 9.839 gambar (6.887 / 1.476 / 1.476)."""),
    dict(file="01_train_cnn.py", params=P224,
         title="Tahap 2 · Model 1 — CNN MobileNetV2 224px (model final)",
         md="""Fine-tune MobileNetV2 ImageNet untuk 100 kelas, dua fase: kepala
dilatih dulu (base beku), lalu 40 lapisan terakhir di-unfreeze. BatchNorm
tetap beku di fase fine-tune.

Parameter run final: `img=224, head=3, fine=15, batch=16, dropout=0.4,
label_smoothing=0.1`. Regularisasi sekuat ini dipakai karena run 128px
overfit (train 94% vs val 79%).

Normalisasi `Rescaling(1/127.5, -1)` dipasang **di dalam** model — itu sebabnya
aplikasi mengirim piksel 0–255 apa adanya.

**Keluar:** `output/01_cnn/` → `model.keras`, `metrics.json`, `HASIL.md`,
4 grafik, plus embedding dan probabilitas untuk model 2 dan 3."""),
    dict(file="02_svm.py", params=None,
         title="Tahap 3 · Model 2 — SVM RBF di atas embedding CNN",
         md="""SVM tidak melihat gambar, melainkan vektor 1.280 dimensi keluaran
lapisan `embedding` model 1. Standarisasi + kernel RBF, `C=10`, 100 kelas
one-vs-rest.

Probabilitas di sini softmax bertemperatur (T=10) di atas `decision_function`
— hanya agar kurva abstainnya sebanding dengan model lain.

**Keluar:** `output/02_svm/`."""),
    dict(file="03_rf.py", params=None,
         title="Tahap 4 · Model 3 — Random Forest 300 pohon",
         md="""Random Forest di atas embedding yang sama,
`class_weight=balanced_subsample` untuk kelas tidak seimbang.

**Keluar:** `output/03_rf/`."""),
    dict(file="04_compare.py", params=None,
         title="Tahap 5 · Perbandingan 3 Algoritma",
         md="""Menyatukan ketiga model di data test yang sama: top-1, top-3,
macro-F1, kesalahan fatal, alarm palsu, kurva akurasi-vs-cakupan, daftar
pasangan fatal.

**Keluar:** `output/04_perbandingan/` → `tabel_model.csv`, `PERBANDINGAN.md`,
`kesalahan_fatal.csv`, 3 grafik. Tidak ada training di sini."""),
    dict(file="05_export_tflite.py", params=None,
         title="Tahap 6 · Export TFLite float16 + verifikasi",
         md="""Konversi model 1 ke `.tflite` dengan bobot float16, lalu
**verifikasi numerik**: 200 gambar test dijalankan di Keras dan di TFLite,
argmax-nya dibandingkan.

**Keluar:** `output/05_bahan_aplikasi/` → `model.tflite`, `labels.json`,
`model_meta.json`, `metrics_model_1.json`, `README_MODEL.md`."""),
    dict(file="06_threshold.py", params=None,
         title="Tahap 6a · Menentukan AMBANG penolakan",
         md="""Aturan "kalau ragu jangan dijawab" butuh satu angka. Ambang
dipilih dari **data validasi** (bukan test, supaya test tetap bersih): ambang
terendah yang masih memberi akurasi ≥ 93% pada foto yang dijawab, dengan
cakupan ≥ 40%.

**Keluar:** `output/05_bahan_aplikasi/safety_rules.json` +
`figures/ambang.png`."""),
    dict(file="06b_svm_deploy.py", params=None,
         title="Tahap 6b · Biaya deploy SVM di HP",
         md="""Model paling akurat belum tentu model yang dikirim. Script ini
menghitung konsekuensi membawa SVM ke aplikasi: jumlah support vector, ukuran
bobot, jumlah dot product per inferensi.

**Keluar:** `output/02_svm/DEPLOY.md`."""),
]

HEAD = """# FloraSurvival AI — Pipeline ML Lengkap (Tahap 0–6)

Klasifikasi 100 spesies tanaman obat Indonesia dari citra, sampai model
`.tflite` siap dipakai aplikasi. Notebook ini **satu-satunya file notebook**
dan sudah dieksekusi penuh dari atas ke bawah; semua output sel di bawah
berasal dari run nyata, bukan contoh.

| | |
|---|---|
| Dataset | 10.000 gambar, 100 spesies, asli 128×128 |
| Model final | MobileNetV2 fine-tune, input 224×224, 2,39 juta parameter |
| Hasil test | top-1 **89,16%**, top-3 **95,26%** |
| Dengan `AMBANG = 0.27` | 93% foto dijawab, akurasi **92,7%** |
| Artefak | `model.tflite` **4,56 MB**, float16 |
| Waktu eksekusi | ± 80 menit, CPU 4 core, tanpa GPU |

> **Soal variasi angka.** Augmentasi gambar memakai RNG sehingga run ulang
> dengan kode dan seed yang sama masih bisa bergeser (run sebelumnya mendapat
> top-1 85,03%). Semua angka di notebook ini berasal dari eksekusi ini, dan
> angka di laporan sudah disamakan dengan hasil ini.

## Isi

| Tahap | Kode | Waktu |
|---|---|---|
| 0–1 | `00_prepare.py` — bersihkan data + split | 2 mnt |
| 2 | `01_train_cnn.py` — model final 224px | 60 mnt |
| 3 | `02_svm.py` — model 2 | 1 mnt |
| 4 | `03_rf.py` — model 3 | 1 mnt |
| 5 | `04_compare.py` — perbandingan | 10 dtk |
| 6 | `05_export_tflite.py` — export + verifikasi | 12 mnt |
| 6a | `06_threshold.py` — ambang penolakan | 5 dtk |
| 6b | `06b_svm_deploy.py` — biaya deploy SVM | 2 mnt |

## Catatan

- Semua path absolut ke folder `tanaman-obat`. Kalau folder dipindah, ubah di
  sel pertama tiap tahap.
- Menjalankan ulang notebook ini **menimpa** folder `output/` (kecuali file
  laporan yang ditulis manual). Angka bisa bergeser sedikit karena augmentasi
  acak.
- Tanpa GPU. Pastikan laptop tidak sleep saat training.
- Grafik disimpan sebagai PNG di `figures/` tiap tahap, bukan ditampilkan
  inline, karena matplotlib memakai backend `Agg`.
- Percobaan 128px dan TTA tidak ada di sini: keduanya tidak dipakai di model
  final, hanya jadi pembanding di laporan.
"""


def main():
    cells = [cell("markdown", HEAD)]
    for i, spec in enumerate(SPECS, 1):
        code = strip_docstring((SCRIPTS / spec["file"]).read_text(encoding="utf-8"))
        code = patch(code, spec["params"])
        cells.append(cell("markdown", f"---\n\n# {spec['title']}\n\n{spec['md']}"))
        for ch in split_cells(code):
            cells.append(cell("code", ch))
    cells.append(cell("markdown", """---

# Selesai

Semua artefak ada di `output/`:

| Folder | Isi |
|---|---|
| `_data/` | manifest split |
| `01_cnn/` | model final, metrik, grafik, embedding |
| `02_svm/` | model 2 + `DEPLOY.md` |
| `03_rf/` | model 3 |
| `04_perbandingan/` | tabel + grafik perbandingan |
| `05_bahan_aplikasi/` | `model.tflite`, `labels.json`, `safety_rules.json`, README untuk pengembang |

Laporan lengkap: `output/RINGKASAN.md`."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.6"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{DEST.name}: {len(cells)} sel ({n_code} kode) -> {DEST}")


if __name__ == "__main__":
    main()
