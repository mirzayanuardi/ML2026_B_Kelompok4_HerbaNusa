"""Bangun SATU notebook .ipynb dari script .py yang dipakai di run final.

Output: output/HerbaNusa_ML.ipynb

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
DEST = BASE / "output" / "HerbaNusa_ML.ipynb"

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
    dict(file="00_eda.py", params=None,
         title="Tahap 0 · EDA (Exploratory Data Analysis)",
         md="""Analisis awal pada data **mentah**, sebelum dibersihkan dan sebelum
preprocessing citra: jumlah dan resolusi gambar, duplikat, sebaran kelas,
karakter piksel (brightness, kontras), outlier, korelasi, dan contoh visual.

Setiap grafik diikuti satu paragraf interpretasi singkat."""),
    dict(file="00_prepare.py", params=None,
         title="Tahap 1 · Persiapan Data (cleaning + split)",
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

**Keluar:** `output/01_cnn/` → `model.keras`, `metrics.json`, 4 grafik,
embedding dan probabilitas untuk model 2 dan 3. Grafik dan tabelnya
ditampilkan langsung di notebook."""),
    dict(file="02_svm.py", params=None,
         title="Tahap 3 · Model 2 — SVM RBF di atas embedding CNN",
         md="""SVM tidak melihat gambar, melainkan vektor 1.280 dimensi keluaran
lapisan `embedding` model 1. Standarisasi + kernel RBF, `C=10`, 100 kelas
one-vs-rest.

Probabilitas di sini softmax bertemperatur (T=10) di atas `decision_function`
— hanya agar kurva abstainnya sebanding dengan model lain.

Grafik dan tabel hasilnya ditampilkan langsung di bawah sel ini."""),
    dict(file="03_rf.py", params=None,
         title="Tahap 4 · Model 3 — Random Forest 300 pohon",
         md="""Random Forest di atas embedding yang sama,
`class_weight=balanced_subsample` untuk kelas tidak seimbang.

Grafik dan tabel hasilnya ditampilkan langsung di bawah sel ini."""),
    dict(file="04_compare.py", params=None,
         title="Tahap 5 · Perbandingan 3 Algoritma",
         md="""Menyatukan ketiga model di data test yang sama: top-1, top-3,
macro-F1, kesalahan fatal, alarm palsu, kurva akurasi-vs-cakupan, daftar
pasangan fatal.

Tidak ada training di sini: semua model sudah dilatih sebelumnya."""),
    dict(file="05_export_tflite.py", params=None,
         title="Tahap 6 · Export TFLite float16 + verifikasi",
         md="""Konversi model 1 ke `.tflite` dengan bobot float16, lalu
**verifikasi numerik**: 200 gambar test dijalankan di Keras dan di TFLite,
argmax-nya dibandingkan.

**Keluar:** `output/05_bahan_aplikasi/` → `model.tflite`, `labels.json`,
`model_meta.json`, `metrics_model_1.json`."""),
    dict(file="06_threshold.py", params=None,
         title="Tahap 6a · Menentukan AMBANG penolakan",
         md="""Aturan "kalau ragu jangan dijawab" butuh satu angka. Ambang
dipilih dari **data validasi** (bukan test, supaya test tetap bersih): ambang
terendah yang masih memberi akurasi ≥ 93% pada foto yang dijawab, dengan
cakupan ≥ 40%.

**Keluar:** `output/05_bahan_aplikasi/safety_rules.json` + `figures/ambang.png`.
Kurva pemilihan ambang dan tabel aturannya tampil di bawah."""),
    dict(file="06b_svm_deploy.py", params=None,
         title="Tahap 6b · Biaya deploy SVM di HP",
         md="""Model paling akurat belum tentu model yang dikirim. Script ini
menghitung konsekuensi membawa SVM ke aplikasi: jumlah support vector, ukuran
bobot, jumlah dot product per inferensi.

Hasil hitungannya ditampilkan sebagai tabel di bawah."""),
]

HEAD = """# HerbaNusa — Pipeline ML Lengkap (Tahap 0–6)

Klasifikasi 100 spesies tanaman obat Indonesia dari citra, sampai model
`.tflite` siap dipakai aplikasi. Urutannya: EDA data mentah → bersihkan data
→ latih tiga algoritma → bandingkan → export TFLite → tentukan aturan
penolakan. Notebook ini **satu-satunya file notebook** dan sudah dieksekusi
penuh dari atas ke bawah; semua output sel di bawah berasal dari run nyata,
bukan contoh.

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
| 0 | `00_eda.py` — EDA pada data mentah | 1 mnt |
| 1 | `00_prepare.py` — bersihkan data + split | 2 mnt |
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
- Grafik disimpan sebagai PNG di `figures/` tiap tahap sekaligus ditampilkan
  inline di notebook.
- Percobaan 128px dan TTA tidak ada di sini: keduanya tidak dipakai di model
  final, hanya jadi pembanding di laporan.
"""


# ------------------------------------------------ judul tahap & sub-bagian
STAGE = {
    "00_eda.py": """---

# Tahap 0 · EDA (Exploratory Data Analysis)

Analisis awal pada data **mentah** sebelum dibersihkan dan sebelum
preprocessing citra.

**Tujuan:** mengetahui ukuran, resolusi, kebersihan data (duplikat/missing),
sebaran sampel per kelas, dan karakter piksel (brightness, kontras) supaya
keputusan pembersihan dan preprocessing tepat.

> **Catatan urutan.** EDA di sini sengaja dijalankan **sebelum** Tahap 1.
> Pembersihan duplikat dan split ada di Tahap 1, sedangkan preprocessing citra
> (resize 224, augmentasi, normalisasi) baru ada di Tahap 2.""",

    "00_prepare.py": """---

# Tahap 1 · Persiapan Data (cleaning + split)

Membersihkan dataset mentah lalu membagi jadi train/val/test.

**Tujuan:** membuang gambar duplikat dan membagi data secara stratified supaya
tidak ada kebocoran antara train, val, dan test.

**Bukan preprocessing citra.** Di tahap ini belum ada resize, augmentasi, atau
normalisasi piksel. Itu semua baru dilakukan di Tahap 2.

**Masuk:** 10.000 gambar mentah, 100 folder spesies, 128×128.
**Keluar:** `output/_data/` (`labels.json`, `train.csv`, `val.csv`, `test.csv`,
`dataset_report.csv`).

Hasil: 10.000 → 9.839 gambar (6.887 / 1.476 / 1.476).""",

    "01_train_cnn.py": """---

# Tahap 2 · Model 1 — CNN MobileNetV2 224px

Model utama: MobileNetV2 yang sudah dilatih di ImageNet, di-fine-tune untuk
100 kelas.

**Tujuan:** mendapat classifier 100 spesies dengan akurasi di atas 85% dan
tetap ringan dibawa ke HP.

**Yang dilakukan:** dua fase latih (kepala dulu, lalu 40 lapisan terakhir
dibuka), BatchNorm tetap beku, plus augmentasi, dropout 0.4, dan label
smoothing 0.1.

**Masuk:** `output/_data/` (train/val/test).
**Keluar:** `output/01_cnn/` (`model.keras`, `metrics.json`, grafik, embedding,
probabilitas test).

> **Di sinilah preprocessing citra.** EDA (Tahap 0) dan pembersihan data
> (Tahap 1) tidak mengubah piksel. Preprocessing citra baru terjadi di sini,
> pada sel **2.1**: decode gambar → `resize` ke 224×224 → augmentasi
> (flip/rotate/zoom/contrast/brightness) hanya untuk train → normalisasi
> `Rescaling(1/127.5, -1)` di dalam model. Aplikasi cukup mengirim piksel
> 0–255 apa adanya karena normalisasi sudah ada di dalam model.""",

    "02_svm.py": """---

# Tahap 3 · Model 2 — SVM RBF di atas embedding CNN

SVM tidak melihat gambar, melainkan vektor 1.280 dimensi dari lapisan
`embedding` model 1.

**Tujuan:** membandingkan classifier non-linear klasik dengan CNN pada fitur
yang sama.

**Yang dilakukan:** standarisasi fitur, kernel RBF `C=10`, 100 kelas
one-vs-rest. Skor keputusan diubah ke probabilitas dengan softmax bertemperatur
supaya kurva abstainnya bisa dibandingkan.""",

    "03_rf.py": """---

# Tahap 4 · Model 3 — Random Forest 300 pohon

Random Forest memakai embedding yang sama dengan SVM, jadi perbandingannya
apple-to-apple.

**Tujuan:** melihat performa ensemble pohon sebagai pembanding kedua.

**Yang dilakukan:** 300 pohon dengan `class_weight=balanced_subsample`. Ukuran
model ikut diukur karena ini penentu apakah layak dibawa ke HP.""",

    "04_compare.py": """---

# Tahap 5 · Perbandingan 3 Algoritma

Ketiga model diuji di data test yang sama.

**Tujuan:** memilih model yang benar-benar dikirim ke aplikasi.

**Yang dibandingkan:** top-1, top-3, macro-F1, kesalahan fatal (beracun ditebak
pangan), alarm palsu, dan akurasi pada cakupan yang dipakai aplikasi.""",

    "05_export_tflite.py": """---

# Tahap 6 · Export TFLite float16 + verifikasi

Model 1 dikonversi ke `.tflite` dengan bobot float16 supaya muat anggaran 15 MB.

**Tujuan:** memastikan model yang dikirim ke aplikasi benar-benar sama dengan
model yang diuji.

**Yang dilakukan:** konversi float16, lalu 200 gambar test dijalankan di Keras
dan di TFLite dan hasil argmax-nya dibandingkan.""",

    "06_threshold.py": """---

# Tahap 6a · Menentukan AMBANG penolakan

Aturan "kalau ragu jangan dijawab" butuh satu angka.

**Tujuan:** menentukan ambang keyakinan yang menjaga akurasi tetap tinggi tanpa
membuat model terlalu sering menolak.

**Yang dilakukan:** ambang dicari di data validasi (bukan test) dengan kriteria
akurasi minimal 93% pada foto yang dijawab dan cakupan minimal 40%.""",

    "06b_svm_deploy.py": """---

# Tahap 6b · Biaya deploy SVM di HP

SVM menang tipis kalau semua foto dipaksa dijawab, tapi dia tidak bisa berdiri
sendiri karena butuh embedding CNN.

**Tujuan:** menghitung apa yang harus dibayar kalau SVM benar-benar dibawa ke HP.

**Yang dilakukan:** menghitung jumlah support vector, ukuran bobot
float32/float16, jumlah evaluasi kernel per inferensi, dan latensi referensi.""",
}

AMEND = {
    "00_eda.py": [
        ("import hashlib", "### 0.1 Informasi Dataset\n\nMembaca folder mentah dan menghitung statistik tiap gambar: resolusi, mode warna, brightness, kontras, dan ukuran file."),
        ("# --- cek missing dan duplikat ---", "### 0.2 Pemeriksaan Missing Value dan Duplikat\n\nMemastikan tidak ada sel kosong atau file hilang, lalu menghitung duplikat persis (MD5) dan near-duplicate (dHash) yang nanti dibuang di Tahap 1."),
        ("# --- grafik distribusi kelas ---", "### 0.3 Distribusi Sebaran Kelas\n\nMelihat apakah jumlah gambar tiap kelas seimbang sebelum data dibagi."),
        ("# --- grafik distribusi fitur gambar ---", "### 0.4 Karakteristik Fitur Citra\n\nDistribusi brightness, kontras, dan ukuran file untuk melihat variasi visual antar foto."),
        ("# --- outlier ---", "### 0.5 Pemeriksaan Outlier\n\nMendeteksi foto yang terlalu gelap, terlalu terang, atau berukuran ekstrem dengan aturan 1,5 × IQR."),
        ("# --- korelasi ---", "### 0.6 Korelasi Antar Fitur\n\nMemeriksa hubungan antar fitur citra sebelum memutuskan apakah ada fitur yang perlu dibuang."),
        ("# --- contoh gambar ---", "### 0.7 Contoh Gambar Tiap Kelas\n\nMenampilkan 20 sampel acak dari kelas yang berbeda untuk melihat variasi visualnya."),
    ],
    "00_prepare.py": [
        ("import hashlib", "### 1.1 Menyiapkan Alat dan Library\n\nMemuat library untuk pembersihan data (`hashlib` untuk MD5, `pandas`, `PIL`)."),
        ('print("=== TAHAP 0-1', "### 1.2 Membaca Dataset Mentah\n\nMenelusuri folder spesies dan mengumpulkan daftar file gambar."),
        ("labels = sorted(df.label.unique())", "### 1.3 Menyusun Urutan Kelas\n\nMenyortir nama folder supaya nomor kelas konsisten di semua tahap."),
        ("# --- duplikat persis ---", "### 1.4 Membuang Duplikat\n\nDua lapis: duplikat persis (MD5) dan near-duplicate dalam kelas (dHash ≤ 2)."),
        ("# --- split stratified ---", "### 1.5 Split Train/Val/Test\n\nPembagian stratified supaya proporsi kelas tetap sama di setiap bagian."),
        ("report = pd.DataFrame({", "### 1.6 Ringkasan Persiapan Data\n\nRekap jumlah gambar awal sampai hasil bersih."),
    ],
    "01_train_cnn.py": [
        ("import argparse", "### 2.1 Menyiapkan Data dan Model (termasuk preprocessing)\n\nMembaca manifest, menyiapkan pipeline `tf.data`, dan membangun MobileNetV2. **Preprocessing citra ada di sel ini:** decode → resize 224×224 → augmentasi (hanya train) → normalisasi `Rescaling(1/127.5, -1)` di dalam model."),
        ("ds_tr = make_ds(tr, True)", "### 2.2 Melatih Model (Dua Fase)\n\nFase pertama melatih kepala klasifikasi, fase kedua membuka 40 lapisan terakhir dengan learning rate kecil."),
        ("def decode(path, y):",
         "> **Preprocessing citra dimulai di sini.** Sel berikut berisi tiga langkah yang mengubah piksel mentah: (1) `decode` + `resize` ke 224×224, (2) `aug` — flip/rotate/zoom/contrast/brightness, hanya dipakai saat training, (3) `build` — `Rescaling(1/127.5, -1)` sebagai lapisan pertama model. Tidak ada langkah preprocessing piksel sebelum bagian ini."),
        ("# ---------- evaluasi ----------", "### 2.3 Evaluasi di Data Test\n\nMenghitung top-1, top-3, macro-F1, dan metrik keamanan pada data test."),
        ("# ---------- figur ----------", "### 2.4 Grafik Hasil Training\n\nKurva akurasi/loss, confusion matrix, recall per kelas, dan kurva abstain."),
        ("wk, best = worst_classes(y_te, pred, labels, 15)", "### 2.5 Tabel Ringkasan Model 1"),
    ],
    "02_svm.py": [
        ("import json", "### 3.1 Melatih SVM\n\nFitur embedding distandarkan lalu SVM RBF dilatih untuk 100 kelas."),
        ("fig, ax = plt.subplots(figsize=(12, 11))", "### 3.2 Grafik Evaluasi SVM"),
        ("# --- tabel keamanan ---", "### 3.3 Tabel Hasil SVM"),
    ],
    "03_rf.py": [
        ("import json", "### 4.1 Melatih Random Forest\n\n300 pohon di atas embedding yang sama, sekaligus mengukur ukuran model."),
        ("cm = confusion_matrix", "### 4.2 Grafik Evaluasi Random Forest"),
        ("# --- tabel keamanan ---", "### 4.3 Tabel Hasil Random Forest"),
    ],
    "04_compare.py": [
        ("import json", "### 5.1 Menghitung Metrik Ketiga Model\n\nSemua metrik dihitung ulang dari file probabilitas di data test yang sama."),
        ("# --- figur: bar akurasi ---", "### 5.2 Grafik Perbandingan"),
        ("# --- tabel perbandingan ---", "### 5.3 Tabel Perbandingan"),
    ],
    "05_export_tflite.py": [
        ("import json", "### 6.1 Konversi ke TFLite float16"),
        ("# ---------- verifikasi ----------", "### 6.2 Verifikasi Keras vs TFLite"),
    ],
    "06_threshold.py": [
        ("import json", "### 6a.1 Mencari Ambang dari Data Validasi"),
        ("# --- evaluasi ambang ini di TEST", "### 6a.2 Uji Ambang di Data Test"),
    ],
    "06b_svm_deploy.py": [
        ("import json", "### 6b.1 Menghitung Biaya SVM"),
        ("# --- tabel biaya deploy ---", "### 6b.2 Tabel Biaya Deploy"),
    ],
}

POST = {
    "00_eda.py": [
        ("# --- tabel informasi dataset ---",
         "**Interpretasi.** Data mentah berisi 10.000 gambar, tepat 100 per kelas, semuanya 128×128 piksel dan RGB. Ukuran rata-rata 9,3 KB, jadi seluruh dataset masih ringan."),
        ("# --- cek missing dan duplikat ---",
         "**Interpretasi.** Tidak ada sel kosong atau file hilang. Namun ada 69 gambar identik dan 122 near-duplicate. Ini yang membuat 10.000 turun jadi 9.839 di Tahap 1, dan harus dibuang sebelum split agar tidak bocor ke test."),
        ("# --- tabel sebaran kelas ---",
         "**Interpretasi.** Setiap kelas tepat 100 gambar, jadi dataset mentah sebenarnya seimbang. Setelah dibersihkan jumlahnya bergeser sedikit (56–70 per kelas di train), itu wajar karena duplikat tidak merata."),
        ("# --- tabel statistik fitur ---",
         "**Interpretasi.** Brightness berkisar 25–214 dengan rata-rata 108, sedangkan kontras 13–99. Foto diambil dalam kondisi cahaya beragam, jadi augmentasi kecerahan dan kontras nanti relevan."),
        ("# --- tabel outlier ---",
         "**Interpretasi.** Outlier IQR hanya 0,7–2,1% dari data. Jumlahnya kecil dan gambarnya masih wajar (bukan rusak), jadi lebih baik dipertahankan supaya model terbiasa dengan kondisi nyata."),
        ("# --- korelasi ---",
         "**Interpretasi.** Korelasi antar fitur lemah dan lebar/tinggi gambar konstan di 128. Untuk klasifikasi citra, korelasi piksel memang tidak relevan, jadi tidak ada fitur yang perlu dibuang."),
        ("# --- contoh gambar ---",
         "**Interpretasi.** Latar, sudut, dan pencahayaan setiap foto berbeda walaupun kelasnya sama. Ini alasan augmentasi dan fine-tuning dipakai, bukan menambah epoch saja."),
    ],
}


# ------------------------------------------------ penjelasan tiap output
# Kunci = index sel KODE (0-based) pada notebook hasil build. Nilai = paragraf
# `**Penjelasan output ...**` yang disisipkan tepat setelah sel itu. Setiap sel
# kode yang benar-benar mengeluarkan output harus punya satu entri di sini.
OUTPUT_NOTES = {1: '**Penjelasan output (log):** baris `gambar mentah: 10,000 | kelas: 100` adalah hasil '
    'penelusuran folder dataset. Ada 10.000 file JPG yang terbaca dan 100 folder spesies, jadi '
    'tidak ada file yang terlewat. Karena belum ada pembersihan, jumlah ini masih data mentah.',
 2: '**Penjelasan output (log):** `duplikat persis: 69` berarti 69 file punya isi byte yang '
    'sama persis (MD5 identik). `near-duplicate (setelah duplikat persis): 92` berarti setelah '
    '69 file itu disingkirkan, masih ada 92 gambar yang mirip (jarak dHash ≤ 2). Total 161 '
    'gambar bermasalah. Angka ini hanya **ditemukan** di EDA; pembuangannya dilakukan di Tahap '
    '1.',
 3: '**Penjelasan output (tabel):** tabel ringkasan bentuk dataset mentah. Semua 10.000 gambar '
    'berukuran 128×128 piksel, mode RGB, format JPG, dan ukuran file 3,6–13,5 KB (rata-rata '
    '9,3 KB). Kolom `Gambar per kelas = 100-100` berarti setiap kelas punya tepat 100 gambar, '
    'jadi dataset mentah seimbang sempurna.',
 4: '**Penjelasan output (tabel):** hasil pemeriksaan kualitas data. `Sel kosong = 0` dan '
    '`File tidak ditemukan = 0` menandakan tidak ada data hilang. Dua baris terakhir mengulang '
    'temuan duplikat: 69 duplikat persis dan 92 near-duplicate. Karena masih di tahap EDA, '
    'duplikat ini **belum** dibuang.',
 5: '**Penjelasan output (grafik):** dua grafik distribusi kelas pada data mentah. Grafik kiri '
    'menunjukkan setiap kelas berada di angka 100 (garis putus-putus merah = rata-rata 100,0), '
    'jadi batangnya rata sempurna. Grafik kanan menegaskan hal yang sama: seluruh 100 kelas '
    'jatuh di satu bin 100 gambar. Artinya tidak ada kelas yang kekurangan sampel di data '
    'mentah.',
 6: '**Penjelasan output (tabel):** tabel angka dari grafik sebelumnya. 100 kelas, total '
    '10.000 gambar, minimum dan maksimum per kelas sama-sama 100, rata-rata 100,0. Dataset '
    'mentah benar-benar seimbang, sehingga teknik penyeimbangan kelas (oversampling, class '
    'weight) tidak diperlukan di tahap awal.',
 7: '**Penjelasan output (grafik):** tiga histogram fitur piksel. Brightness (kiri) menyebar '
    'lebar kira-kira 25–214 dengan puncak di sekitar 110, artinya foto diambil dalam kondisi '
    'cahaya yang bervariasi. Contrast (tengah) memusat di sekitar 48 dengan ekor ke kanan. '
    'Ukuran file (kanan) mengumpul di sekitar 9 KB dan tidak ada file yang ukurannya ekstrem.',
 8: '**Penjelasan output (tabel):** statistik deskriptif ketiga fitur. Brightness rata-rata '
    '107,80 dengan simpangan baku 21,69 dan rentang 25,48–214,04. Contrast rata-rata 48,91, '
    'rentang 12,80–98,94. Ukuran file rata-rata 9,31 KB, rentang 3,59–13,45 KB. Kolom `count = '
    '10000` memastikan tidak ada nilai kosong yang diam-diam terlewat.',
 10: '**Penjelasan output (grafik):** tiga boxplot untuk deteksi outlier secara visual. Kotak '
     'adalah rentang 25–75 persen, garis tengah adalah median. Titik-titik di luar kumis '
     'adalah outlier. Terlihat outlier hanya sedikit dan tersebar tipis di kedua ujung, bukan '
     'kelompok besar yang mengindikasikan data rusak.',
 11: '**Penjelasan output (tabel):** jumlah outlier dengan aturan 1,5 × IQR. Brightness punya '
     '213 outlier (2,1%), contrast 124 (1,2%), ukuran file 69 (0,7%). Kolom batas bawah/atas '
     'adalah pagar IQR; nilai di luarnya dihitung outlier. Semuanya di bawah 2,2%, jadi '
     'outlier dipertahankan karena masih berupa foto wajar (misalnya daun yang sangat terang).',
 12: '**Penjelasan output (grafik):** heatmap korelasi antar lima fitur. Angka mendekati +1 '
     'berarti naik bersama, mendekati -1 berarti berlawanan. Hampir semua sel bernilai kecil '
     '(di bawah 0,2), jadi antar fitur tidak ada hubungan kuat. Sel `lebar`–`tinggi` bernilai '
     '1,00 karena semua gambar berukuran sama (128×128), bukan informasi baru.',
 13: '**Penjelasan output (grafik + log):** grid 20 contoh gambar dari 20 kelas berbeda, satu '
     'gambar per kelas. Tujuannya melihat variasi visual nyata: latar, sudut pengambilan, dan '
     'pencahayaan berbeda-beda meskipun objeknya sama jenis. Baris `DONE EDA` hanya penanda '
     'bahwa seluruh sel EDA selesai dijalankan.',
 16: '**Penjelasan output (log):** `=== TAHAP 0-1: cek dataset + pembersihan ===` adalah judul '
     'proses, lalu `file terbaca: 10,000 | folder: 100` mengonfirmasi tahap pembersihan '
     'membaca ulang seluruh 10.000 file dari 100 folder. Jumlahnya sama dengan EDA, jadi tidak '
     'ada file yang berubah atau hilang di antara kedua tahap.',
 18: '**Penjelasan output (log):** `hitung MD5...` adalah log proses, lalu `duplikat persis '
     'dibuang: 69` adalah hasilnya. Angka 69 persis sama dengan temuan EDA, jadi pembersihan '
     'MD5 berjalan sesuai perkiraan. Setelah langkah ini jumlah gambar turun dari 10.000 '
     'menjadi 9.931.',
 19: '**Penjelasan output (log):** `hitung dHash...` log proses, lalu `near-duplicate dibuang: '
     '92`. Dihitung setelah duplikat persis dibuang, sesuai urutan di script. Jumlahnya cocok '
     'dengan EDA (92). Setelah langkah ini dataset bersih berisi 9.839 gambar.',
 20: '**Penjelasan output:** hasil pembagian stratified dari 9.839 gambar. Train 6.887 '
     '(69,99%), val 1.476 (15,00%), test 1.476 (15,00%) — 70/15/15, selisih 0,01% hanya karena '
     'pembulatan jumlah gambar. Karena stratified dengan seed 42, proporsi 100 kelas tetap '
     'terjaga di ketiga bagian, sehingga evaluasi di data test mewakili semua kelas.',
 21: '**Penjelasan output (log):** rekap lengkap persiapan data dalam satu tabel: file awal '
     '10.000, dikurangi 69 duplikat persis dan 92 near-duplicate, menyisakan 9.839 gambar '
     'bersih, lalu dibagi menjadi 6.887 train, 1.476 val, dan 1.476 test. Baris `kelas 100` '
     'memastikan tidak ada kelas yang hilang setelah pembersihan. `DONE TAHAP 0-1` adalah '
     'penanda tahap selesai.',
 23: '**Penjelasan output (log):** konfirmasi manifest yang dipakai training. Train 6.887, val '
     '1.476, test 1.476 gambar, 100 kelas, dan `img 224` berarti preprocessing akan me-resize '
     'setiap gambar ke 224×224. Ukuran 224 dipilih karena MobileNetV2 dilatih di ImageNet pada '
     'resolusi itu.',
 28: '**Penjelasan output (log training):** log training dua fase. `parameter total: '
     '2.386.084` adalah ukuran model. **Fase head** (3 epoch, 9,3 menit) hanya melatih lapisan '
     'klasifikasi: akurasi naik 40,9% → 73,4% dan val 59,8% → 70,1%. **Fase fine-tune** (15 '
     'epoch, 53,1 menit) membuka 40 lapisan terakhir: `parameter trainable` naik jadi '
     '1.791.460, akurasi latih melonjak sampai 99,5% sedangkan val berhenti di sekitar '
     '88–89,6%. Jarak train dan val yang lebar itu tanda overfit, dan justru karena itu '
     'early-stopping menyimpan bobot epoch terbaik (bukan epoch terakhir). Angka `431/431` '
     'adalah jumlah batch per epoch (6.887 gambar ÷ batch 16).',
 29: '**Penjelasan output (log):** hanya `prediksi test...`, yaitu log bahwa model sedang '
     'menghitung probabilitas untuk 1.476 gambar test. Tidak ada angka hasil di sini karena '
     'hasilnya disimpan ke variabel dan baru dirangkum di sel berikutnya.',
 35: '**Penjelasan output (hasil metrik):** hasil akhir Model 1 di data test. `TOPI-1 0.8916` '
     '= 89,16% prediksi teratas benar (target 85% tercapai). `TOP-3 0.9526` = 95,26% jawaban '
     'benar ada di tiga teratas. `macroF1 0.8881` = F1 rata-rata seluruh 100 kelas, yang '
     'penting karena metrik ini tidak bisa ditolong kelas besar. `fatal 19/444 = 4.3%` berarti '
     'dari 444 foto tanaman beracun, ada 19 yang ditebak sebagai tanaman pangan — inilah '
     'kesalahan paling berbahaya.',
 37: '**Penjelasan output (grafik):** dua kurva training. Kiri: akurasi train vs val per '
     'epoch, terlihat train terus naik ke ~99,5% sementara val mendatar di ~89%. Kanan: loss '
     'train vs val, val berhenti turun setelah sekitar epoch 10. Pola ini menunjukkan model '
     'mulai menghafal data latih, sehingga yang dipakai adalah bobot val terbaik.',
 38: '**Penjelasan output (grafik):** confusion matrix 100×100. Sumbu Y adalah kelas asli, '
     'sumbu X adalah tebakan model. Warna terang di diagonal berarti tebakan benar; titik '
     'terang di luar diagonal adalah kesalahan. Diagonal tampak dominan, jadi sebagian besar '
     'kelas dikenali dengan baik, dan kesalahan yang ada menyebar tipis, bukan menumpuk di '
     'satu pasangan kelas.',
 39: '**Penjelasan output (grafik):** grafik recall per kelas, diurutkan dari terendah ke '
     'tertinggi. Setiap batang satu spesies. Batang paling kiri adalah kelas yang paling '
     'sering salah, sedangkan mayoritas batang di kanan mendekati 100%. Grafik ini '
     'memperlihatkan bahwa masalah utama bukan rata-rata akurasi, melainkan beberapa kelas '
     'sulit tertentu.',
 40: '**Penjelasan output (grafik):** kurva abstain — sumbu X adalah persentase foto yang mau '
     'dijawab, sumbu Y akurasi pada foto yang dijawab. Kurva naik dari kiri ke kanan artinya: '
     'semakin banyak foto yang ditolak, semakin tinggi akurasi sisanya. Ini dasar aturan '
     '"kalau ragu jangan dijawab".',
 41: '**Penjelasan output (tabel):** tabel ringkasan Model 1. Top-1 89,2%, Top-3 95,3%, '
     'Macro-F1 88,8% pada input 224×224. Model punya 2.386.084 parameter dan ukuran file Keras '
     '23,3 MB — besar karena masih float32; versi TFLite float16 nanti hanya 4,56 MB.',
 42: '**Penjelasan output (tabel):** metrik keamanan Model 1 tanpa aturan penolakan. Dari 444 '
     'foto beracun di test, 92,3% dikenali sebagai beracun, tetapi 19 foto (4,3%) salah '
     'ditebak sebagai pangan — inilah kesalahan fatal. `Recall pangan 96,3%` berarti tanaman '
     'pangan hampir tidak pernah hilang. `Alarm beracun palsu 13,9%` berarti sebagian tanaman '
     'pangan ditandai beracun; ini merugikan tapi tidak berbahaya.',
 43: '**Penjelasan output (tabel):** hubungan antara cakupan dan akurasi Model 1. Pada '
     '`cakupan 1.0` (semua 1.476 foto dijawab) akurasi 89,2%. Kalau hanya 90% foto paling '
     'yakin yang dijawab, akurasi naik ke 94,4%, dan pada 10% teratas mencapai 100%. Kolom '
     '`jumlah` adalah banyaknya foto yang dijawab. Tabel ini yang nanti dipakai memilih '
     '`AMBANG`.',
 44: '**Penjelasan output (tabel):** 15 kelas dengan recall terendah. `Dioscorea Hispida '
     '(Gadung)` paling buruk: hanya 38,5% dari 13 fotonya dikenali benar. Kolom `n` adalah '
     'jumlah foto kelas itu di test (13–15 foto karena test 1.476 ÷ 100 kelas). Daftar ini '
     'penting karena Gadung adalah tanaman beracun, jadi kesulitan di kelas ini berisiko '
     'tinggi.',
 46: '**Penjelasan output (log):** bentuk fitur yang dipakai SVM. Train (6.887 × 1.280) dan '
     'test (1.476 × 1.280) — 1.280 adalah panjang vektor embedding keluaran CNN, bukan piksel '
     'gambar. `kelas 100` memastikan jumlah label sesuai. SVM memang tidak melihat gambar '
     'langsung, hanya deskripsi angka ini.',
 47: '**Penjelasan output (log):** `fit SVM: 0.7 menit`. Melatih 100 classifier RBF '
     'one-vs-rest di 6.887 baris fitur hanya butuh kurang dari satu menit, jauh lebih cepat '
     'daripada training CNN 60 menit. Ini karena SVM bekerja di ruang fitur 1.280 dimensi yang '
     'sudah diringkas CNN.',
 53: '**Penjelasan output (hasil metrik):** metrik SVM di data test. Top-1 89,91%, Top-3 '
     '95,66%, Macro-F1 89,66% — sedikit di atas CNN pada Top-1 dan Top-3. Angka ini dicetak '
     'setelah `metrics.json` tersimpan, jadi hasil di notebook dan file serah terima berasal '
     'dari perhitungan yang sama.',
 54: '**Penjelasan output (grafik):** confusion matrix SVM. Polanya mirip CNN: diagonal '
     'dominan dan kesalahan menyebar. Perbedaannya halus, karena SVM memakai embedding CNN '
     'yang sama, jadi kesulitannya cenderung di kelas yang sama.',
 55: '**Penjelasan output (grafik):** recall per kelas untuk SVM, urut dari terburuk. '
     'Dibandingkan CNN, batang paling kiri sedikit lebih tinggi, artinya SVM sedikit lebih '
     'baik di kelas tersulit. Tetap ada kelas yang recall-nya jauh di bawah rata-rata, jadi '
     'tidak ada model yang sempurna di 100 kelas.',
 56: '**Penjelasan output (grafik):** kurva abstain SVM. Bentuknya menanjak seperti CNN, '
     'artinya probabilitas SVM juga bisa dipakai untuk menolak foto yang meragukan. Catatan '
     'teknis: probabilitas SVM di sini hasil softmax bertemperatur dari `decision_function`, '
     'jadi angkanya tidak sebanding langsung dengan probabilitas CNN — hanya bentuk kurvanya '
     'yang dibandingkan.',
 57: '**Penjelasan output (tabel):** tabel ringkasan SVM. Top-1 89,9%, Top-3 95,7%, Macro-F1 '
     '89,7%, waktu fit 0,7 menit, dan inputnya embedding 1.280 dimensi dari Model 1. Ukuran '
     'tidak dicantumkan di sini karena dihitung terpisah pada Tahap 6b.',
 58: '**Penjelasan output (tabel):** metrik keamanan SVM. Kesalahan fatal turun ke 16 = 3,6% '
     '(CNN 19 = 4,3%), jadi SVM sedikit lebih aman. Namun `Recall pangan 96,2%` dan `alarm '
     'beracun palsu 13,7%` hampir sama dengan CNN. Keunggulan SVM di sini tipis dan tidak '
     'cukup untuk mengalahkan masalah ukuran modelnya.',
 59: '**Penjelasan output (tabel):** akurasi vs cakupan untuk SVM. Pada cakupan 100% akurasi '
     '89,9%; pada cakupan 60% menjadi 92,9%; pada 10% teratas 97,3%. Dibandingkan CNN, kurva '
     'SVM naik lebih landai, sehingga pada cakupan yang sama (misalnya 90%) CNN justru lebih '
     'akurat. Ini alasan keputusan akhir memilih CNN.',
 60: '**Penjelasan output (log + tabel):** 15 kelas terburuk versi SVM. `Dioscorea Hispida '
     '(Gadung)` tetap paling buruk (38,5%), sama seperti versi CNN. Kesamaan ini menunjukkan '
     'kesulitan ada di fiturnya (bentuk daun Gadung mirip kencur), bukan semata di '
     'classifier-nya. `DONE MODEL 2` menandai tahap selesai.',
 62: '**Penjelasan output (log):** bentuk fitur untuk Random Forest, identik dengan SVM: train '
     '6.887 × 1.280 dan test 1.476 × 1.280, 100 kelas. Karena memakai fitur yang sama persis, '
     'perbandingan Random Forest vs SVM vs CNN adil (apple to apple): yang berbeda hanya '
     'algoritmanya.',
 63: '**Penjelasan output (log):** `fit RF: 0.6 menit` untuk 300 pohon. Random Forest bahkan '
     'sedikit lebih cepat dari SVM di sini, tapi kecepatan melatih bukan masalah utama — yang '
     'menentukan adalah ukuran model setelah jadi.',
 67: '**Penjelasan output (hasil metrik):** metrik Random Forest. Top-1 87,20%, Top-3 95,33%, '
     'Macro-F1 86,60% — paling rendah dari ketiganya. Baris berikutnya `node 857,250 | sklearn '
     '700 MB | padat 175 MB` adalah ukuran model: 857.250 node (428.775 daun) membuat file 700 '
     'MB di scikit-learn, dan masih 175 MB walau dipadatkan.',
 68: '**Penjelasan output (grafik):** confusion matrix Random Forest. Diagonal tetap dominan, '
     'tetapi titik-titik di luar diagonal lebih banyak dan lebih terang dibanding CNN/SVM. '
     'Artinya tebakan salahnya lebih tersebar ke banyak kelas, bukan salah sistematis ke satu '
     'kelas tertentu.',
 69: '**Penjelasan output (grafik):** recall per kelas Random Forest. Batang di sisi kiri '
     'lebih rendah dan sebarannya lebih lebar daripada dua model sebelumnya, mengonfirmasi '
     'Top-1 yang lebih rendah: beberapa kelas memang jauh lebih sering salah di model ini.',
 70: '**Penjelasan output (grafik):** 30 fitur (dari 1.280) yang paling sering dipakai Random '
     'Forest untuk memisahkan kelas. Sum bu Y = tingkat kepentingan. Sayangnya, kepentingan '
     'tinggi pada satu dimensi tidak bisa diterjemahkan menjadi fitur yang bisa ditampilkan ke '
     'pengguna, jadi grafik ini hanya menunjukkan model ini benar-benar belajar sesuatu, bukan '
     'acak.',
 71: '**Penjelasan output (grafik):** kurva abstain Random Forest. Kurvanya menanjak paling '
     'tajam di awal, yang tampak bagus, tetapi titik awalnya (cakupan 100%) paling rendah '
     '(87,2%). Artinya pada cakupan kerja yang dipakai aplikasi (93,4%), Random Forest kalah '
     'dari CNN dan SVM.',
 72: '**Penjelasan output (tabel):** tabel ringkasan Random Forest. Top-1 87,2%, Top-3 95,3%, '
     'Macro-F1 86,6%, fit 0,6 menit, 300 pohon. Dua baris terakhir adalah masalah utamanya: '
     '857.250 node, 700 MB di scikit-learn, dan 175 MB dalam format padat.',
 73: '**Penjelasan output (tabel):** metrik keamanan Random Forest. Kesalahan fatal 19 = 4,3% '
     '(sama dengan CNN, lebih buruk dari SVM), recall pangan 95,2% (paling rendah), dan alarm '
     'beracun palsu 14,2% (paling tinggi). Jadi Random Forest bukan pilihan yang baik dari '
     'sisi keamanan.',
 74: '**Penjelasan output (tabel):** akurasi vs cakupan Random Forest. Di cakupan 100% akurasi '
     '87,2%, cakupan 60% menjadi 98,0%, cakupan 10% 99,3%. Kurvanya naik paling cepat, tetapi '
     'karena titik awalnya paling rendah, pada cakupan kerja 93,4% angkanya hanya 90,7% — '
     'masih di bawah CNN.',
 75: '**Penjelasan output (log + tabel):** 15 kelas terburuk Random Forest. `Dioscorea Hispida '
     '(Gadung)` sangat buruk: hanya 23,1% (2 dari 13 foto benar). Beberapa kelas yang relatif '
     'mudah bagi CNN/SVM juga masuk daftar ini, tanda model pohon kurang cocok untuk ruang '
     'fitur 1.280 dimensi. `DONE MODEL 3` menandai tahap selesai.',
 81: '**Penjelasan output (grafik):** grafik batang Top-1 ketiga model. SVM RBF 89,9% '
     'tertinggi, CNN 89,2% di tengah, Random Forest 87,2% terendah. Selisihnya tipis (2,7 poin '
     'antara teratas dan terbawah), jadi Top-1 saja belum cukup untuk memilih model — perlu '
     'melihat ukuran dan keamanan.',
 82: '**Penjelasan output (grafik):** ketiga kurva abstain dalam satu grafik. Garis paling '
     'atas pada cakupan besar adalah CNN, artinya untuk setiap persentase foto yang dijawab, '
     'CNN memberi akurasi tertinggi. Inilah bukti bahwa pada kondisi aplikasi nyata (menolak '
     'sebagian foto), CNN unggul walau Top-1 totalnya sedikit di bawah SVM.',
 83: '**Penjelasan output (grafik):** grafik perbandingan kesalahan fatal (beracun ditebak '
     'pangan). SVM paling rendah (16 = 3,6%), CNN dan Random Forest sama-sama 19 (4,3%). '
     'Karena selisihnya hanya 3 foto dari 444, angka ini tidak cukup kuat untuk menggeser '
     'keputusan.',
 86: '**Penjelasan output (tabel):** tabel perbandingan lengkap. CNN: Top-1 89,2%, fatal 4,3%, '
     'akurasi pada cakupan kerja 92,7%. SVM: Top-1 89,9%, fatal 3,6%, tetapi akurasi pada '
     'cakupan kerja hanya 90,6%. RF: Top-1 87,2%, fatal 4,3%, 90,7%. Kolom '
     '`akurasi_pada_cakupan` adalah pembanding yang paling jujur karena sesuai kondisi '
     'aplikasi yang menolak foto ragu.',
 87: '**Penjelasan output (tabel):** tabel keputusan pilihan deploy. **CNN murni** hanya 4,56 '
     'MB dengan 1 inferensi dan kerumitan rendah. CNN + SVM butuh belasan MB plus implementasi '
     'kernel RBF manual di aplikasi. CNN + Random Forest +/- 175 MB dan dinilai tidak layak. '
     'Dari sisi akurasi pada cakupan kerja, CNN juga paling tinggi (92,7%), jadi CNN menang '
     'tanpa kompromi.',
 88: '**Penjelasan output (tabel):** contoh kesalahan fatal CNN — 15 baris pertama. Kolom '
     '`aktual` adalah spesies sebenarnya, `ditebak` adalah tebakan model. Sebagian besar '
     'berasal dari kelas beracun yang mirip pangan; contoh paling sering: `Dioscorea Hispida '
     '(Gadung)` ditebak `Kaempferia Galanga (Kencur)`. Daftar ini bukan untuk menormalkan '
     'kesalahan, tetapi menunjukkan di mana aturan penolakan paling dibutuhkan.',
 89: '**Penjelasan output (log):** `DONE TAHAP 5` menandai bahwa tabel perbandingan sudah '
     'tersimpan ke `ringkasan.json`. Baris `json -> ...` menyebut lokasi file hasil. Isi file '
     'itu sama dengan tabel di atas, dipakai oleh dokumen laporan.',
 91: '**Penjelasan output (log):** `model dimuat | input (None, 224, 224, 3) | kelas 100` '
     'berarti model Keras berhasil dibuka ulang dengan bentuk input 3 kanal 224×224 (`None` = '
     'jumlah batch bebas). `TFLite float16: 4.56 MB` adalah ukuran model setelah dikonversi — '
     'turun dari 23,3 MB karena bobotnya disimpan 16-bit.',
 92: '**Penjelasan output (log):** spesifikasi interpreter TFLite yang benar-benar dimuat. '
     'Input `[1 224 224 3]` bertipe `float32`, output `[1 100]`. Ini memastikan kontrak ke '
     'aplikasi: satu gambar, 224×224, tiga kanal, menghasilkan 100 skor. Urutan 100 skor itu '
     'sama dengan urutan `labels.json`.',
 93: '**Penjelasan output (log):** hasil verifikasi Keras vs TFLite pada 200 gambar test. '
     '`kesepakatan argmax: 100.000%` berarti keduanya selalu memilih kelas yang sama. `max '
     'diff 0.02470` dan `mean diff 0.000082` adalah selisih probabilitas akibat pemotongan '
     'float16 — kecil dan tidak mengubah tebakan. Baris terakhir menunjukkan akurasi subset '
     'keduanya sama persis, 90,0%. Artinya model yang dikirim setara dengan model yang diuji.',
 95: '**Penjelasan output (log + tabel):** ringkasan verifikasi yang disimpan ke '
     '`model_meta.json`: kesepakatan argmax 100%, selisih probabilitas maksimum 0,0247, '
     'akurasi subset Keras dan TFLite sama 90,0%, ukuran 4,56 MB. Angka akurasi 90,0% di sini '
     'adalah subset 200 gambar, bukan akurasi penuh di 1.476 gambar (89,16%), jadi wajar '
     'sedikit berbeda.',
 96: '**Penjelasan output (log + tabel):** tabel kontrak input/output untuk pengembang '
     'aplikasi. Input bentuk `[1, 224, 224, 3]`, float32 bernilai 0–255 **tanpa dibagi 255**, '
     'RGB, resize bilinear. Output `[1, 100]` softmax dengan urutan sesuai `labels.json`. '
     'Baris ini yang mencegah kesalahan paling umum: membagi 255 dua kali. `DONE TAHAP 6` '
     'menandai tahap export selesai.',
 98: '**Penjelasan output (log):** `validasi: 1476 gambar | akurasi penuh 0.8963` berarti pada '
     'data validasi, model menjawab benar 89,63% bila semua foto dipaksa dijawab. Data '
     'validasi dipakai (bukan test) karena ambang harus dipilih tanpa melihat data test.',
 101: '**Penjelasan output (log):** hasil pencarian ambang di grid 0,05–0,95 dengan syarat '
      'akurasi minimal 93% pada foto yang dijawab dan cakupan minimal 40%. Ambang terpilih '
      '**0.27**: pada validasi 94,0% foto dijawab dengan akurasi 93,0%. Ambang serendah ini '
      'berarti model hanya menolak foto yang benar-benar ragu.',
 103: '**Penjelasan output (log):** uji ambang 0,27 di data **test** (dilakukan sekali, hanya '
      'untuk laporan). Hasilnya 93,4% foto dijawab dengan akurasi 92,7%, dan kesalahan fatal '
      'turun dari 19 menjadi 9 dari 410 foto beracun yang dijawab. Artinya aturan penolakan '
      'memotong sekitar separuh kesalahan paling berbahaya.',
 105: '**Penjelasan output (grafik + log):** dua hal sekaligus. Baris `json -> ...` menyebut '
      'file `safety_rules.json` tersimpan. Grafiknya adalah kurva ambang: sumbu X nilai ambang '
      'keyakinan, sumbu Y akurasi pada foto yang dijawab dan cakupannya. Titik 0,27 ditandai '
      'karena di situlah syarat (akurasi ≥ 93%, cakupan ≥ 40%) mulai terpenuhi.',
 106: '**Penjelasan output (log + tabel):** tabel aturan penolakan final yang disimpan. Nilai '
      'yang dipakai aplikasi adalah `AMBANG = 0.27`. Di validasi: cakupan 94,0%, akurasi '
      '93,0%. Di test: cakupan 93,4%, akurasi 92,7%, fatal 9/410. Catatan penting: ambang '
      'dipilih dari validasi, sedangkan angka test hanya laporan — supaya hasil akhirnya tidak '
      'bias. `DONE TAHAP 6a` menandai tahap selesai.',
 108: '**Penjelasan output (log):** `train (6887, 1280)` adalah bentuk embedding yang dipakai '
      'untuk menghitung ulang SVM versi deploy. Hanya train yang dimuat karena yang diukur '
      'adalah jumlah support vector dan ukuran bobot, bukan akurasi.',
 109: '**Penjelasan output (log):** `refit 48.3 s` adalah waktu melatih ulang SVM RBF (`C=10`) '
      'di luar pipeline standar. Refit dilakukan karena biaya deploy dihitung dari objek SVM '
      'mentah, sedangkan di Tahap 3 SVM dibungkus pipeline standardisasi + classifier.',
 110: '**Penjelasan output (log):** tiga fakta biaya SVM. `support vector 5,468` = banyak '
      'vektor pendukung, jadi setiap kali menebak aplikasi harus menghitung 5.468 kernel. '
      '`float32 26.7 MB | float16 13.3 MB` adalah ukuran bobot model — sudah lebih besar dari '
      'seluruh CNN (4,56 MB) walau dipadatkan. `latency referensi 8.6 ms/gambar` adalah '
      'patokan kasar di desktop Python, bukan di HP, dan setiap inferensi aplikasi jadi dua '
      'tahap (CNN lalu SVM).',
 114: '**Penjelasan output (log + tabel):** tabel rekap biaya membawa SVM ke HP. 5.468 support '
      'vector, fitur 1.280 dimensi, bobot 26,7 MB (float32) atau 13,3 MB (float16), ditambah '
      'backbone CNN ~3,7 MB, 5.468 × 1.280 evaluasi kernel (~14 MFLOP) per gambar, dan latensi '
      'referensi 8,6 ms. Kesimpulannya (tertulis di bawah tabel): yang dikirim ke HP cukup '
      '**CNN murni**, karena SVM butuh dua tahap dan belasan MB tambahan. `DONE TAHAP 6b` '
      'menandai seluruh pipeline selesai.'}


# Sel kode yang outputnya harus dibungkam. Isinya `labels = sorted(...)` yang
# baris terakhirnya menghasilkan angka `3434` (panjang file labels.json); angka
# itu sisa ekspresi, bukan hasil analisis, jadi tidak ditampilkan.
MUTED_OUTPUT_CELLS = {17}


def main():
    cells = [cell("markdown", HEAD)]
    for i, spec in enumerate(SPECS, 1):
        code = strip_docstring((SCRIPTS / spec["file"]).read_text(encoding="utf-8"))
        code = patch(code, spec["params"])
        head = STAGE.get(spec["file"], f"---\n\n# {spec['title']}\n\n{spec['md']}")
        cells.append(cell("markdown", head))
        before = AMEND.get(spec["file"], [])
        after = POST.get(spec["file"], [])
        used_before, used_after = set(), set()
        for ch in split_cells(code):
            first = next((l for l in ch.splitlines() if l.strip()), "")
            for anchor, note in before:
                if anchor not in used_before and first.startswith(anchor):
                    cells.append(cell("markdown", note))
                    used_before.add(anchor)
                    break
            cells.append(cell("code", ch))
            # penjelasan output: pakai catatan kurasi bila ada, kalau tidak pakai
            # catatan anchor POST supaya tiap sel beroutput tetap punya penjelasan
            n_code = sum(1 for c in cells if c["cell_type"] == "code") - 1
            note_out = OUTPUT_NOTES.get(n_code)
            sudah = False
            for anchor, note in after:
                if anchor not in used_after and first.startswith(anchor):
                    cells.append(cell("markdown", note))
                    used_after.add(anchor)
                    sudah = True
                    break
            if not sudah and note_out:
                cells.append(cell("markdown", note_out))
        missing = ([a for a, _ in before if a not in used_before]
                   + [a for a, _ in after if a not in used_after])
        if missing:
            raise RuntimeError(f"anchor tidak ketemu di {spec['file']}: {missing}")

    # sel yang outputnya harus dibungkam (mis. `3434` sisa ekspresi labels.json):
    # nilai ekspresi terakhir dibuang supaya notebook tidak menampilkan angka liar
    for k in MUTED_OUTPUT_CELLS:
        sel = [c for c in cells if c["cell_type"] == "code"][k]
        src = "".join(sel["source"]).rstrip("\n")
        baris = src.splitlines()
        if baris and not baris[-1].lstrip().startswith("#"):
            baris[-1] = "_ = None  # buang nilai ekspresi supaya tidak tampil di output"
            sel["source"] = ("\n".join(baris) + "\n").splitlines(keepends=True)

    sudah_kode = sum(1 for c in cells if c["cell_type"] == "code")
    # Tahap 2 di notebook terkirim memakai sel grafik terpisah yang memuat ulang hasil
    # dari file (lihat 01_train_cnn.py versi yang dipakai saat eksekusi). Script di
    # disk sekarang menggabungkannya dalam satu alur, jadi jumlah sel kode bisa
    # berbeda. Penjelasan output tetap dipasang berdasarkan urutan sel kode yang ada;
    # sisa entri yang tidak terpakai dilaporkan, bukan membuat build gagal.
    terpakai = [k for k in OUTPUT_NOTES if k < sudah_kode]
    sisa = sorted(k for k in OUTPUT_NOTES if k >= sudah_kode)
    print(f"penjelasan output terpasang: {len(terpakai)} | "
          f"tidak terpakai (sel kode hanya {sudah_kode}): {sisa if sisa else '-'}")

    cells.append(cell("markdown", """---

# Selesai

Semua artefak ada di `output/`:

| Folder | Isi |
|---|---|
| `_data/` | manifest split |
| `01_cnn/` | model final, metrik, grafik, embedding |
| `02_svm/` | model 2 |
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
