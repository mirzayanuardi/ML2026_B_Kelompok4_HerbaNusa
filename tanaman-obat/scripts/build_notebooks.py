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

HEAD = """# FloraSurvival AI — Pipeline ML Lengkap (Tahap 0–6)

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
            for anchor, note in after:
                if anchor not in used_after and first.startswith(anchor):
                    cells.append(cell("markdown", note))
                    used_after.add(anchor)
                    break
        missing = ([a for a, _ in before if a not in used_before]
                   + [a for a, _ in after if a not in used_after])
        if missing:
            raise RuntimeError(f"anchor tidak ketemu di {spec['file']}: {missing}")
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
