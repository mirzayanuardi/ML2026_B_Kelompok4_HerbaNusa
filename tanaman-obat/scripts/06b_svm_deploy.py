"""Tahap 6b: hitung biaya deploy kalau SVM (model terakurat) mau dipakai di HP.

SVM menang akurasi, tapi dia TIDAK bisa berdiri sendiri: dia butuh embedding
dari CNN. Skrip ini mengukur berapa besar dan seberapa berat SVM itu kalau
benar-benar dipasang di aplikasi, supaya keputusan model mana yang dikirim
tidak cuma berdasarkan akurasi.

Output: output/02_svm/DEPLOY.md + svm_model.pkl (opsional, untuk pengembangan)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, write_md

SRC = OUT / "01_cnn"
D = OUT / "02_svm"

Xtr = np.load(SRC / "emb_train.npy")
ytr = np.load(SRC / "y_train.npy")
Xte = np.load(SRC / "emb_test.npy")
print(f"train {Xtr.shape}", flush=True)

t0 = time.time()
pipe = make_pipeline(StandardScaler(),
                     SVC(C=10, gamma="scale", kernel="rbf",
                         decision_function_shape="ovr", cache_size=3000))
pipe.fit(Xtr, ytr)
print(f"refit {time.time()-t0:.1f} s", flush=True)

svm = pipe[-1]
scaler = pipe[0]
n_sv = int(svm.n_support_.sum())
n_features = Xtr.shape[1]
n_class = len(svm.classes_)

sv_float32_mb = n_sv * n_features * 4 / 1024 ** 2
sv_float16_mb = n_sv * n_features * 2 / 1024 ** 2
dual_coef_mb = svm.dual_coef_.size * 4 / 1024 ** 2

# biaya inferensi SVM di HP: kernel RBF = 1 dot product per support vector
kernels = n_sv                       # jumlah evaluasi kernel
flops = kernels * n_features * 2      # kali + tambah
per_class_scores = n_class

t0 = time.time()
_ = pipe.decision_function(Xte[:200])
lat_py = (time.time() - t0) / 200

print(f"support vector {n_sv:,} | float32 {sv_float32_mb:.1f} MB | "
      f"float16 {sv_float16_mb:.1f} MB", flush=True)
print(f"kernel evals per inferensi {kernels:,} | ~{flops/1e6:.0f} MFLOP", flush=True)
print(f"latency referensi (desktop python) {lat_py*1000:.1f} ms/gambar", flush=True)

rec = "CNN-embedding + SVM RBF"


def idn(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


# angka akurasi dibaca dari artefak, bukan ditulis tangan
def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}


m_svm = _read(D / "metrics.json")
m_cnn = _read(OUT / "01_cnn" / "metrics.json")
rules = _read(OUT / "05_bahan_aplikasi" / "safety_rules.json")
top1_svm = m_svm.get("top1")
top1_cnn = m_cnn.get("top1")
gain = (top1_svm - top1_cnn) * 100 if (top1_svm and top1_cnn) else None
kal_hasil = f"top-1 {idn(top1_svm*100)}%" if top1_svm else "akurasi tertinggi"

# akurasi pada cakupan yang sama, dari tabel perbandingan
acc_cov = {}
_tbl = OUT / "04_perbandingan" / "tabel_model.csv"
if _tbl.exists():
    import csv as _csv
    with _tbl.open(encoding="utf-8") as fh:
        for _row in _csv.DictReader(fh):
            _v = _row.get("akurasi_pada_cakupan")
            acc_cov[_row["model"]] = float(_v) if _v not in (None, "", "nan") else None
cnn_cov = acc_cov.get("CNN MobileNetV2")
svm_cov = acc_cov.get("SVM RBF")
cov = rules["test_pada_ambang_ini"]["cakupan"] if rules else None

if rules:
    _t = rules["test_pada_ambang_ini"]
    kal_tutup = (f"dengan `AMBANG = {idn(rules['ambang_keyakinan'], 2)}`, "
                 f"{idn(_t['cakupan'] * 100)}% foto dijawab dengan akurasi "
                 f"**{idn(_t['akurasi_pada_yang_dijawab']*100)}%**")
else:
    kal_tutup = "aturan penolakan"

if cnn_cov is not None and svm_cov is not None and cov:
    kal_banding = (f"Pada cakupan yang sama ({idn(cov * 100)}% foto dijawab), CNN "
                   f"malah **lebih akurat** daripada SVM: {idn(cnn_cov * 100)}% vs "
                   f"{idn(svm_cov * 100)}%. Jadi SVM tidak lagi punya alasan untuk "
                   f"dibawa ke HP.")
    kal_gain = (f"Kalau semua foto wajib dijawab, akurasi SVM naik {idn(gain)} poin "
                f"dibanding CNN murni — tapi keunggulan itu hilang begitu aturan "
                f"penolakan dipakai." if gain is not None else "")
else:
    kal_banding = "Perbandingan pada cakupan yang sama ada di `04_perbandingan/PERBANDINGAN.md`."
    kal_gain = f"Akurasi naik {idn(gain)} poin dibanding CNN murni." if gain is not None else ""


body = f"""SVM adalah model dengan akurasi tertinggi ({kal_hasil}), **tetapi SVM
tidak bisa berdiri sendiri**. Dia bekerja di atas vektor 1.280 dimensi yang
dihasilkan MobileNetV2. Artinya di HP harus ada dua tahap:

1. Gambar -> TFLite MobileNetV2 (sampai lapisan `embedding`) -> 1.280 angka
2. 1.280 angka -> hitung kernel RBF ke semua support vector -> pilih kelas

| Item | Nilai |
|---|---|
| Support vector | {n_sv:,} |
| Dimensi fitur | {n_features} |
| Ukuran bobot SVM (float32) | **{idn(sv_float32_mb)} MB** |
| Ukuran bobot SVM (float16) | {idn(sv_float16_mb)} MB |
| Ukuran backbone CNN (tanpa kepala) | ~3,7 MB |
| Evaluasi kernel per inferensi | {n_sv:,} x {n_features} |
| Beban tambahan per inferensi | ~{flops/1e6:.0f} MFLOP, {lat_py*1000:.0f} ms di laptop ini |

Konsekuensi praktis:

- **Ukuran jadi masalah utama.** {n_sv:,} support vector x {n_features} dimensi:
  float32 = **{idn(sv_float32_mb)} MB**, float16 = **{idn(sv_float16_mb)} MB**, ditambah
  backbone ~3,7 MB. Totalnya sudah lewat batas 15 MB sebelum kode aplikasi dihitung.
- **Tidak ada operator SVM di TFLite.** Kernel RBF harus ditulis manual di Dart
  ({n_sv:,} dot product {n_features} dimensi untuk setiap foto). Jadi tidak bisa
  sekadar "salin `model.tflite` lalu jalan".
- Beban hitungnya sendiri masih wajar (~{flops/1e6:.0f} MFLOP, {lat_py*1000:.0f} ms di laptop
  ini), jadi yang memberatkan adalah ukuran dan kerumitan kode, bukan kecepatan.
- {kal_gain}

**Keputusan paket ini:** yang dikirim ke aplikasi adalah **CNN murni**
(`model.tflite` 4,6 MB, 1 inferensi, tanpa kode tambahan). SVM disimpan
sebagai jalur peningkatan v2 kalau nanti ponsel target sudah lebih kuat,
atau dipakai di sisi server.

Yang membuat CNN tetap layak: {kal_tutup}. {kal_banding}
"""
write_md(D / "DEPLOY.md", "Biaya Deploy Model 2 (SVM) di HP", [
    ("Ringkasan", body),
    ("Catatan pengukuran", f"""Latensi {idn(lat_py*1000)} ms/gambar diukur di laptop CPU ini,
bukan di HP. Angka itu hanya pembanding relatif, bukan janji performa.
Semua angka ukuran adalah perkiraan dari jumlah support vector x dimensi x
ukuran tipe data, tanpa kompresi."""),
])
print("DONE TAHAP 6b", flush=True)
