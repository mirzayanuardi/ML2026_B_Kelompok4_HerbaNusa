# HerbaNusa — ML Inti (Tahap 0–6)

Output pipeline machine learning untuk klasifikasi 100 spesies tanaman obat
Indonesia. Berhenti tepat di export TFLite, sesuai lingkup yang diminta.

## Baca dulu

| Urutan | File | Untuk siapa |
|---|---|---|
| 1 | `RINGKASAN.md` | Anda — laporan lengkap: semua angka, keputusan, dan batasan |
| 2 | `PENJELASAN_PROYEK.md` | Anda/penguji — penjelasan proyek dari nol: latar belakang, dataset, ketiga algoritma, dan alasannya |
| 3 | `HerbaNusa_ML.ipynb` | Anda — seluruh pipeline; grafik dan tabel tampil langsung di dalamnya |
| 4 | `05_bahan_aplikasi/README.md` | **Pengembang aplikasi** — kontrak input, aturan penolakan, matriks badge, checklist uji |

## Yang diserahkan ke teman Anda

Seluruh isi `05_bahan_aplikasi/`:

```
model.tflite        4,56 MB   input [1,224,224,3] float32 0..255 RGB
labels.json         100 kelas, urutan = urutan output
model_meta.json     spesifikasi + hasil verifikasi numerik
safety_rules.json   AMBANG = 0.27 + aturan penolakan
README.md           dokumen acuan aplikasi
metrics_model_1.json
```

**Satu aturan yang tidak boleh dilanggar:** kirim piksel 0–255 apa adanya.
Normalisasi ada di dalam model. Dibagi 255 → akurasi hancur tanpa error.

## Hasil singkat

| Model | Top-1 | Top-3 | Fatal | Ukuran |
|---|---|---|---|---|
| CNN MobileNetV2 224px (**dikirim**) | 89,2% | 95,3% | 4,3% → **2,2%** dengan AMBANG | 4,56 MB |
| SVM RBF (embedding CNN) | 89,9% | 95,7% | 3,6% | ±17 MB + kode Dart |
| Random Forest 300 | 87,2% | 95,3% | 4,3% | ±175 MB |

Dengan `AMBANG = 0.27`: 93,4% foto dijawab, akurasi **92,7%**. Pada cakupan yang sama CNN mengalahkan SVM (92,7% vs 90,6%).

## Cara menjalankan ulang

Script ada di `../scripts/`, jalan berurutan:

```powershell
$py = "C:\Users\Wahyu Pratomo\anaconda3\envs\ENV_001\python.exe"
$s  = "C:\Users\Wahyu Pratomo\Project Machine Learning\tanaman-obat\scripts"

& $py $s\00_eda.py              # EDA pada data mentah (1 menit)
& $py $s\00_prepare.py          # bersihkan + split (2 menit)
& $py $s\01_train_cnn.py --img 224 --head 3 --fine 15 --batch 16 `
      --dropout 0.4 --ls 0.1 --out 01_cnn      # 60 menit, CPU
& $py $s\02_svm.py              # 1 menit
& $py $s\03_rf.py               # 1 menit
& $py $s\04_compare.py          # 10 detik
& $py $s\05_export_tflite.py    # 12 menit (verifikasi 200 gambar)
& $py $s\06_threshold.py        # tentukan AMBANG dari validasi
& $py $s\06b_svm_deploy.py      # hitung biaya deploy SVM
& $py $s\build_notebooks.py     # bangun ulang output/HerbaNusa_ML.ipynb (belum dieksekusi)
cd output
& $py -m jupyter nbconvert --to notebook --execute --inplace `
      --ExecutePreprocessor.timeout=10800 HerbaNusa_ML.ipynb
```

`build_notebooks.py` hanya merakit notebook dari script; menjalankannya butuh
`nbconvert` di atas (±80 menit) supaya sel punya output seperti file yang
diserahkan.

Urutan tahap di notebook: **Tahap 0 EDA (data mentah)** → **Tahap 1 pembersihan
+ split** → **Tahap 2 training**. EDA sengaja dijalankan sebelum pembersihan
dan sebelum preprocessing; preprocessing citra (resize 224, augmentasi,
normalisasi) baru ada di Tahap 2.

Urutan penting: `02`–`06b` membaca artefak dari `01_cnn`, jadi latih model
dulu. `06_threshold.py` harus dijalankan setelah `01_train_cnn.py` dan
sebelum laporan dipakai, karena `safety_rules.json` berasal dari situ.

Percobaan pembanding (tidak perlu dijalankan ulang):

```powershell
& $py $s\01_train_cnn.py --img 128 --out 01_cnn_128
& $py $s\02_tta_eval.py --src 01_cnn_128
```

## Isi folder

| Folder | Isi |
|---|---|
| `_data/` | manifest split + `labels.json` |
| `eda/` | grafik dan statistik EDA (dihitung dari data mentah) |
| `01_cnn/` | model final 224px, metrik, 4 grafik, embedding |
| `01_cnn_128/` | percobaan 128px (pembanding) |
| `01_cnn_128_tta/` | evaluasi TTA (pembanding) |
| `02_svm/` | SVM, metrik, grafik |
| `03_rf/` | Random Forest |
| `04_perbandingan/` | tabel + grafik perbandingan |
| `05_bahan_aplikasi/` | **paket serah terima** |
| `HerbaNusa_ML.ipynb` | Notebook tunggal seluruh pipeline, sudah dieksekusi |

## Lingkungan

Python 3.14.6 (conda `ENV_001`), TensorFlow 2.22.0-rc0, scikit-learn 1.9.0.
CPU-only, 4 core. Dataset asal:
`C:\Users\Wahyu Pratomo\Downloads\Indonesia-Medicinal-Plant-dataset`.
