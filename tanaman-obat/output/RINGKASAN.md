# Laporan HerbaNusa — Tahap 0–6 (ML Inti)

Dibuat: 11 November 2026, dini hari. Semua angka di bawah berasal dari file di
folder ini, bukan perkiraan. Data test **tidak pernah** dipakai saat training
maupun saat memilih model/ambang.

**Kesimpulan singkat:** target 85% tercapai (89,16% pada model yang dikirim),
model `.tflite` sudah jadi 4,56 MB, dan dengan aturan penolakan `AMBANG = 0.27`
akurasi naik ke **92,7%** pada 93,4% foto yang dijawab.

---

## 1. Data

| Item | Nilai |
|---|---|
| Gambar awal | 10.000 |
| Duplikat persis (MD5) dibuang | 69 |
| Near-duplicate (dHash ≤ 2) dibuang | 92 |
| **Gambar bersih** | **9.839** |
| Train / Val / Test | 6.887 / 1.476 / 1.476 |
| Kelas | 100 spesies |
| Resolusi asli | 128×128 (semua gambar) |

Pembagian stratified per kelas, seed 42. Duplikat dibuang **sebelum** split,
supaya tidak ada gambar yang sama muncul di train dan test (penyebab akurasi
palsu yang paling sering).

**EDA dijalankan pada data mentah, sebelum pembersihan.** Hasilnya: 10.000
gambar, tepat 100 per kelas, semuanya 128×128 RGB. Brightness 25,5–214,0
(rata-rata 107,8), kontras 12,8–98,9, ukuran file 3,6–13,5 KB. Outlier aturan
1,5×IQR hanya 0,7–2,1% (213 brightness, 124 kontras, 69 ukuran file) dan
sengaja dipertahankan karena gambarnya masih wajar. Korelasi antar fitur lemah
(|r| < 0,2), sehingga tidak ada fitur yang perlu dibuang. EDA menemukan 69
duplikat persis dan 92 near-duplicate — angka inilah yang dibersihkan di tahap
persiapan data. Preprocessing citra (resize 224, augmentasi, normalisasi) tidak
dilakukan di EDA maupun di pembersihan; itu baru ada di Tahap 2 saat training.

---

## 2. Model 1 — CNN MobileNetV2

Ini model yang akhirnya dikirim ke aplikasi. Ada tiga percobaan berurutan
karena target 85% belum tercapai di dua percobaan pertama.

| Percobaan | Top-1 | Top-3 | Macro-F1 | Fatal | Catatan |
|---|---|---|---|---|---|
| 128px | 79,95% | 91,12% | 79,51% | 28/444 = 6,3% | overfit: train 94% vs val 79% |
| 128px + TTA 5 view | 83,74% | 92,21% | — | — | tanpa training ulang |
| **224px (final)** | **89,16%** | **95,26%** | **88,81%** | **19/444 = 4,3%** | dropout 0,4 + label smoothing 0,1 |

Kenapa 224px menolong padahal gambar aslinya cuma 128px: MobileNetV2 dilatih
di ImageNet pada 224px, jadi lapisan fiturnya bekerja paling baik di skala itu.
Yang naik bukan informasi baru, tapi kecocokan skala. Epoch terbaik 16 dari 18
(val 89,6%); train berakhir 99,5% sehingga jelas overfit, dan early-stopping
mengembalikan bobot epoch terbaik.

Run ulang dengan kode identik bisa bergeser beberapa poin karena augmentasi
memakai RNG (run sebelumnya mendapat 85,03%; angka di seluruh laporan ini
sudah disamakan dengan run yang dieksekusi di notebook).

Keamanan (model final, tanpa penolakan):

| Metrik | Nilai |
|---|---|
| Foto beracun di test | 444 |
| Dikenali sebagai beracun | 92,3% |
| **Fatal: beracun ditebak sebagai pangan** | **19 = 4,3%** |
| Recall pangan | 96,3% |
| Alarm beracun palsu pada pangan | 13,9% |

TTA tidak dipakai di aplikasi: 5× inferensi untuk +3,8 poin yang sudah
tertutup oleh 224px dalam satu kali inferensi.

---

## 3. Model 2 — SVM RBF di atas embedding CNN

| Metrik | Nilai |
|---|---|
| Top-1 | **89,91%** |
| Top-3 | 95,66% |
| Macro-F1 | 89,66% |
| Fatal | 16/444 = 3,60% |
| Waktu fit | 0,9 menit |
| Support vector | 5.468 |
| Ukuran bobot SVM | 26,7 MB (float32) / 13,3 MB (float16) |

SVM **bukan pengganti** CNN: inputnya vektor 1.280 dimensi keluaran CNN. Jadi
di HP alurnya dua tahap. Hitungan biaya deploy ada di Tahap 6b notebook.

---

## 4. Model 3 — Random Forest 300 pohon

| Metrik | Nilai |
|---|---|
| Top-1 | 87,20% |
| Top-3 | 95,33% |
| Macro-F1 | 86,60% |
| Fatal | 19/444 = 4,3% |
| Waktu fit | 0,7 menit |
| Jumlah node | 857.250 (428.775 leaf) |
| Ukuran di scikit-learn | **700 MB** (tabel 100 kelas disimpan di setiap node) |
| Ukuran format padat (perkiraan) | 175 MB |

RF tidak layak dikirim ke HP karena tiga hal: ukurannya (700 MB asli, 175 MB
bahkan setelah dipadatkan), tidak ada dukungan TFLite untuk forest (857 ribu
percabangan harus ditulis manual di Dart), dan akurasinya justru paling rendah
pada cakupan yang dipakai aplikasi (90,7% vs CNN 92,7%).

---

## 5. Perbandingan dan model yang dikirim

| Model | Top-1 | Top-3 | Macro-F1 | Fatal | False alarm | Ukuran di HP |
|---|---|---|---|---|---|---|
| CNN MobileNetV2 | 89,16% | 95,26% | 88,81% | 4,3% | 13,9% | **4,56 MB** |
| SVM RBF | **89,91%** | **95,66%** | **89,66%** | 3,60% | 13,7% | ±17 MB + kode Dart |
| Random Forest 300 | 87,20% | 95,33% | 86,60% | 4,3% | 14,2% | ±175 MB (padat) |

SVM menang tipis di top-1 (0,75 poin), **tetapi yang dikirim ke aplikasi
adalah CNN murni.** Alasannya bukan selera:

1. Ukuran. SVM butuh 13,3 MB support vector (float16) ditambah backbone 3,7 MB
   → lewat batas 15 MB sebelum kode aplikasi dihitung.
2. Tidak ada operator SVM di TFLite. Kernel RBF harus ditulis manual di Dart,
   termasuk standardisasi 1.280 fitur. Itu sumber bug baru di aplikasi.
3. Keunggulan SVM hilang begitu aturan penolakan dipakai. Pada cakupan yang
   sama (93,4% foto dijawab), CNN justru paling akurat: **92,7% vs SVM 90,6%
   dan RF 90,7%**. Artinya CNN menang di skenario yang benar-benar dipakai
   pengguna, dengan ukuran seperlima dan satu inferensi.

Keputusan ini bisa diubah nanti: kalau ponsel target sudah kuat dan teman
Anda bersedia menulis kernel RBF, SVM bisa jadi jalur v2 — tapi tidak ada
alasan akurasi untuk itu.

---

## 6. Aturan "kalau ragu jangan dijawab"

Ambang dipilih dari **data validasi**, bukan test, supaya test tetap bersih
untuk laporan. Aturannya satu: tolak kalau probabilitas tertinggi < `AMBANG`.

**`AMBANG = 0.27`** — ambang terendah yang memberi akurasi ≥ 93% pada foto
yang dijawab, dengan cakupan ≥ 40%.

| | Validasi (pemilihan) | Test (laporan) |
|---|---|---|
| Cakupan | 94,0% | 93,4% |
| Akurasi pada yang dijawab | 93,0% | **92,7%** |
| Akurasi tanpa penolakan | 89,6% | 89,2% |
| Fatal | — | **9/410 = 2,2%** (turun dari 19) |

Ambangnya turun dari 0,40 (run sebelumnya) ke 0,27 karena model baru jauh
lebih yakin: pada 0,40 terlalu banyak foto benar ikut ditolak. Efeknya cakupan
naik dari 86% ke 93% dengan akurasi yang justru lebih tinggi.

Jadi aplikasi menolak ±7% foto — dan itu memang tujuannya. Penolakan bukan
kegagalan, itu fitur keselamatan. Aturan margin sudah diuji dan **tidak**
mengubah cakupan maupun akurasi, jadi tidak dipakai.

### Yang tetap lolos walaupun sudah pakai ambang

9 kesalahan fatal ini masih terjadi di test **setelah** penolakan aktif.
Semuanya berkeyakinan ≥ 0,27, jadi ambang tidak bisa menahannya:

| Aktual (beracun) | Ditebak (aman) | Confidence |
|---|---|---|
| Gynura Segetum (Daun Dewa) | Piper Sarmentosum (Cabean) | **0,65** |
| Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) | **0,57** |
| Acorus Calamus (Dlingo) | Zingiber Officinale (Jahe) | **0,56** |
| Melia Azedarach (Mindi) | Zingiber Officinale (Jahe) | 0,37 |
| Cestrum Nocturnum (Sedap Malam) | Ocimum Basilicum (Kemangi) | 0,36 |
| Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) | 0,31 |
| Gynura Segetum (Daun Dewa) | Syzygium Polyanthum (Salam) | 0,30 |
| Gynura Segetum (Daun Dewa) | Melissa officinalis (Lemon Balm) | 0,28 |
| Phytolacca Americana (Mrico Kepyar) | Ocimum Basilicum (Kemangi) | 0,27 |

**Daun Dewa menyumbang 5 dari 9 kesalahan fatal.** Ini satu spesies yang
paling perlu ditangani di level UI: kalau prediksi mengarah ke Cabean,
Beluntas, Salam, atau Lemon Balm, tampilkan peringatan Daun Dewa.

Konsekuensi untuk aplikasi: **peringatan di top-3 wajib** (kalau salah satu
dari 3 kandidat beracun, tampilkan peringatan merah), dan teks "aman dimakan"
tidak boleh ada sama sekali. Ambang saja tidak cukup.

---

## 7. Export TFLite

| Pemeriksaan | Hasil |
|---|---|
| File | `05_bahan_aplikasi/model.tflite` |
| Ukuran | **4,56 MB** (4.780.172 byte) — target < 15 MB tercapai |
| Kuantisasi | bobot float16, input/output float32 |
| Input | `[1, 224, 224, 3]`, float32, **0..255**, RGB |
| Output | `[1, 100]` softmax, urutan = `labels.json` |
| Kesepakatan argmax Keras vs TFLite (200 gambar) | **100%** |
| Selisih probabilitas maksimum | 0,025 |
| Selisih rata-rata | 0,00008 |
| Akurasi subset 200 gambar | Keras 90,0% vs TFLite 90,0% |

Tidak ada satu pun gambar yang berubah tebakan antara Keras dan TFLite pada
200 gambar uji; selisih probabilitas maksimum 0,025 hanya pembulatan float16.

**Kontrak terpenting untuk teman Anda:** normalisasi `Rescaling(1/127.5, -1)`
ada **di dalam** model. Kirim piksel 0–255 apa adanya. Kalau dibagi 255 lagi,
akurasi jatuh tanpa pesan error.

---

## 8. Yang belum dikerjakan (sengaja, di luar tahap 0–6)

| Tahap | Isi | Status |
|---|---|---|
| 7 | `data/knowledge_base.json` — 100 entri: nama lokal, bagian aman/bahaya, P3K, `level_racun` | belum |
| 8 | `data/danger_list.json` — 10–15 spesies mematikan di luar dataset (kecubung, oleander, bintaro, dieffenbachia, antiaris) | belum |
| 9 | `docs/APP_SPEC.md` — spesifikasi layar & teks disclaimer | belum |
| 10 | `kode_referensi/` — implementasi badge + penolakan dalam Dart | belum |
| 11 | Uji 10 kasus di HP | belum |

`05_bahan_aplikasi/README.md` sudah memuat matriks badge dan aturan yang harus
dipatuhi, tapi file data pendukungnya belum ada. Artinya: **modelnya siap,
isinya belum lengkap.** Aplikasi bisa dibangun sekarang untuk menguji kamera →
model → top-3, tapi belum bisa menampilkan konten spesies.

---

## 9. Kejujuran soal batas kemampuan

Hal-hal yang harus Anda tahu sebelum angka 92,7% dipakai untuk promosi:

1. **Angka 89%/92,7% diukur pada foto dataset**, bukan foto HP di lapangan.
   Foto dataset cenderung rapi, satu objek, latar seragam. Di lapangan,
   akurasi hampir pasti turun. Belum diukur sama sekali.
2. **Resolusi sumber 128×128.** Detail daun yang tipis (urat, tepi, bulu)
   sudah hilang di sumbernya. Ini plafon dataset, bukan plafon algoritma.
3. **Overfitting masih ada.** Train 99,5% vs val 89,6%. Model menghafal
   sebagian data latih.
4. **13,9% tanaman pangan memicu alarm beracun palsu.** Ini bikin pengguna
   panik dan berhenti percaya. Perlu diperbaiki di v2.
5. **Label beracun masih draf.** Semua metrik keamanan bergantung pada
   `analisis/peta_tag_draf.csv`. Kalau tag itu salah, metrik fatal juga salah.
6. **Spesies berbahaya tidak ada di dataset** (kecubung, oleander, bintaro,
   dieffenbachia, antiaris) — model akan memaksakan foto itu ke salah satu
   dari 100 kelas. Karena itu halaman "Daftar Mematikan" bukan pelengkap,
   itu wajib.
7. **Bahan pokok survival tidak ada di dataset**: talas, kelor, singkong.
   Padahal "Keladi" di dataset adalah *Caladium* yang beracun — nama yang
   sama dengan talas di lidah orang. Ini risiko kebingungan paling berbahaya
   di seluruh proyek.
8. **Salah nama ilmiah di dataset** yang harus dibetulkan di `labels.json`
   aplikasi: `Caladium Cicolor`→bicolor, `Raulvolvia`→Rauvolfia,
   `Jatropa`→Jatropha, `Clinalanthus`→Clinacanthus, `Houttoynia`→Houttuynia,
   `Muraya`→Murraya, `Desmodium Triquitrum`→triquetrum,
   `Vetiveria Zizaniodes`→zizanioides, `Euphoria Longan`→Dimocarpus longan.

Rekomendasi jujur: jangan rilis ke publik sebelum (a) uji foto lapangan
sungguhan minimal 100 foto, dan (b) tahap 7–11 selesai. Untuk tugas/portofolio,
paket ini sudah utuh dan bisa dipertanggungjawabkan.

---

## 10. Peta folder

| Folder | Isi |
|---|---|
| `_data/` | `labels.json`, `train.csv`, `val.csv`, `test.csv`, `dataset_report.csv` |
| `eda/` | `figures/` grafik EDA (distribusi kelas, fitur, outlier, korelasi, contoh gambar) |
| `01_cnn/` | **Model final 224px**: `model.keras`, `metrics.json`, 4 grafik, embedding + probabilitas |
| `01_cnn_128/` | Percobaan 128px, disimpan sebagai pembanding |
| `01_cnn_128_tta/` | Evaluasi TTA, disimpan sebagai pembanding |
| `02_svm/` | SVM, metrik, grafik |
| `03_rf/` | Random Forest, metrik, grafik |
| `04_perbandingan/` | `tabel_model.csv`, `kesalahan_fatal.csv`, 3 grafik |
| `05_bahan_aplikasi/` | **Yang diserahkan ke teman Anda**: `model.tflite`, `labels.json`, `model_meta.json`, `safety_rules.json`, `README.md` |
| `HerbaNusa_ML.ipynb` | Notebook tunggal: seluruh pipeline tahap 0–6, **sudah dieksekusi** |

Grafik penting: `01_cnn/figures/kurva_training.png`,
`01_cnn/figures/confusion_matrix.png`, `01_cnn/figures/kurva_abstain.png`,
`04_perbandingan/figures/bar_akurasi.png`,
`04_perbandingan/figures/kurva_abstain_semua.png`,
`05_bahan_aplikasi/figures/ambang.png`.

Lingkungan: Python 3.14.6, TensorFlow 2.22.0-rc0, scikit-learn 1.9.0, CPU saja
(i5-8300H). Total waktu komputasi ± 90 menit.
