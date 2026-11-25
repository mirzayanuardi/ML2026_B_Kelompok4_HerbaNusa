# HerbaNusa — Penjelasan Proyek

Dokumen ini menjelaskan proyek dari nol: kenapa dibuat, datanya apa, tiga
algoritma yang dipakai dan bagaimana masing-masing bekerja, sampai kenapa
model terakhir yang dipilih adalah CNN. Ditulis untuk dibaca orang yang belum
pernah melihat proyek ini — calon penguji, teman satu tim, atau Anda sendiri
enam bulan dari sekarang.

Semua angka di sini berasal dari file di folder `output/`, bukan perkiraan.
Laporan teknis lengkap ada di `RINGKASAN.md`.

---

## 1. Latar belakang

Indonesia punya sekitar 30.000 spesies tumbuhan dan lebih dari 7.000 di
antaranya tercatat dipakai sebagai obat tradisional. Di praktiknya, orang
mengenali tanaman obat dari ingatan, dari cerita orang tua, atau dari foto di
internet. Tiga masalah muncul dari cara itu:

1. **Salah kenal bisa mematikan.** Banyak tanaman obat punya kembaran yang
   beracun. Gadung liar, daun dewa, dlingo, dan mindi adalah contoh yang
   sudah memakan korban. Salah satu contoh paling sering: "keladi" di lidah
   orang bisa berarti talas yang dimakan, padahal *Caladium* yang ada di
   dataset justru beracun.
2. **Pengetahuan tidak merata.** Orang yang paham tanaman obat biasanya orang
   tua di desa. Anak muda yang merantau ke kota tidak punya akses itu.
3. **Internet tidak bisa diandalkan di lapangan.** Sinyal sering tidak ada di
   kebun, hutan, atau daerah 3T. Aplikasi yang butuh server tidak berguna
   justru di tempat ia paling dibutuhkan.

Karena itu arah proyeknya: **aplikasi ponsel yang bisa mengenali tanaman obat
secara offline** — tanpa internet, tanpa server, cukup kamera.

### Prinsip yang dipegang

> **Kalau ragu, jangan dijawab.**

Aplikasi ini bukan mainan kuis. Kalau model tidak yakin, lebih baik ia
mengatakan "tidak dikenali" daripada menebak dan membuat pengguna memakan
tanaman beracun. Seluruh desain teknis di bawah mengikuti prinsip ini, bukan
sebaliknya.

---

## 2. Tujuan dan ruang lingkup

| Item | Target | Status |
|---|---|---|
| Jumlah spesies | 100 tanaman obat Indonesia | tercapai |
| Akurasi top-1 | di atas 85% | **89,16%** |
| Ukuran model di HP | di bawah 15 MB | **4,56 MB** |
| Jalan offline | tanpa server | ya, `.tflite` |
| Perilaku saat ragu | menolak menjawab | ya, `AMBANG = 0.27` |
| Bandingkan 3 algoritma | berurutan lalu disimpulkan | ya |

**Yang dikerjakan:** tahap 0–6, yaitu pengolahan data, training tiga model,
perbandingan, export TFLite, dan penentuan aturan penolakan.

**Yang belum dikerjakan:** aplikasi Flutter, basis pengetahuan 100 spesies
(nama lokal, bagian aman/bahaya, P3K), daftar tanaman mematikan di luar
dataset, dan uji di HP. Modelnya siap; isi kontennya belum.

---

## 3. Kenapa 100 spesies, bukan 3 kategori

Ide pertama yang terlihat lebih sederhana: klasifikasikan langsung ke tiga
kategori — **pangan / obat / beracun**. Satu model, tiga kelas, selesai.

Ide itu dibuang, dan alasannya penting untuk dipahami:

- **Salah kategori = salah total.** Model yang hanya punya tiga kelas akan
  memaksa setiap foto masuk ke salah satunya. Foto tanaman yang tidak dikenal
  tetap akan dilabeli "pangan" atau "beracun" dengan yakin. Tidak ada jalan
  untuk bilang "saya tidak tahu".
- **Kategori bukan sifat gambar.** "Beracun" bukan bentuk daun. Tanaman yang
  sama bisa pangan di satu bagian dan beracun di bagian lain (contoh: buah
  okra aman, getahnya tidak). Informasi itu tidak terlihat dari piksel.
- **Tidak bisa diaudit.** Kalau model bilang "beracun", pengguna tidak tahu
  tanaman apa yang dimaksud, jadi tidak bisa memverifikasi sendiri.

Keputusan yang diambil: **model hanya menjawab nama spesies.** Kategori,
badge, bagian aman, dan peringatan diturunkan dari tabel pengetahuan
berdasarkan nama spesies itu.

```
foto → model → nama spesies → tabel pengetahuan → badge + peringatan
```

Konsekuensinya lebih aman: satu kesalahan klasifikasi masih bisa ditangkap di
lapisan aturan, sedangkan kesalahan kategori langsung menyesatkan.

---

## 4. Dataset

| Item | Nilai |
|---|---|
| Sumber | dataset Indonesia-Medicinal-Plant, 100 folder spesies |
| Jumlah gambar awal | 10.000 |
| Gambar per spesies | ± 100 |
| Resolusi asli | 128×128 piksel (semua gambar) |
| Format nama folder | `Nama Latin (Nama Lokal)` |

Contoh kelas: `Zingiber Officinale (Jahe)`, `Acorus Calamus (Dlingo)`,
`Gynura Segetum (Daun Dewa)`, `Piper Betle (Sirih)`,
`Dioscorea Hispida (Gadung)`.

### 4.1 EDA — membaca data sebelum diubah

EDA dikerjakan pada **data mentah**, sebelum pembersihan dan sebelum
preprocessing citra. Tujuannya membaca karakter data lebih dulu, supaya
keputusan pembersihan dan preprocessing punya dasar.

| Pemeriksaan | Hasil pada data mentah |
|---|---|
| Jumlah gambar / kelas | 10.000 / 100, tepat 100 per kelas |
| Resolusi & mode | 128×128, RGB, JPG |
| Ukuran file | 3,6–13,5 KB (rata-rata 9,3 KB) |
| Brightness (0–255) | 25,5–214,0, rata-rata 107,8 |
| Kontras (std piksel) | 12,8–98,9, rata-rata 48,9 |
| Missing value | 0 |
| Duplikat persis (MD5) | 69 |
| Near-duplicate (dHash ≤ 2) | 92 |
| Outlier 1,5×IQR | 213 brightness (2,1%), 124 kontras (1,2%), 69 ukuran file (0,7%) |
| Korelasi antar fitur | lemah (|r| < 0,2); lebar & tinggi konstan 128 |

Yang dibaca dari sini: dataset seimbang sempurna di data mentah, cahaya foto
bervariasi cukup lebar (karena itu augmentasi kecerahan dan kontras relevan),
outlier-nya sedikit dan masih wajar sehingga dipertahankan, dan ada 161 gambar
bermasalah yang harus dibuang. Grafik EDA ada di `eda/figures/`.

EDA **tidak** mengubah data: tidak ada resize, augmentasi, atau normalisasi di
tahap ini. Preprocessing citra baru dilakukan di Tahap 2, saat training.

### 4.2 Pembersihan data

Sebelum apa pun, dataset dibersihkan dua lapis:

| Langkah | Dibuang | Sisa |
|---|---|---|
| Awal | — | 10.000 |
| Duplikat persis (MD5) | 69 | 9.931 |
| Near-duplicate dalam kelas (dHash ≤ 2) | 92 | **9.839** |

**Kenapa ini wajib.** Dataset gambar sering punya foto yang sama tersimpan
dua kali, atau dua foto beruntun dari objek yang sama. Kalau dibiarkan, foto
yang sama bisa masuk ke data latih *dan* data uji. Model lalu terlihat pintar
padahal cuma menghafal. Angka akurasi palsu seperti ini adalah jebakan paling
umum di proyek klasifikasi gambar.

dHash bekerja dengan mengecilkan gambar jadi 9×8 piksel abu-abu, lalu
membandingkan pola terang-gelap tetangganya menjadi 64 bit. Dua gambar
dianggap near-duplicate kalau jarak Hamming-nya ≤ 2 bit. Ambang 2 dipilih
konservatif supaya tidak membuang variasi yang sah.

### 4.3 Pembagian data

Split 70/15/15, stratified per kelas, seed 42:

| Bagian | Gambar | Dipakai untuk |
|---|---|---|
| Train | 6.887 | melatih bobot model |
| Validasi | 1.476 | memilih model & menentukan `AMBANG` |
| Test | 1.476 | **hanya** untuk laporan akhir |

Aturan yang dipegang: **data test tidak pernah dilihat saat training maupun
saat memilih ambang.** Kalau test dipakai untuk memilih ambang, angka
akhirnya tidak lagi jujur — model sudah "mengintip" jawaban.

---

## 5. Alur kerja

```
data mentah (10.000)
   │  EDA: statistik, distribusi, duplikat, outlier
   │  (belum ada preprocessing piksel di sini)
   ▼
10.000 gambar terperiksa
   │  bersihkan duplikat (69 MD5 + 92 near-duplicate)
   ▼
9.839 gambar
   │  split 70/15/15
   ▼
train ────► preprocessing citra (resize 224, augmentasi, normalisasi)
                │
                ▼
             Model 1: CNN 224px ──► probabilitas + embedding
                │                        │
                │                        ├──► Model 2: SVM RBF
                │                        └──► Model 3: Random Forest
                ▼
        perbandingan 3 model di data test
                ▼
        pilih model → export TFLite float16 → tentukan AMBANG
                ▼
        model.tflite 4,56 MB + safety_rules.json
```

Model 2 dan 3 **bukan** model yang melihat gambar langsung. Mereka memakai
"embedding" — vektor 1.280 angka keluaran CNN yang merangkum isi gambar.
Analoginya: CNN sudah menuliskan deskripsi angka tentang daun itu, lalu SVM
dan Random Forest belajar memilah deskripsi tersebut.

---

## 6. Tiga algoritma yang dipakai

### 6.0 Kenapa tiga, dan kenapa justru tiga ini

**1. Satu angka tanpa pembanding tidak bisa dipercaya.**
"89%" sendirian tidak berarti apa-apa — mungkin bagus, mungkin jelek, mungkin
kebetulan. Angka baru punya makna kalau ada pembanding di data yang sama.

**2. Ketiganya mewakili tiga cara berpikir yang berbeda.**

| Keluarga algoritma | Wakil | Cara memandang data |
|---|---|---|
| Deep learning end-to-end | CNN MobileNetV2 | belajar fitur sendiri langsung dari piksel, memakai struktur spasial (bentuk, tepi, tekstur) |
| Kernel method | SVM RBF | memisahkan titik di ruang 1.280 dimensi dengan batas yang melengkung |
| Ensemble pohon | Random Forest | aturan if-then bertingkat pada fitur yang sama |

Kalau tiga pendekatan yang cara kerjanya sangat berbeda menghasilkan angka
yang berdekatan, hasilnya bukan kebetulan satu metode. Kalau salah satu jauh
tertinggal, kita tahu pendekatan itu memang tidak cocok untuk masalah ini.

**3. Model 2 dan 3 sengaja memakai embedding model 1 yang sama.**
Ini membuat perbandingannya bersih: fitur yang masuk identik, yang berbeda
hanya classifier-nya. Jadi kalau SVM menang, penyebabnya classifier, bukan
fitur — bukan karena ia "melihat" gambar yang berbeda. Tanpa penjagaan ini,
perbandingan tiga model mudah berubah jadi perbandingan tiga dataset.

**4. Yang dicari bukan pemenang kuis, tapi model yang dikirim ke HP.**
Karena itu perbandingan tidak berhenti di akurasi. Ukuran, jumlah inferensi,
dan kerumitan kode ikut dihitung — dan justru faktor-faktor itulah yang
membalikkan keputusan di bagian 7.

**Kenapa bukan algoritma lain:**

| Kandidat | Kenapa tidak dipakai |
|---|---|
| K-Nearest Neighbor | harus menyimpan seluruh data latih; 6.887 vektor × 1.280 dimensi, lambat dan besar |
| Regresi logistik | batasnya linear, terlalu lemah untuk 100 kelas citra |
| Gradient boosting (XGBoost/LightGBM) | mirip Random Forest, tidak lebih baik untuk fitur padat 1.280 dimensi, dan tetap tidak ada operator TFLite-nya |
| EfficientNet / ResNet | lebih akurat, tapi 2–10× lebih besar dari MobileNetV2 — bertabrakan langsung dengan batas 15 MB |
| YOLO / deteksi objek | masalahnya klasifikasi satu objek, bukan mencari lokasi objek |

**Urutannya juga disengaja:** CNN lebih dulu karena hanya dia yang melihat
gambar langsung. SVM dan Random Forest menyusul di atas embedding-nya, supaya
perbandingan tetap adil. Urutan ini sekaligus menjawab pertanyaan yang lebih
dalam: *apakah tugas ini butuh batas keputusan yang lebih rumit daripada
softmax biasa?* SVM dengan kernel RBF bisa membuat batas melengkung, Random
Forest bisa membuat aturan bertingkat. Kalau salah satunya jauh menang, berarti
kepala klasifikasi CNN yang sederhana memang terlalu lemah. Ternyata tidak —
lihat 7.2.

**Satu keterbatasan yang harus jujur disebut.** Model 2 dan 3 memakai
embedding dari CNN yang **sudah selesai dilatih dan dibekukan**. CNN sendiri
punya keuntungan struktural: lapisan fiturnya ikut disesuaikan bersama kepala
klasifikasinya. Jadi perbandingan ini tidak sepenuhnya setara — SVM dan RF
bermain di atas fitur yang tidak bisa mereka ubah. Konsekuensinya, kalau
CNN menang, kita tidak bisa memisahkan seberapa besar kemenangannya berasal
dari arsitektur dan seberapa besar dari keleluasaan fine-tune. Untuk keperluan
memilih model yang dikirim, ini tidak masalah (yang dibandingkan adalah
kandidat nyata, apa adanya). Untuk klaim ilmiah "CNN lebih baik daripada SVM",
ini perlu disebut sebagai catatan.

### 6.1 Model 1 - CNN MobileNetV2 (transfer learning)

**Jenis:** convolutional neural network, arsitektur MobileNetV2 yang sudah
dilatih di ImageNet (1,4 juta gambar, 1.000 kelas umum).

**Cara kerja singkat.** Lapisan awal CNN mendeteksi hal sederhana: garis,
tepi, warna. Lapisan tengah menggabungkan itu jadi bentuk: ujung daun,
tulang daun, tekstur. Lapisan akhir menggabungkannya jadi keputusan kelas.
Karena itu CNN jauh lebih kuat untuk gambar dibanding algoritma tabular.

**Kenapa transfer learning, bukan dari nol.** 6.887 gambar untuk 100 kelas
itu sangat sedikit (≈69 gambar per kelas). Melatih CNN dari nol pada data
sebanyak itu hampir pasti gagal. MobileNetV2 sudah tahu cara melihat bentuk
dari ImageNet, jadi kita hanya perlu mengajarkannya membedakan 100 spesies.

**Arsitektur yang dipakai:**

```
input 224×224×3 (piksel 0–255, RGB)
   │
Rescaling(1/127.5, offset=-1)      ← normalisasi ADA DI DALAM model
   │
MobileNetV2 (tanpa kepala klasifikasi)
   │
GlobalAveragePooling2D  → "embedding" 1.280 dimensi
   │
Dropout 0,4
   │
Dense 100 + softmax
```

Total parameter: **2.386.084**. Ukuran file `.keras`: 23,3 MB.

**Dua fase training:**

| Fase | Yang dilatih | Epoch | Learning rate |
|---|---|---|---|
| 1. Head | hanya lapisan Dense terakhir, base beku | 3 | 1e-3 |
| 2. Fine-tune | 40 lapisan terakhir MobileNetV2 (BatchNorm tetap beku) | 15 | 3e-4 |

Fase 1 membuat kepala klasifikasi "menyesuaikan diri" dulu tanpa merusak
fitur ImageNet. Fase 2 baru menyetel ulang fitur tingkat tinggi dengan
learning rate kecil, supaya tidak menghapus pengetahuan lama.

**Regularisasi** (ini yang menyelamatkan model):

- `Dropout 0,4`
- `label_smoothing 0,1` — mencegah model terlalu yakin pada satu kelas
- augmentasi: flip horizontal, rotasi ±12%, zoom 12%
- `EarlyStopping` (patience 3, restore bobot terbaik)
- `ReduceLROnPlateau` (faktor 0,5, patience 2)

**Kenapa 224 piksel padahal gambar aslinya 128.** MobileNetV2 dilatih di
ImageNet pada 224×224. Lapisan fiturnya "terbiasa" melihat objek sebesar itu.
Menaikkan input ke 224 membuat skala objek cocok dengan yang pernah dilihat
model. Yang bertambah bukan informasi baru, tapi kecocokan skala — dan itu
terbukti menaikkan akurasi 5 poin (79,95% → 85,03% pada run pertama).

**Hasil di data test:**

| Metrik | Nilai |
|---|---|
| Top-1 | **89,16%** |
| Top-3 | 95,26% |
| Macro-F1 | 88,81% |
| Fatal (beracun ditebak pangan) | 19/444 = 4,3% |
| Recall pangan | 96,29% |
| Alarm beracun palsu pada pangan | 13,93% |

**Catatan jujur:** train berakhir 99,52% sementara validasi 89,63%. Model
masih overfit — ia menghafal sebagian data latih. Ini normal untuk dataset
sekecil ini dan sudah ditekan semaksimal mungkin dengan regularisasi di atas.

### 6.2 Model 2 — SVM RBF di atas embedding CNN

**Jenis:** Support Vector Machine dengan kernel Radial Basis Function,
100 kelas one-vs-rest.

**Cara kerja singkat.** SVM mencari batas pemisah terbaik antar kelas. Kalau
data tidak bisa dipisahkan dengan garis lurus, kernel RBF "mengangkat" data
ke dimensi yang lebih tinggi secara implisit, sehingga batasnya bisa melengkung.
SVM hanya menyimpan sebagian kecil data latih sebagai **support vector** —
titik-titik yang menentukan batas itu.

**Input-nya bukan gambar**, melainkan vektor 1.280 dimensi dari lapisan
`embedding` CNN. Sebelum masuk SVM, fitur distandarkan
(`StandardScaler`) karena RBF sangat sensitif terhadap skala.

**Konfigurasi:** `C=10`, `gamma="scale"`, kernel `rbf`, `decision_function_shape="ovr"`.

**Soal probabilitas.** SVM aslinya tidak mengeluarkan probabilitas, hanya
skor keputusan. Supaya bisa dibandingkan adil dengan model lain (dan bisa
dipakai aturan ambang), skor itu diubah jadi probabilitas dengan softmax
bertemperatur `T=10`. Ini bukan probabilitas terkalibrasi sempurna, tapi cukup
untuk membandingkan tingkat keyakinan antar model.

**Hasil di data test:**

| Metrik | Nilai |
|---|---|
| Top-1 | **89,91%** |
| Top-3 | 95,66% |
| Macro-F1 | 89,66% |
| Fatal | 16/444 = 3,6% |
| Support vector | 5.468 |
| Ukuran bobot | 26,7 MB float32 / 13,3 MB float16 |

SVM adalah model **paling akurat** kalau semua foto wajib dijawab. Tapi lihat
bagian 7 — keunggulan itu hilang di skenario yang benar-benar dipakai.

### 6.3 Model 3 — Random Forest 300 pohon

**Jenis:** ensemble dari 300 decision tree, masing-masing dilatih pada sampel
data dan subset fitur yang berbeda (`max_features="sqrt"`), lalu hasilnya
divoting.

**Cara kerja singkat.** Satu decision tree mudah overfit. 300 pohon yang
melihat bagian data berbeda akan salah di tempat berbeda, dan suara mayoritas
biasanya lebih stabil daripada satu pohon. `class_weight="balanced_subsample"`
dipakai karena jumlah gambar per kelas tidak persis sama.

**Input:** embedding 1.280 dimensi yang sama dengan SVM.

**Hasil di data test:**

| Metrik | Nilai |
|---|---|
| Top-1 | 87,20% |
| Top-3 | 95,33% |
| Macro-F1 | 86,60% |
| Fatal | 19/444 = 4,3% |
| Jumlah node | 857.250 (428.775 leaf) |
| Ukuran di scikit-learn | **700 MB** |
| Ukuran format padat (perkiraan) | 175 MB |

**Kenapa tidak dipakai:** tiga alasan sekaligus. Pertama, ukurannya — 700 MB
apa adanya dari scikit-learn, dan bahkan setelah dipadatkan menjadi 175 MB
masih 11× lebih besar dari seluruh anggaran 15 MB. Kedua, TFLite tidak punya
operator untuk Random Forest, jadi 857 ribu percabangan harus ditulis manual
di Dart dan dieksekusi satu per satu. Ketiga, akurasinya justru paling rendah
pada cakupan yang dipakai aplikasi (90,7% vs CNN 92,7%), sehingga tidak ada
satu pun keunggulan yang bisa dipakai sebagai kompensasi.

---

## 7. Perbandingan dan keputusan

### 7.1 Hasil di data test yang sama (1.476 gambar)

| Model | Top-1 | Top-3 | Macro-F1 | Fatal | Alarm palsu | Ukuran di HP |
|---|---|---|---|---|---|---|
| CNN MobileNetV2 | 89,16% | 95,26% | 88,81% | 4,3% | 13,9% | **4,56 MB** |
| SVM RBF | **89,91%** | **95,66%** | **89,66%** | 3,60% | 13,7% | ±17 MB + kode Dart |
| Random Forest | 87,20% | 95,33% | 86,60% | 4,3% | 14,2% | ±175 MB (padat) |

### 7.2 Perbandingan pada cakupan yang sama

Ini bagian yang paling penting dan paling sering salah dibaca.

Tabel 7.1 membandingkan ketiga model pada **semua** foto. Padahal aplikasi
tidak begitu: ia menolak menjawab kalau ragu. Jadi pertanyaan yang benar
adalah — *pada jumlah foto yang sama-sama dijawab, siapa paling akurat?*

Dengan `AMBANG = 0.27`, aplikasi menjawab 93,4% foto. Pada cakupan itu:

| Model | Akurasi pada 93,4% foto yang dijawab |
|---|---|
| **CNN MobileNetV2** | **92,74%** |
| SVM RBF | 90,57% |
| Random Forest | 90,71% |

**Kesimpulannya berbalik.** SVM menang 0,75 poin kalau semua foto dipaksa
dijawab, tapi **kalah 2,2 poin** di kondisi yang benar-benar dipakai
pengguna. Artinya CNN bukan pilihan "kompromi karena ukuran" — CNN memang
lebih akurat di skenario nyata, sekaligus 4× lebih kecil dan cuma butuh satu
kali inferensi.

**Kenapa bisa berbalik?** Karena aturan penolakan bekerja dengan memeringkat
foto berdasarkan keyakinan, lalu membuang yang paling tidak yakin. Jadi yang
menentukan bukan cuma akurasi mentah, tapi **seberapa jujur model menilai
keyakinannya sendiri**.

- CNN dilatih bersama-sama: fiturnya ikut menyesuaikan diri dengan kepala
  softmax-nya, sehingga skor keyakinannya selaras dengan fiturnya.
- Probabilitas SVM di sini bukan probabilitas terkalibrasi, melainkan softmax
  bertemperatur di atas skor keputusan (bagian 6.2). Peringkat keyakinannya
  jadi kurang tajam, sehingga subset "paling yakin" yang dihasilkan tidak
  sebersih milik CNN.

Inilah sebabnya membandingkan model **setelah** aturan penolakan dipakai itu
wajib, bukan opsional.

### 7.3 Keputusan

**Yang dikirim ke aplikasi: CNN murni (`model.tflite` 4,56 MB).**

Alasan, berurutan:

1. **Lebih akurat pada cakupan nyata** (92,74% vs 90,57%).
2. **Ukuran.** SVM butuh 13,3 MB support vector + 3,7 MB backbone = lewat
   batas 15 MB sebelum kode aplikasi dihitung.
3. **Kerumitan.** Tidak ada operator SVM di TFLite. Kernel RBF harus ditulis
   manual di Dart, termasuk standardisasi 1.280 fitur. Itu sumber bug baru.
4. **Satu inferensi** vs dua tahap (CNN + perhitungan SVM).

SVM tetap disimpan sebagai jalur v2 kalau nanti ponsel target jauh lebih kuat,
tapi tidak ada alasan akurasi untuk itu.

### 7.4 Percobaan yang dibuang

| Percobaan | Top-1 | Kenapa tidak dipakai |
|---|---|---|
| 128px | 79,95% | gagal target 85%, overfit (train 94% vs val 79%) |
| 128px + TTA 5 view | 83,74% | 5× inferensi untuk +3,8 poin, sudah tertutup 224px |
| Aturan margin penolakan | — | diuji: tidak mengubah cakupan maupun akurasi |

TTA (test-time augmentation) adalah menjalankan satu foto dalam beberapa
versi (flip, zoom) lalu merata-ratakan hasilnya. Di desktop ia menolong, tapi
di HP artinya 5× kerja untuk keuntungan yang sudah didapat gratis dari 224px.

---

## 8. Aturan "kalau ragu jangan dijawab"

### 8.1 Cara menentukan ambang

Setiap prediksi keluar sebagai 100 probabilitas. Yang dipakai hanya satu
angka: probabilitas tertinggi (confidence).

Aturannya:

```
conf = max(probs)
if conf < AMBANG:  →  "TANAMAN TIDAK DIKENALI"
```

`AMBANG` dicari dari **data validasi**, bukan test:

- Ambang diturunkan dari 0,05 sampai 0,95.
- Untuk tiap ambang dihitung: berapa foto yang dijawab (cakupan) dan berapa
  akurasinya.
- Dipilih ambang **terendah** yang masih memberi akurasi ≥ 93% pada foto yang
  dijawab, dengan cakupan ≥ 40%.

Hasilnya: **`AMBANG = 0.27`**.

Kenapa bukan 0,40 (nilai pada run sebelumnya): model baru jauh lebih yakin,
sehingga ambang 0,40 menolak terlalu banyak foto yang sebenarnya benar.
Menurunkan ambang ke 0,27 menaikkan cakupan dari 86% ke 93% **sekaligus**
menaikkan akurasi — tidak ada trade-off yang dikorbankan.

### 8.2 Efeknya di data test

| | Validasi (pemilihan) | Test (laporan) |
|---|---|---|
| Cakupan | 94,0% | 93,4% |
| Akurasi pada yang dijawab | 93,0% | **92,7%** |
| Akurasi tanpa penolakan | 89,6% | 89,2% |
| Fatal | — | **9/410 = 2,2%** (turun dari 19) |

Jadi aplikasi menolak ±6,6% foto, dan justru itu yang membuat akurasinya naik
dari 89,2% ke 92,7%. **Penolakan bukan kegagalan — itu fitur keselamatan.**

### 8.3 Yang tidak bisa ditolong ambang

9 kesalahan fatal masih terjadi setelah ambang aktif, karena semuanya
berkeyakinan di atas 0,27:

| Aktual (beracun) | Ditebak (aman) | Confidence |
|---|---|---|
| Gynura Segetum (Daun Dewa) | Piper Sarmentosum (Cabean) | **0,65** |
| Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) | **0,57** |
| Acorus Calamus (Dlingo) | Zingiber Officinale (Jahe) | **0,56** |
| Melia Azedarach (Mindi) | Zingiber Officinale (Jahe) | 0,37 |
| Cestrum Nocturnum (Sedap Malam) | Ocimum Basilicum (Kemangi) | 0,36 |
| Gynura Segetum (Daun Dewa) | Syzygium Polyanthum (Salam) | 0,30 |
| Gynura Segetum (Daun Dewa) | Melissa officinalis (Lemon Balm) | 0,28 |
| Phytolacca Americana (Mrico Kepyar) | Ocimum Basilicum (Kemangi) | 0,27 |

**Daun Dewa menyumbang 5 dari 9 kesalahan fatal.** Karena itu aplikasi wajib
punya jaring pengaman kedua: kalau salah satu dari **top-3** kandidat punya
`level_racun ≥ 2`, tampilkan peringatan merah. Model boleh salah peringkat,
tapi kandidat beracun harus tetap terlihat.

Aturan margin (menolak kalau selisih top-1 dan top-2 terlalu kecil) sudah
diuji dan **tidak** mengubah cakupan maupun akurasi pada model ini, jadi
tidak dipakai. Menambah aturan yang tidak berefek hanya menambah kode dan
risiko bug.

---

## 9. Export TFLite

| Pemeriksaan | Hasil |
|---|---|
| File | `05_bahan_aplikasi/model.tflite` |
| Ukuran | **4,56 MB** (4.780.172 byte) — target < 15 MB |
| Kuantisasi | bobot float16, input/output tetap float32 |
| Input | `[1, 224, 224, 3]`, float32, **0..255**, RGB |
| Output | `[1, 100]` softmax, urutan = `labels.json` |
| Kesepakatan argmax Keras vs TFLite (200 gambar) | **100%** |
| Selisih probabilitas maksimum | 0,025 |
| Akurasi subset 200 gambar | Keras 90,0% vs TFLite 90,0% |

Float16 memotong ukuran hampir 5× dengan kerugian yang tidak terukur: pada
200 gambar uji, tidak ada satu pun tebakan yang berubah. Selisih probabilitas
maksimum 0,025 murni pembulatan.

**Kontrak terpenting untuk pengembang aplikasi:**

```
Normalisasi Rescaling(1/127.5, offset=-1) ADA DI DALAM model.
Kirim piksel 0–255 apa adanya. Jangan dibagi 255 lagi.
```

Kalau dibagi 255, akurasi hancur total **tanpa pesan error**. Ini kesalahan
paling umum saat memindahkan model ke aplikasi.

---

## 10. Keterbatasan dan risiko

Ditulis terbuka supaya tidak ada kejutan di kemudian hari.

1. **Angka 89%/92,7% diukur pada foto dataset, bukan foto HP.** Foto dataset
   rapi, satu objek, latar seragam. Di lapangan, akurasi hampir pasti turun.
   Belum diukur sama sekali.
2. **Resolusi sumber hanya 128×128.** Detail daun yang tipis (urat, tepi,
   bulu) sudah hilang di sumbernya. Ini plafon dataset, bukan plafon
   algoritma.
3. **Overfitting masih ada.** Train 99,5% vs validasi 89,6%.
4. **13,9% tanaman pangan memicu alarm beracun palsu.** Ini membuat pengguna
   panik dan berhenti percaya. Perlu diperbaiki di v2.
5. **Label beracun masih draf.** Semua metrik keselamatan bergantung pada
   `analisis/peta_tag_draf.csv` (pangan 51, obat 83, beracun 30). Kalau tag
   itu salah, metrik fatal juga salah.
6. **Spesies berbahaya tidak ada di dataset** — kecubung (*Datura*), oleander,
   bintaro, dieffenbachia, antiaris. Model akan memaksakan foto itu ke salah
   satu dari 100 kelas. Karena itu halaman "Daftar Mematikan" wajib ada,
   bukan pelengkap.
7. **Bahan pokok survival tidak ada di dataset**: talas, kelor, singkong.
   Padahal "Keladi" di dataset adalah *Caladium* yang beracun — nama yang
   sama dengan talas di lidah orang. Ini risiko kebingungan paling berbahaya
   di seluruh proyek.
8. **Salah nama ilmiah di dataset** yang harus dibetulkan di label aplikasi:
   `Caladium Cicolor`→bicolor, `Raulvolvia`→Rauvolfia, `Jatropa`→Jatropha,
   `Clinalanthus`→Clinacanthus, `Houttoynia`→Houttuynia, `Muraya`→Murraya,
   `Desmodium Triquitrum`→triquetrum, `Vetiveria Zizaniodes`→zizanioides,
   `Euphoria Longan`→Dimocarpus longan.

Rekomendasi: jangan rilis ke publik sebelum (a) uji foto lapangan sungguhan
minimal 100 foto, dan (b) basis pengetahuan 100 spesies selesai. Untuk
tugas/portofolio, paket ini sudah utuh dan bisa dipertanggungjawabkan.

---

## 11. Istilah yang sering ditanya

| Istilah | Arti singkat |
|---|---|
| **Top-1** | Tebakan pertama model benar. |
| **Top-3** | Jawaban benar ada di antara 3 kandidat teratas. |
| **Macro-F1** | Rata-rata F1 per kelas tanpa dibobot jumlah gambar. Adil untuk kelas kecil. |
| **Fatal** | Tanaman beracun ditebak sebagai tanaman pangan. Metrik paling penting untuk keselamatan. |
| **Alarm palsu** | Tanaman pangan ditandai beracun. Membuat pengguna panik. |
| **Cakupan (coverage)** | Persentase foto yang dijawab model (sisanya ditolak). |
| **Embedding** | Vektor angka ringkasan isi gambar; di sini 1.280 dimensi. |
| **dHash** | Sidik jari gambar berbasis pola terang-gelap, untuk deteksi duplikat. |
| **TTA** | Menjalankan satu foto dalam beberapa versi lalu merata-ratakan hasil. |
| **AMBANG** | Batas keyakinan minimum; di bawahnya model menolak menjawab. |
| **Fine-tune** | Melatih ulang sebagian model yang sudah terlatih, dengan learning rate kecil. |
| **TFLite** | Format model TensorFlow untuk perangkat mobile, tanpa server. |

---

## 12. Artefak dan cara menjalankan ulang

### Struktur folder

| Folder/File | Isi |
|---|---|
| `_data/` | `labels.json`, `train.csv`, `val.csv`, `test.csv`, `dataset_report.csv` |
| `eda/` | grafik dan statistik EDA (data mentah) |
| `01_cnn/` | model final 224px, `metrics.json`, 4 grafik, embedding, probabilitas |
| `02_svm/` | SVM, metrik, grafik |
| `03_rf/` | Random Forest, metrik, grafik |
| `04_perbandingan/` | `tabel_model.csv`, `kesalahan_fatal.csv`, 3 grafik |
| `05_bahan_aplikasi/` | **yang diserahkan**: `model.tflite`, `labels.json`, `model_meta.json`, `safety_rules.json`, `README.md` |
| `HerbaNusa_ML.ipynb` | seluruh pipeline dalam satu notebook, sudah dieksekusi |
| `RINGKASAN.md` | laporan lengkap dengan semua angka |
| `PENJELASAN_PROYEK.md` | dokumen ini |

### Menjalankan ulang

Script ada di `../scripts/`, dijalankan berurutan:

```powershell
$py = "C:\Users\Wahyu Pratomo\anaconda3\envs\ENV_001\python.exe"
$s  = "C:\Users\Wahyu Pratomo\Project Machine Learning\tanaman-obat\scripts"

& $py $s\00_eda.py              # EDA data mentah: statistik + grafik (1 menit)
& $py $s\00_prepare.py          # bersihkan + split (2 menit)
& $py $s\01_train_cnn.py --img 224 --head 3 --fine 15 --batch 16 `
      --dropout 0.4 --ls 0.1 --out 01_cnn      # 60 menit, CPU
& $py $s\02_svm.py              # 1 menit
& $py $s\03_rf.py               # 1 menit
& $py $s\04_compare.py          # 10 detik
& $py $s\05_export_tflite.py    # 12 menit (verifikasi 200 gambar)
& $py $s\06_threshold.py        # tentukan AMBANG dari validasi
& $py $s\06b_svm_deploy.py      # hitung biaya deploy SVM
```

Urutan penting: `02`–`06b` membaca artefak dari `01_cnn`, jadi latih model
dulu. `06_threshold.py` harus dijalankan setelah `01_train_cnn.py` karena
`safety_rules.json` berasal dari situ.

**Catatan soal variasi angka.** Augmentasi gambar memakai RNG, jadi run ulang
dengan kode dan seed yang sama masih bisa bergeser beberapa poin (run pertama
mendapat top-1 85,03%, run berikutnya 89,16%). Semua angka di dokumen ini
berasal dari run terakhir yang dieksekusi di notebook.

---

## 13. Ringkasan satu menit

1. **Masalah:** salah mengenali tanaman obat bisa berakibat fatal, dan
   pengetahuan itu tidak merata.
2. **Solusi:** aplikasi offline yang mengenali 100 spesies dari foto.
3. **Data:** 10.000 gambar → 9.839 setelah dibersihkan → split 70/15/15.
4. **Tiga algoritma:** CNN MobileNetV2 (langsung dari gambar), SVM RBF dan
   Random Forest (di atas embedding CNN). Tiga keluarga berbeda ini dipilih
   supaya hasilnya bukan kebetulan satu metode, dan supaya perbandingannya
   bersih — model 2 dan 3 memakai fitur yang sama persis dengan model 1.
5. **Hasil:** CNN 89,16% · SVM 89,91% · RF 87,20% pada semua foto.
6. **Keputusan:** CNN dikirim, karena pada cakupan nyata (93,4% foto dijawab)
   CNN justru paling akurat — 92,74% vs SVM 90,57% — dan hanya 4,56 MB.
7. **Keselamatan:** `AMBANG = 0.27` menolak foto yang meragukan, menaikkan
   akurasi ke 92,7% dan menekan kesalahan fatal dari 19 menjadi 9.
8. **Jujur soal batas:** angka ini dari foto dataset, bukan foto lapangan.
