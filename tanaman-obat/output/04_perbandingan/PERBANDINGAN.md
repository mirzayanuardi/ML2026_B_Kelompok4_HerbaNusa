# Perbandingan 3 Algoritma

## Tabel perbandingan

| Model | Top-1 | Top-3 | Macro-F1 | Fatal | Alarm palsu |
|---|---|---|---|---|---|
| CNN MobileNetV2 | 89.2% | 95.3% | 88.8% | 4.3% | 13.9% |
| SVM RBF | 89.9% | 95.7% | 89.7% | 3.6% | 13.7% |
| Random Forest 300 | 87.2% | 95.3% | 86.6% | 4.3% | 14.2% |

## Pemenang

Kalau semua foto wajib dijawab, **SVM RBF** unggul di top-1
(89,9%). Tapi aplikasi ini punya aturan "kalau ragu jangan
dijawab", dan aturan itu mengubah pemenangnya.

Pada cakupan yang sama (93,4% foto dijawab), CNN justru **paling akurat**: 92,7% vs SVM 90,6% dan RF 90,7%. CNN menang setelah aturan penolakan dipakai, jadi SVM tidak perlu dibawa ke HP sama sekali.

Tabel di bawah memakai angka yang benar-benar dipakai aplikasi. Model 2 dan 3
butuh **dua tahap** di HP karena memakai embedding dari model 1 (jaringan CNN
+ perhitungan SVM/RF di Dart). Lihat `02_svm/DEPLOY.md` untuk hitungan
ukuran dan bebannya.

| Pilihan | Top-1 semua foto | Akurasi @ cakupan 93,4% | Ukuran tambahan | Kerumitan di aplikasi |
|---|---|---|---|
| CNN murni (`model.tflite`) | 89,2% | 92,7% | 4,56 MB | rendah, 1 inferensi |
| CNN + SVM | 89,9% | 90,6% | belasan MB | tinggi, kernel RBF manual |
| CNN + Random Forest | 87,2% | 90,7% | ± 171 MB | sangat tinggi, tidak layak |

**Rekomendasi paket ini: CNN murni.** Dengan `AMBANG = 0,27`, 93,4% foto dijawab dengan akurasi 92,7%, tanpa tambahan ukuran
maupun kode di HP.

## Akurasi vs cakupan

Semakin banyak foto ditolak, semakin tinggi akurasi.
Inilah dasar aturan "kalau ragu jangan dijawab".

**CNN MobileNetV2**

| Cakupan | Akurasi |
|---|---|
| 100% | 89.2% |
| 90% | 94.4% |
| 80% | 97.0% |
| 70% | 98.9% |
| 60% | 99.2% |
| 50% | 99.2% |
| 40% | 99.3% |
| 30% | 99.5% |
| 20% | 99.7% |
| 10% | 100.0% |

**SVM RBF**

| Cakupan | Akurasi |
|---|---|
| 100% | 89.9% |
| 90% | 90.7% |
| 80% | 91.4% |
| 70% | 92.0% |
| 60% | 92.9% |
| 50% | 93.2% |
| 40% | 94.7% |
| 30% | 96.4% |
| 20% | 97.3% |
| 10% | 97.3% |

**Random Forest 300**

| Cakupan | Akurasi |
|---|---|
| 100% | 87.2% |
| 90% | 92.1% |
| 80% | 94.8% |
| 70% | 97.1% |
| 60% | 98.0% |
| 50% | 98.5% |
| 40% | 99.0% |
| 30% | 99.5% |
| 20% | 99.3% |
| 10% | 99.3% |

## Contoh kesalahan fatal

| Model | Aktual (beracun) | Ditebak (aman dimakan) |
|---|---|---|
| CNN MobileNetV2 | Dioscorea Hispida (Gadung) | Kaempferia Galanga (Kencur) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) |
| CNN MobileNetV2 | Ruta Angustifolia (Inggu) | Amaranthus Spinosus (Bayam Duri) |
| CNN MobileNetV2 | Typhonium Flagelliforme (Keladi Tikus) | Cyclea Barbata (Cincau Hijau) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Melissa officinalis (Lemon Balm) |
| CNN MobileNetV2 | Melia Azedarach (Mindi) | Murraya Koenigii (Daun Kari) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Syzygium Polyanthum (Salam) |
| CNN MobileNetV2 | Alstonia Scholaris (Pulai) | Euphoria Longan (Kelengkeng) |
| CNN MobileNetV2 | Melia Azedarach (Mindi) | Zingiber Officinale (Jahe) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Cyclea Barbata (Cincau Hijau) |
| CNN MobileNetV2 | Cestrum Nocturnum (Sedap Malam) | Ocimum Americanum (Selasih) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) |
| CNN MobileNetV2 | Alstonia Scholaris (Pulai) | Euphoria Longan (Kelengkeng) |
| CNN MobileNetV2 | Cestrum Nocturnum (Sedap Malam) | Ocimum Basilicum (Kemangi) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Piper Sarmentosum (Cabean) |
| CNN MobileNetV2 | Phytolacca Americana (Mrico Kepyar) | Ocimum Basilicum (Kemangi) |
| CNN MobileNetV2 | Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) |
| CNN MobileNetV2 | Acorus Calamus (Dlingo) | Zingiber Officinale (Jahe) |
| CNN MobileNetV2 | Melia Azedarach (Mindi) | Amaranthus Spinosus (Bayam Duri) |
| SVM RBF | Dioscorea Hispida (Gadung) | Kaempferia Galanga (Kencur) |
| SVM RBF | Gynura Segetum (Daun Dewa) | Pluchea Indica (Beluntas) |
| SVM RBF | Ruta Angustifolia (Inggu) | Amaranthus Spinosus (Bayam Duri) |
| SVM RBF | Gynura Segetum (Daun Dewa) | Nepeta Cataria (Catnip) |
| SVM RBF | Dioscorea Hispida (Gadung) | Syzygium Polyanthum (Salam) |
| SVM RBF | Annona Muricata (Sirsak) | Curcuma Domestica (Kunyit) |

## Cara membaca

- **Top-1**: tebakan pertama benar.
- **Top-3**: jawaban benar ada di 3 kandidat teratas.
- **Fatal**: foto tanaman beracun ditebak sebagai spesies pangan yang aman.
  Ini metrik utama untuk keselamatan.
- **Alarm palsu**: tanaman pangan yang ditandai beracun, membuat pengguna panik.
