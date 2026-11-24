# HerbaNusa

HerbaNusa adalah proyek mata kuliah Machine Learning untuk mengenali **100 spesies tanaman obat Indonesia** dari foto. Proyek ini mencakup pengolahan dataset, pelatihan dan perbandingan model klasifikasi gambar, serta penerapan model terpilih dalam aplikasi Android berbasis Flutter. Aplikasi memproses foto di perangkat sehingga identifikasi dapat berjalan secara offline.

## Fitur aplikasi

- **Identifikasi dari foto.** Ambil foto dengan kamera atau pilih foto dari galeri. Aplikasi menampilkan hingga tiga kandidat spesies beserta skor keyakinannya.
- **Penolakan prediksi.** Jika skor tertinggi di bawah 0,27, aplikasi menampilkan pesan bahwa tanaman belum dikenali dan memberi saran untuk mengambil foto yang lebih jelas.
- **Katalog tanaman.** Pengguna dapat menelusuri spesies dan membaca informasi tentang bagian tanaman, pemanfaatan, dan potensi risiko. Aplikasi menampilkan peringatan jika salah satu dari tiga kandidat memiliki tingkat toksisitas 2 atau lebih menurut basis pengetahuan proyek.
- **Riwayat identifikasi.** Hasil yang diterima disimpan secara lokal dan dapat dibuka kembali.

## Dataset dan alur machine learning

Dataset berisi **10.000 foto** beresolusi 128 × 128 piksel. Setelah foto duplikat dan hampir duplikat dihapus, **9.839 foto** dibagi secara stratifikasi menjadi 6.887 data latih, 1.476 data validasi, dan 1.476 data uji.

1. **Siapkan data.** Bersihkan dataset, buat pembagian data, dan simpan urutan label spesies.
2. **Latih model.** Lakukan transfer learning dan fine-tuning MobileNetV2. Embedding dari CNN yang sama dipakai untuk melatih SVM RBF dan Random Forest sebagai pembanding.
3. **Evaluasi.** Bandingkan akurasi top-1, top-3, dan macro-F1. Tentukan ambang penolakan menggunakan data validasi, lalu laporkan kinerjanya pada data uji.
4. **Ekspor ke aplikasi.** Konversi MobileNetV2 ke TensorFlow Lite dengan bobot float16. Model menerima gambar RGB 224 × 224 piksel dengan rentang nilai 0–255 dan menghasilkan skor untuk 100 kelas sesuai urutan `labels.json`. Normalisasi dilakukan di dalam model.

## Hasil evaluasi

| Model | Akurasi top-1 | Akurasi top-3 | Macro-F1 |
| --- | ---: | ---: | ---: |
| **CNN MobileNetV2** | 89,16% | 95,26% | 88,81% |
| SVM RBF | 89,91% | 95,66% | 89,66% |
| Random Forest (300 pohon) | 87,20% | 95,33% | 86,60% |

Angka dalam tabel dihitung pada 1.476 foto uji sebelum aturan penolakan diterapkan. **MobileNetV2 dipilih untuk aplikasi** karena model TFLite berukuran 4,56 MB, cukup dengan satu tahap inferensi, dan memberi akurasi tertinggi pada cakupan yang sama. Dengan ambang 0,27, model menjawab **93,36%** foto uji dan mencapai **92,74%** akurasi pada foto yang dijawab. [Tabel perbandingan](tanaman-obat/output/04_perbandingan/tabel_model.csv) · [Hasil evaluasi ambang](tanaman-obat/output/05_bahan_aplikasi/safety_rules.json)

## Struktur proyek

| Lokasi | Isi |
| --- | --- |
| [`Indonesia-Medicinal-Plant-dataset/`](Indonesia-Medicinal-Plant-dataset/) | Foto tanaman yang dikelompokkan berdasarkan spesies |
| [`tanaman-obat/scripts/`](tanaman-obat/scripts/) | Skrip persiapan data, pelatihan, evaluasi, dan ekspor model |
| [`tanaman-obat/output/`](tanaman-obat/output/) | Notebook, laporan evaluasi, grafik, dan hasil eksperimen |
| [`tanaman-obat/output/05_bahan_aplikasi/`](tanaman-obat/output/05_bahan_aplikasi/) | Model TFLite, label, spesifikasi input dan output, serta aturan penolakan |
| `herbanusa/` | Kode aplikasi Flutter, aset, dan tes |

## Menjalankan proyek

### Aplikasi Android

Siapkan Flutter SDK, Android SDK, serta emulator atau perangkat Android. Jalankan perintah berikut dari folder utama proyek.

```sh
cd herbanusa
flutter pub get
flutter run
```

Gunakan `flutter test` untuk menjalankan tes atau `flutter build apk --release` untuk membuat APK rilis. Model dan aset tersedia di `herbanusa/assets/`, sehingga aplikasi dapat dijalankan tanpa melatih ulang model.

### Pipeline machine learning

[Notebook FloraSurvival ML](tanaman-obat/output/FloraSurvival_ML.ipynb) memuat pipeline dan hasil eksperimen. Untuk menjalankan ulang pelatihan, sesuaikan path dataset dan direktori kerja di `tanaman-obat/scripts/00_prepare.py` dan `tanaman-obat/scripts/common.py`. Ikuti urutan eksekusi dan prasyarat Python, TensorFlow, serta scikit-learn di [panduan pipeline](tanaman-obat/output/README.md).

## Batasan

Hasil evaluasi menggunakan foto dari dataset proyek. Akurasi pada foto lapangan dan kemampuan mengenali spesies di luar 100 kelas belum terukur. Informasi toksisitas di aplikasi masih memerlukan verifikasi ahli. HerbaNusa ditujukan untuk pembelajaran dan identifikasi awal. Keputusan konsumsi atau pengobatan memerlukan konfirmasi dari ahli yang sesuai. [Laporan eksperimen](tanaman-obat/output/RINGKASAN.md)

