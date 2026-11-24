# Model TFLite untuk Aplikasi

## Isi folder

| File | Isi |
|---|---|
| `model.tflite` | Model siap pakai, input float32 0..255 ukuran 1x224x224x3 |
| `labels.json` | 100 nama kelas, urutan sama dengan output model |
| `model_meta.json` | Spesifikasi input/output + hasil verifikasi |
| `metrics_model_1.json` | Angka akurasi model |
| `safety_rules.json` | Nilai AMBANG + aturan penolakan |

## Cara pakai di Flutter

Tambahkan `tflite_flutter` di `pubspec.yaml`, taruh `model.tflite` dan
`labels.json` di folder `assets/`.

Alur inferensi:

1. Ambil gambar (kamera/galeri), decode jadi RGB.
2. Resize ke **224x224** (bilinear).
3. Konversi ke float32 dengan nilai **0..255** — JANGAN dibagi 255.
   Normalisasi sudah ada di dalam model (`Rescaling(1/127.5, offset=-1)`).
4. Bentuk tensor `[1, 224, 224, 3]`, jalankan interpreter.
5. Output `[1, 100]` = probabilitas, urutannya sesuai `labels.json`.
6. Ambil 3 nilai terbesar -> top-3 kandidat.
7. Tolak menjawab kalau probabilitas tertinggi < `AMBANG` (lihat `safety_rules.json`).

## Verifikasi

| Pemeriksaan | Hasil |
|---|---|
| Kesepakatan argmax Keras vs TFLite (200 gambar) | 100.00% |
| Selisih probabilitas maksimum | 0.02470 |
| Ukuran model | 4.56 MB |

Kalau teman Anda mendapat hasil berbeda jauh di HP, kemungkinan besar
preprocessing-nya tidak sama (resize atau rentang nilai).

## Catatan penting

- Model ini **hanya** mengenali 100 spesies di `labels.json`.
- Tanaman di luar daftar tetap akan dipaksa masuk ke salah satu kelas,
  karena itu aturan penolakan berbasis confidence WAJIB dipakai.
- Jangan menampilkan satu jawaban saja; selalu tampilkan top-3.
