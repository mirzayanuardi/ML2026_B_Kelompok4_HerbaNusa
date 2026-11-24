# Biaya Deploy Model 2 (SVM) di HP

## Ringkasan

SVM adalah model dengan akurasi tertinggi (top-1 89,9%), **tetapi SVM
tidak bisa berdiri sendiri**. Dia bekerja di atas vektor 1.280 dimensi yang
dihasilkan MobileNetV2. Artinya di HP harus ada dua tahap:

1. Gambar -> TFLite MobileNetV2 (sampai lapisan `embedding`) -> 1.280 angka
2. 1.280 angka -> hitung kernel RBF ke semua support vector -> pilih kelas

| Item | Nilai |
|---|---|
| Support vector | 5,468 |
| Dimensi fitur | 1280 |
| Ukuran bobot SVM (float32) | **26,7 MB** |
| Ukuran bobot SVM (float16) | 13,3 MB |
| Ukuran backbone CNN (tanpa kepala) | ~3,7 MB |
| Evaluasi kernel per inferensi | 5,468 x 1280 |
| Beban tambahan per inferensi | ~14 MFLOP, 7 ms di laptop ini |

Konsekuensi praktis:

- **Ukuran jadi masalah utama.** 5,468 support vector x 1280 dimensi:
  float32 = **26,7 MB**, float16 = **13,3 MB**, ditambah
  backbone ~3,7 MB. Totalnya sudah lewat batas 15 MB sebelum kode aplikasi dihitung.
- **Tidak ada operator SVM di TFLite.** Kernel RBF harus ditulis manual di Dart
  (5,468 dot product 1280 dimensi untuk setiap foto). Jadi tidak bisa
  sekadar "salin `model.tflite` lalu jalan".
- Beban hitungnya sendiri masih wajar (~14 MFLOP, 7 ms di laptop
  ini), jadi yang memberatkan adalah ukuran dan kerumitan kode, bukan kecepatan.
- Kalau semua foto wajib dijawab, akurasi SVM naik 0,7 poin dibanding CNN murni — tapi keunggulan itu hilang begitu aturan penolakan dipakai.

**Keputusan paket ini:** yang dikirim ke aplikasi adalah **CNN murni**
(`model.tflite` 4,6 MB, 1 inferensi, tanpa kode tambahan). SVM disimpan
sebagai jalur peningkatan v2 kalau nanti ponsel target sudah lebih kuat,
atau dipakai di sisi server.

Yang membuat CNN tetap layak: dengan `AMBANG = 0,27`, 93,4% foto dijawab dengan akurasi **92,7%**. Pada cakupan yang sama (93,4% foto dijawab), CNN malah **lebih akurat** daripada SVM: 92,7% vs 90,6%. Jadi SVM tidak lagi punya alasan untuk dibawa ke HP.

## Catatan pengukuran

Latensi 7,1 ms/gambar diukur di laptop CPU ini,
bukan di HP. Angka itu hanya pembanding relatif, bukan janji performa.
Semua angka ukuran adalah perkiraan dari jumlah support vector x dimensi x
ukuran tipe data, tanpa kompresi.
