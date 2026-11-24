# FloraSurvival AI — Paket Serah Terima ke Pengembang Aplikasi

Dokumen ini untuk **pengembang aplikasi** (Flutter/Android). Isinya: apa yang
sudah jadi dari sisi machine learning, cara memakainya, dan aturan keselamatan
yang **wajib** dipatuhi di level kode.

Dibuat oleh: tim ML. Dataset: 100 spesies tanaman obat Indonesia, 10.000 gambar.

---

## 1. Isi paket

| File | Isi | Wajib? |
|---|---|---|
| `model.tflite` | Model siap pakai, MobileNetV2 100 kelas | ✅ |
| `labels.json` | 100 nama kelas, urutan = urutan output model | ✅ |
| `model_meta.json` | Spesifikasi input/output + hasil verifikasi numerik | ✅ |
| `safety_rules.json` | **Nilai AMBANG + aturan penolakan** | ✅ |
| `metrics_model_1.json` | Angka akurasi model | referensi |
| `README.md` | Dokumen ini | ✅ |

Folder `data/`, `docs/`, `kode_referensi/` (basis pengetahuan spesies dan
implementasi referensi logika keselamatan) **belum ada di paket ini** — itu
tahap berikutnya. Saat ini yang bisa dipakai baru model + aturan penolakan.

### Angka model (data test 1.476 gambar, tidak dipakai saat training)

| Kondisi | Foto dijawab | Akurasi pada yang dijawab |
|---|---|---|
| Tanpa penolakan | 100% | 89,2% |
| **Dengan `AMBANG = 0.27`** | **93,4%** | **92,7%** |
| Top-3 tanpa penolakan | 100% | 95,3% (jawaban benar ada di 3 teratas) |

---

## 2. Cara menjalankan model

### 2.1 Input — tepat seperti ini

```
Bentuk   : [1, 224, 224, 3]
Tipe     : float32
Rentang  : 0.0 – 255.0        ← JANGAN dibagi 255
Warna    : RGB
Resize   : bilinear
```

> **Peringatan:** normalisasi **sudah ada di dalam model**
> (`Rescaling(1/127.5, offset=-1)`). Kalau Anda membagi 255 lagi,
> akurasi akan hancur total tanpa pesan error. Ini kesalahan paling umum.

### 2.2 Output

```
Bentuk   : [1, 100]
Tipe     : float32
Isi      : probabilitas (jumlah = 1.0)
Urutan   : indeks ke-i = labels.json[i]
```

### 2.3 Alur inferensi

```dart
// pseudo-code
final image  = await decodeImage(path);          // RGB
final resized = resizeBilinear(image, 224, 224);
final input  = toFloat32Array(resized, scale: 1.0); // 0..255
final probs  = interpreter.run([1,224,224,3], input); // -> [100]

// urutkan menurun
final top3 = argSortDesc(probs, 3);
```

---

## 3. Aturan keselamatan — WAJIB, jangan diubah

### 3.1 Tolak menjawab kalau ragu

Ini inti produk. Tanpa aturan ini, aplikasi berbahaya.

```
conf = probs[top3[0]]

if (conf < 0.27) {
    tampilkan layar "TANAMAN TIDAK DIKENALI";
    return;
}
```

- **`AMBANG = 0.27`** (sudah final, ada di `safety_rules.json`). Nilai ini
  dipilih dari data validasi: ambang terendah yang memberi akurasi ≥93%
  pada foto yang dijawab.
- Efek terukur di data test: **93,4% foto dijawab, akurasi 92,7%**, dan
  kesalahan fatal turun dari 4,3% menjadi **2,2%**. Sisanya ditolak dengan
  jujur sebagai "tidak dikenali".
- **Jangan** menurunkan ambang agar "pengguna tidak kecewa". Menolak adalah
  fitur, bukan bug.
- **Jangan** memakai aturan margin sebagai penolakan — sudah diuji,
  tidak menambah akurasi sama sekali pada model ini.

### 3.2 Badge diturunkan dari spesies, bukan ditebak

AI **hanya** menjawab "tanaman apa ini". Tiga kategori (Pangan / Obat / Beracun)
dibaca dari `knowledge_base.json` berdasarkan nama spesies hasil prediksi.

```
prediksi -> nama spesies -> knowledge_base[spesies] -> badge
```

### 3.3 Matriks badge

| Kondisi | Badge utama | Badge sekunder |
|---|---|---|
| `level_racun == 3`, atau racun ada di bagian yang dikonsumsi | 🔴 DILARANG | — |
| `level_racun == 2` dan bagian bahaya ≠ bagian konsumsi | 🟠 HARUS DIOLAH | 🟢 Pangan |
| `pangan == 1`, `level_racun <= 2`, bagian bahaya ≠ bagian konsumsi | 🟢 PANGAN | 🟠 catatan bagian |
| `obat == 1`, `pangan == 0`, `level_racun <= 1` | 🟡 OBAT P3K | — |
| `beracun == 1` tapi `level_racun == 1` dan bukan pangan | 🟠 HATI-HATI | 🟡/🟢 jika ada |
| tidak cocok | ⚪ TIDAK DIKENALI | — |

Badge **tidak saling eksklusif**. Contoh: kayu putih = 🟠 (minyak tidak boleh
ditelan) + 🟡 (obat gosok).

### 3.4 Larangan teks UI

| ❌ Jangan | ✅ Pakai |
|---|---|
| "Aman dimakan" | "Bagian yang biasa dikonsumsi: buah matang" |
| "Tanaman ini obat X" | "Secara tradisional digunakan untuk X — bukan pengganti obat" |
| Menampilkan 1 jawaban saja | Selalu tampilkan top-3 dengan persentase |

### 3.5 Peringatan di top-3

Kalau **salah satu** dari 3 kandidat teratas punya `level_racun >= 2`,
tampilkan peringatan merah yang menyebut nama kandidat tersebut. Ini jaring
pengaman utama: model bisa salah peringkat, tapi kandidat beracun tetap
terlihat oleh pengguna.

### 3.6 Pasangan yang sudah diketahui sering tertukar

Ini bukan spekulasi — diukur di data test **setelah** `AMBANG = 0.27`
diterapkan. Semua baris di bawah lolos ambang, jadi `AMBANG` tidak bisa
menahannya. Pertimbangkan catatan khusus di UI untuk pasangan ini:

| Kalau pengguna memotret | Model bisa menjawab | Keyakinan | Bahaya |
|---|---|---|---|
| Daun Dewa / Gynura (beracun) | Cabean / Piper Sarmentosum | 0,65 | fatal |
| Daun Dewa / Gynura (beracun) | Beluntas / Pluchea Indica | 0,57 | fatal |
| Dlingo / Acorus Calamus (beracun) | Jahe / Zingiber Officinale | 0,56 | fatal |
| Mindi / Melia Azedarach (beracun) | Jahe / Zingiber Officinale | 0,37 | fatal |
| Sedap Malam / Cestrum (beracun) | Kemangi / Ocimum Basilicum | 0,36 | fatal |
| Daun Dewa / Gynura (beracun) | Salam / Syzygium Polyanthum | 0,30 | fatal |
| Daun Dewa / Gynura (beracun) | Lemon Balm / Melissa officinalis | 0,28 | fatal |
| Mrico Kepyar / Phytolacca (beracun) | Kemangi / Ocimum Basilicum | 0,27 | fatal |

**Daun Dewa (Gynura Segetum) adalah risiko terbesar**: 5 dari 9 kesalahan
fatal di data test adalah foto Daun Dewa yang ditebak sebagai sayur. Kalau
prediksi mengarah ke salah satu kolom kanan, tampilkan peringatan merah.

Cara paling murah menanganinya: kalau hasil prediksi adalah salah satu dari
kolom "Model bisa menjawab", tampilkan baris tambahan
*"Sering tertukar dengan: <nama beracun> — periksa manual sebelum digunakan."*

---

## 4. Layar yang harus dibuat

| Layar | Fungsi |
|---|---|
| Disclaimer | Muncul sekali di awal. Teks di `docs/APP_SPEC.md` |
| Home | Tombol scan + katalog + riwayat |
| Scanner | Kamera live + capture + torch + galeri |
| Unknown | Layar abu-abu "tidak dikenali" + tips foto lebih jelas |
| Kartu spesies | Nama ilmiah (paling menonjol), nama lokal, badge, tabel bagian aman/bahaya, P3K, kemiripan berbahaya |
| Katalog | Search + filter kategori, offline |
| Daftar mematikan | 10–15 spesies luar dataset, untuk dicocokkan manual |
| Riwayat | Log scan + catatan pengguna, lokal |

---

## 5. Checklist uji sebelum rilis

| # | Kasus | Ekspektasi |
|---|---|---|
| 1 | Foto tanaman jelas dari dataset | Dikenali, badge muncul |
| 2 | Foto meja/kursi | "Tidak dikenali" |
| 3 | Foto gelap/blur | "Tidak dikenali" + tips |
| 4 | Foto sirih (obat, bukan pangan) | Badge 🟡 saja |
| 5 | Foto jarak (Ricinus) | Badge 🔴 |
| 6 | Foto pir | 🟢 + catatan "jangan makan biji" |
| 7 | Foto gadung | 🟠 "harus diolah" |
| 8 | Mode pesawat | Semua fitur tetap jalan |
| 9 | Torch di kamar gelap | Menyala |
| 10 | 100 scan berturut-turut | Tidak crash, memori stabil |

---

## 6. Batasan yang harus ditulis di aplikasi

1. Model hanya mengenali **100 spesies**. Di luar itu, tetap akan dipaksa masuk
   ke salah satu kelas → karena itu aturan 3.1 wajib.
2. Data latih aslinya **128×128 piksel** (model menerima 224×224). Di foto
   lapangan asli akurasi turun — angka 89% diukur pada foto dataset, bukan foto HP.
3. **Bukan alat diagnosis.** Bukan pengganti dokter atau apoteker.
4. Kecubung, oleander, bintaro, dieffenbachia, antiaris **tidak ada di model** —
   hanya ada di halaman "Daftar Mematikan" untuk dicocokkan manual.

---

## 7. Yang perlu ditanyakan ke tim ML

- Isi `knowledge_base.json` (100 entri) dan `danger_list.json`
- Hasil evaluasi lengkap: `metrics_model_1.json`, `PERBANDINGAN.md`
