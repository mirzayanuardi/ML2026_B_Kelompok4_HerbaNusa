# Model 3 - Random Forest 300

## Ringkasan

| Metrik | Nilai |
|---|---|
| Top-1 | **87.2%** |
| Top-3 | **95.3%** |
| Macro-F1 | 86.6% |
| Waktu fit | 0.7 menit |
| Jumlah pohon | 300 |

## Keamanan

| Metrik | Nilai |
|---|---|
| Foto beracun di test | 444 |
| Dikenali sebagai beracun | 91.7% |
| **Kesalahan fatal** | **19 = 4.3%** |
| Recall pangan | 95.2% |
| Alarm beracun palsu pada pangan | 14.2% |

## Kalau ragu jangan dijawab

| Cakupan | n | Akurasi |
|---|---|---|
| 100% | 1476 | 87.2% |
| 90% | 1328 | 92.1% |
| 80% | 1181 | 94.8% |
| 70% | 1033 | 97.1% |
| 60% | 886 | 98.0% |
| 50% | 738 | 98.5% |
| 40% | 590 | 99.0% |
| 30% | 443 | 99.5% |
| 20% | 295 | 99.3% |
| 10% | 148 | 99.3% |

## 15 kelas terburuk

| Spesies | Recall | n |
|---|---|---|
| Dioscorea Hispida (Gadung) | 0.23 | 13 |
| Pluchea Indica (Beluntas) | 0.33 | 15 |
| Gynura Segetum (Daun Dewa) | 0.40 | 15 |
| Houttoynia Cordata (Amis-Amisan) | 0.46 | 13 |
| Murraya Koenigii (Daun Kari) | 0.50 | 14 |
| Cyclea Barbata (Cincau Hijau) | 0.53 | 15 |
| Syzygium Polyanthum (Salam) | 0.53 | 15 |
| Clinalanthus Nutans (Dandang Gendis) | 0.54 | 13 |
| Merremia Mammosa (Bidara) | 0.60 | 15 |
| Desmodium Triquitrum (Daun Duduk) | 0.60 | 15 |
| Acorus Calamus (Dlingo) | 0.60 | 15 |
| Psidium Guajava (Jambu Biji) | 0.67 | 15 |
| Pogostemon Cablin (Nilam) | 0.67 | 15 |
| Impatiens Balsamina (Pacar Air) | 0.71 | 14 |
| Piper Sarmentosum (Cabean) | 0.71 | 14 |

## Catatan

Random Forest memakai embedding dari model 1.
Kelebihannya: tahan noise label. Kekurangannya: biasanya kalah akurat dari CNN
karena tidak bisa memakai struktur spasial gambar.
