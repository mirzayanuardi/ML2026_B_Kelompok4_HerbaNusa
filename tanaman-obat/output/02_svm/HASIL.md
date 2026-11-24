# Model 2 - SVM RBF

## Ringkasan

| Metrik | Nilai |
|---|---|
| Top-1 | **89.9%** |
| Top-3 | **95.7%** |
| Macro-F1 | 89.7% |
| Waktu fit | 0.9 menit |
| Input | embedding 1.280 dimensi dari model 1 |

## Keamanan

| Metrik | Nilai |
|---|---|
| Foto beracun di test | 444 |
| Dikenali sebagai beracun | 92.8% |
| **Kesalahan fatal** | **16 = 3.6%** |
| Recall pangan | 96.2% |
| Alarm beracun palsu pada pangan | 13.7% |

## Kalau ragu jangan dijawab

| Cakupan | n | Akurasi |
|---|---|---|
| 100% | 1476 | 89.9% |
| 90% | 1328 | 90.7% |
| 80% | 1181 | 91.4% |
| 70% | 1033 | 92.0% |
| 60% | 886 | 92.9% |
| 50% | 738 | 93.2% |
| 40% | 590 | 94.7% |
| 30% | 443 | 96.4% |
| 20% | 295 | 97.3% |
| 10% | 148 | 97.3% |

## 15 kelas terburuk

| Spesies | Recall | n |
|---|---|---|
| Dioscorea Hispida (Gadung) | 0.38 | 13 |
| Cyclea Barbata (Cincau Hijau) | 0.53 | 15 |
| Gynura Segetum (Daun Dewa) | 0.53 | 15 |
| Houttoynia Cordata (Amis-Amisan) | 0.54 | 13 |
| Murraya Koenigii (Daun Kari) | 0.57 | 14 |
| Merremia Mammosa (Bidara) | 0.60 | 15 |
| Syzygium Polyanthum (Salam) | 0.60 | 15 |
| Clinalanthus Nutans (Dandang Gendis) | 0.62 | 13 |
| Pluchea Indica (Beluntas) | 0.67 | 15 |
| Desmodium Triquitrum (Daun Duduk) | 0.67 | 15 |
| Piper Sarmentosum (Cabean) | 0.71 | 14 |
| Anredera Cordifolia (Binahong) | 0.73 | 15 |
| Pogostemon Cablin (Nilam) | 0.73 | 15 |
| Acorus Calamus (Dlingo) | 0.73 | 15 |
| Ficus Septica (Awar-Awar) | 0.73 | 15 |

## Catatan

SVM memakai embedding dari model 1, bukan gambar langsung.
Kalau model 1 lemah, SVM tidak bisa lebih pintar dari fitur yang diberi.
Probabilitas di sini adalah softmax bertemperatur (T=10) di atas decision_function,
hanya untuk membuat kurva abstain sebanding dengan model lain.
