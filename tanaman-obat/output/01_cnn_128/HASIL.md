# Model 1 - CNN MobileNetV2

## Ringkasan

| Metrik | Nilai |
|---|---|
| Top-1 | **79.9%** |
| Top-3 | **91.1%** |
| Macro-F1 | 79.5% |
| Resolusi input | 128x128 |
| Parameter | 2,386,084 |
| Ukuran model | 23.3 MB |

## Keamanan

| Metrik | Nilai |
|---|---|
| Foto beracun di test | 444 |
| Dikenali sebagai beracun | 89.6% |
| **Kesalahan fatal (beracun -> spesies pangan)** | **28 = 6.3%** |
| Recall pangan | 89.9% |
| Alarm beracun palsu pada pangan | 17.6% |

## Kalau ragu jangan dijawab

| Cakupan | n | Akurasi |
|---|---|---|
| 100% | 1476 | 79.9% |
| 90% | 1328 | 85.7% |
| 80% | 1181 | 90.0% |
| 70% | 1033 | 93.7% |
| 60% | 886 | 96.0% |
| 50% | 738 | 97.7% |
| 40% | 590 | 98.5% |
| 30% | 443 | 99.1% |
| 20% | 295 | 99.3% |
| 10% | 148 | 99.3% |

## 15 kelas terburuk

| Spesies | Recall | n |
|---|---|---|
| Merremia Mammosa (Bidara) | 0.20 | 15 |
| Peperomia Pellucida (Suruhan) | 0.27 | 15 |
| Centella Asiatica (Pegagan) | 0.33 | 15 |
| Dioscorea Hispida (Gadung) | 0.38 | 13 |
| Cyclea Barbata (Cincau Hijau) | 0.40 | 15 |
| Desmodium Triquitrum (Daun Duduk) | 0.40 | 15 |
| Clinalanthus Nutans (Dandang Gendis) | 0.46 | 13 |
| Houttoynia Cordata (Amis-Amisan) | 0.46 | 13 |
| Typhonium Flagelliforme (Keladi Tikus) | 0.47 | 15 |
| Syzygium Polyanthum (Salam) | 0.47 | 15 |
| Pluchea Indica (Beluntas) | 0.47 | 15 |
| Gynura Segetum (Daun Dewa) | 0.47 | 15 |
| Impatiens Balsamina (Pacar Air) | 0.50 | 14 |
| Murraya Koenigii (Daun Kari) | 0.50 | 14 |
| Acorus Calamus (Dlingo) | 0.53 | 15 |

## 10 kelas terbaik

| Spesies | Recall |
|---|---|
| Annona Muricata (Sirsak) | 1.00 |
| Phytolacca Americana (Mrico Kepyar) | 1.00 |
| Raulvolvia Serpentina (Pule Pandak) | 1.00 |
| Ricinus Communis (Jarak) | 1.00 |
| Parameria Laevigata (Kayu Rapet) | 1.00 |
| Phaleria Macrocarpa (Mahkota Dewa) | 1.00 |
| Tithonia Diversifolia (Daun Insulin) | 1.00 |
| Pyrus Communis (Pir) | 1.00 |
| Zephyranthes Candida (Lili Hujan Putih) | 1.00 |
| Zingiber Zerumbet (Lempuyang Gajah) | 1.00 |

## Cara membaca

- **Top-1**: tebakan peringkat 1 benar.
- **Top-3**: jawaban benar ada di 3 kandidat teratas (jaring pengaman aplikasi).
- **Kesalahan fatal**: foto tanaman beracun ditebak sebagai spesies yang aman dimakan.
  Ini metrik paling penting untuk aplikasi keselamatan lapangan.
