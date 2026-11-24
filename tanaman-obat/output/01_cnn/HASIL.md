# Model 1 - CNN MobileNetV2

## Ringkasan

| Metrik | Nilai |
|---|---|
| Top-1 | **89.2%** |
| Top-3 | **95.3%** |
| Macro-F1 | 88.8% |
| Resolusi input | 224x224 |
| Parameter | 2,386,084 |
| Ukuran model | 23.3 MB |

## Keamanan

| Metrik | Nilai |
|---|---|
| Foto beracun di test | 444 |
| Dikenali sebagai beracun | 92.3% |
| **Kesalahan fatal (beracun -> spesies pangan)** | **19 = 4.3%** |
| Recall pangan | 96.3% |
| Alarm beracun palsu pada pangan | 13.9% |

## Kalau ragu jangan dijawab

| Cakupan | n | Akurasi |
|---|---|---|
| 100% | 1476 | 89.2% |
| 90% | 1328 | 94.4% |
| 80% | 1181 | 97.0% |
| 70% | 1033 | 98.9% |
| 60% | 886 | 99.2% |
| 50% | 738 | 99.2% |
| 40% | 590 | 99.3% |
| 30% | 443 | 99.5% |
| 20% | 295 | 99.7% |
| 10% | 148 | 100.0% |

## 15 kelas terburuk

| Spesies | Recall | n |
|---|---|---|
| Dioscorea Hispida (Gadung) | 0.38 | 13 |
| Gynura Segetum (Daun Dewa) | 0.47 | 15 |
| Merremia Mammosa (Bidara) | 0.47 | 15 |
| Acorus Calamus (Dlingo) | 0.53 | 15 |
| Cyclea Barbata (Cincau Hijau) | 0.53 | 15 |
| Syzygium Polyanthum (Salam) | 0.53 | 15 |
| Clinalanthus Nutans (Dandang Gendis) | 0.54 | 13 |
| Murraya Koenigii (Daun Kari) | 0.57 | 14 |
| Pluchea Indica (Beluntas) | 0.60 | 15 |
| Houttoynia Cordata (Amis-Amisan) | 0.62 | 13 |
| Desmodium Triquitrum (Daun Duduk) | 0.67 | 15 |
| Ficus Septica (Awar-Awar) | 0.73 | 15 |
| Melia Azedarach (Mindi) | 0.73 | 15 |
| Graptophyllum Pictum (Daun Ungu) | 0.73 | 15 |
| Piper Sarmentosum (Cabean) | 0.79 | 14 |

## 10 kelas terbaik

| Spesies | Recall |
|---|---|
| Raulvolvia Serpentina (Pule Pandak) | 1.00 |
| Pyrus Communis (Pir) | 1.00 |
| Rosmarinus Officinalis (Rosemary) | 1.00 |
| Ricinus Communis (Jarak) | 1.00 |
| Tithonia Diversifolia (Daun Insulin) | 1.00 |
| Olea Europaea (Zaitun) | 1.00 |
| Tinospora Tuberculata (Brotowali) | 1.00 |
| Vitex Trifolia (Legundi) | 1.00 |
| Zephyranthes Candida (Lili Hujan Putih) | 1.00 |
| Zingiber Zerumbet (Lempuyang Gajah) | 1.00 |

## Cara membaca

- **Top-1**: tebakan peringkat 1 benar.
- **Top-3**: jawaban benar ada di 3 kandidat teratas (jaring pengaman aplikasi).
- **Kesalahan fatal**: foto tanaman beracun ditebak sebagai spesies yang aman dimakan.
  Ini metrik paling penting untuk aplikasi keselamatan lapangan.
