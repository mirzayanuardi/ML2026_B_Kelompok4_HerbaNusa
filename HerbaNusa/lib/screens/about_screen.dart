import 'package:flutter/material.dart';

import '../main.dart';
import '../theme.dart';
import '../widgets/ui.dart';

String _pct(double value) => '${(value * 100).toStringAsFixed(1)}%';

/// Model card: measured test results and the limits the handoff says to state.
class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final card = AppServices.of(context).catalog.modelCard;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const PageHeader(title: 'Tentang Model'),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
                children: [
                  Text(
                    'MobileNetV2 yang dilatih pada ${card.trainImages} foto dari dataset Indonesia Medicinal Plant '
                    '(100 spesies), lalu diuji pada ${card.testImages} foto yang tidak pernah dipakai saat pelatihan.',
                    style: const TextStyle(color: inkMuted, height: 1.5),
                  ),
                  const SizedBox(height: 16),
                  StatStrip(
                    color: Colors.white,
                    items: [
                      (_pct(card.top1), 'Top-1'),
                      (_pct(card.top3), 'Top-3'),
                      (card.macroF1.toStringAsFixed(2), 'Macro-F1'),
                    ],
                  ),
                  const SizedBox(height: 22),
                  Text('Aturan penolakan', style: heading(17)),
                  const SizedBox(height: 8),
                  Text(
                    'Jika keyakinan tertinggi di bawah ${(card.threshold * 100).round()}%, aplikasi menolak menjawab. '
                    'Pada data uji, ${_pct(card.coverage)} foto tetap dijawab dengan akurasi ${_pct(card.accuracyAnswered)}. '
                    'Tanaman beracun yang ditebak sebagai pangan turun dari ${_pct(card.fatalRate)} menjadi ${_pct(card.fatalRateAnswered)}.',
                    style: const TextStyle(color: inkMuted, height: 1.5),
                  ),
                  const SizedBox(height: 22),
                  Text('Keterbatasan', style: heading(17)),
                  const SizedBox(height: 8),
                  for (final line in const [
                    'Model hanya mengenal 100 spesies. Tanaman lain tetap dipaksa masuk ke salah satu kelas, karena itu ada aturan penolakan.',
                    'Foto latih berukuran 128×128 piksel. Akurasi pada foto HP di lapangan bisa lebih rendah dari angka di atas.',
                    'Badge Pangan, Obat, dan Beracun dibaca dari basis pengetahuan berdasarkan nama spesies, bukan ditebak oleh AI.',
                    'Bukan alat diagnosis dan bukan pengganti dokter atau apoteker.',
                    'Kecubung, oleander, bintaro, dieffenbachia, dan antiaris tidak ada di model.',
                  ])
                    Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Padding(
                            padding: EdgeInsets.only(top: 2),
                            child: Icon(Icons.info_outline, size: 16, color: warning),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(line, style: const TextStyle(height: 1.5)),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
