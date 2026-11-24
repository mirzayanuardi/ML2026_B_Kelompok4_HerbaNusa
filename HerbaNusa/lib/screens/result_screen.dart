import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../main.dart';
import '../models/analysis.dart';
import '../models/plant.dart';
import '../theme.dart';
import '../widgets/plant_sections.dart';
import '../widgets/ui.dart';
import 'encyclopedia_screen.dart';

String _pct(double value) =>
    '${(value * 100).toStringAsFixed(value >= 0.995 ? 0 : 1)}%';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.imagePath});

  final String imagePath;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  Future<Analysis?>? _analysis;
  bool _saved = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _analysis ??= AppServices.of(context).classifier.analyze(widget.imagePath);
  }

  void _saveOnce(Analysis analysis) {
    if (_saved) return;
    _saved = true;
    AppServices.of(context).history.add(
      sourcePath: widget.imagePath,
      species: analysis.best.name,
      probability: analysis.best.probability,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const PageHeader(title: 'Hasil Identifikasi'),
            Expanded(
              child: FutureBuilder<Analysis?>(
                future: _analysis,
                builder: (context, snapshot) {
                  final loading =
                      snapshot.connectionState != ConnectionState.done;
                  final analysis = snapshot.data;
                  final failed =
                      !loading && (snapshot.hasError || analysis == null);
                  final accepted = analysis != null && !analysis.rejected;
                  if (accepted) _saveOnce(analysis);

                  return ListView(
                    padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
                    children: [
                      _Photo(path: widget.imagePath, done: accepted),
                      const SizedBox(height: 12),
                      if (loading) const _Loading(),
                      if (failed)
                        const _Message(
                          icon: Icons.broken_image_outlined,
                          title: 'Foto tidak bisa dibaca',
                          body: 'Coba gunakan foto lain dengan format JPG atau PNG.',
                        ),
                      if (analysis != null && analysis.rejected)
                        _Unknown(analysis: analysis),
                      if (accepted) _Result(analysis: analysis),
                    ],
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Photo extends StatelessWidget {
  const _Photo({required this.path, required this.done});

  final String path;
  final bool done;

  @override
  Widget build(BuildContext context) => ClipRRect(
    borderRadius: BorderRadius.circular(24),
    child: AspectRatio(
      aspectRatio: 4 / 3,
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.file(File(path), fit: BoxFit.cover),
          if (done)
            Align(
              alignment: const Alignment(0, 0.88),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 7,
                ),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.94),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.check, size: 16, color: success),
                    SizedBox(width: 6),
                    Text(
                      'Berhasil dianalisis',
                      style: TextStyle(
                        color: success,
                        fontWeight: FontWeight.w700,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    ),
  );
}

class _Loading extends StatelessWidget {
  const _Loading();

  @override
  Widget build(BuildContext context) => const Padding(
    padding: EdgeInsets.symmetric(vertical: 32),
    child: Column(
      children: [
        CircularProgressIndicator(),
        SizedBox(height: 12),
        Text('Menganalisis foto...', style: TextStyle(color: inkMuted)),
      ],
    ),
  );
}

class _Message extends StatelessWidget {
  const _Message({required this.icon, required this.title, required this.body});

  final IconData icon;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) => SoftCard(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            IconBadge(icon: icon, size: 38),
            const SizedBox(width: 12),
            Expanded(child: Text(title, style: heading(16))),
          ],
        ),
        const SizedBox(height: 10),
        Text(body, style: const TextStyle(color: inkMuted, height: 1.5)),
      ],
    ),
  );
}

/// Stage 1 refusal: the model is below the confidence threshold.
class _Unknown extends StatelessWidget {
  const _Unknown({required this.analysis});

  final Analysis analysis;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      SoftCard(
        key: const Key('unknown_card'),
        color: neutralSoft,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const StatusBadge(status: SafetyStatus.unknown),
            const SizedBox(height: 12),
            Text('Tanaman tidak dikenali', style: heading(20)),
            const SizedBox(height: 8),
            Text(
              'Keyakinan tertinggi model hanya ${_pct(analysis.best.probability)}, di bawah batas '
              '${(analysis.threshold * 100).round()}%. Agar tidak memberi jawaban yang salah, '
              'HerbaNusa tidak menebak nama tanaman ini.',
              style: const TextStyle(color: inkMuted, height: 1.5),
            ),
            const SizedBox(height: 8),
            const Text(
              'Model hanya mengenal 100 spesies. Tanaman di luar daftar sering ditolak, '
              'tetapi bisa juga tertebak sebagai spesies lain.',
              style: TextStyle(color: inkMuted, height: 1.5, fontSize: 12),
            ),
          ],
        ),
      ),
      const SizedBox(height: 16),
      Text('Tips foto lebih jelas', style: heading(16)),
      const SizedBox(height: 8),
      for (final (icon, tip) in const [
        (
          Icons.wb_sunny_outlined,
          'Foto di tempat terang, hindari bayangan dan cahaya dari belakang.',
        ),
        (
          Icons.center_focus_strong_outlined,
          'Dekatkan kamera hingga daun memenuhi bingkai dan fokus tajam.',
        ),
        (
          Icons.eco_outlined,
          'Foto satu jenis tanaman saja, dengan latar belakang polos.',
        ),
      ])
        Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(
            children: [
              IconBadge(icon: icon, size: 34),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  tip,
                  style: const TextStyle(fontSize: 12.5, height: 1.4),
                ),
              ),
            ],
          ),
        ),
      const SizedBox(height: 8),
      FilledButton.icon(
        onPressed: () => Navigator.of(context).pop(),
        icon: const Icon(Icons.document_scanner_outlined, size: 20),
        label: const Text('Coba Foto Lain'),
      ),
    ],
  );
}

/// Stage 2: species matched to the knowledge base.
class _Result extends StatelessWidget {
  const _Result({required this.analysis});

  final Analysis analysis;

  @override
  Widget build(BuildContext context) {
    final catalog = AppServices.of(context).catalog;
    final plants = catalog.plants;
    final plant = plants[analysis.best.name]!;
    final dangerous = catalog.dangerousIn(analysis);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (dangerous.isNotEmpty) ...[
          DangerBanner(
            key: const Key('top3_warning'),
            title: 'PERINGATAN: kandidat beracun di 3 teratas',
            lines: [
              for (final (score, candidate) in dangerous)
                '${candidate.local} (${candidate.scientific}), ${_pct(score.probability)}: ${candidate.status.label}.',
              'Model bisa salah urutan. Jangan konsumsi atau oleskan tanaman ini sebelum dipastikan oleh orang yang ahli.',
            ],
          ),
          const SizedBox(height: 12),
        ],
        _TopCard(score: analysis.best, plant: plant),
        const SizedBox(height: 18),
        Text('3 kandidat teratas', style: heading(18)),
        const SizedBox(height: 4),
        for (final score in analysis.top)
          _CandidateRow(
            plant: plants[score.name]!,
            probability: score.probability,
          ),
        const SizedBox(height: 6),
        const Text(
          'Persentase adalah keyakinan model berdasarkan kemiripan visual.',
          style: TextStyle(color: inkMuted, fontSize: 11),
        ),
        const SizedBox(height: 22),
        PlantSafetySections(plant: plant),
        const SizedBox(height: 18),
        Row(
          children: [
            Expanded(
              flex: 2,
              child: OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                  foregroundColor: ink,
                  side: const BorderSide(color: line),
                ),
                onPressed: () => openPlant(context, plant),
                icon: const Icon(Icons.local_florist_outlined, size: 20),
                label: const Text('Kartu'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              flex: 3,
              child: FilledButton.icon(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.document_scanner_outlined, size: 20),
                label: const Text('Identifikasi Lagi'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        const Text(
          'Hasil ini perkiraan dari foto, bukan alat diagnosis dan bukan pengganti dokter atau apoteker. '
          'Akurasi pada foto lapangan bisa lebih rendah daripada pada data uji.',
          style: TextStyle(color: inkMuted, fontSize: 11, height: 1.5),
        ),
      ],
    );
  }
}

class _TopCard extends StatelessWidget {
  const _TopCard({required this.score, required this.plant});

  final PlantScore score;
  final Plant plant;

  @override
  Widget build(BuildContext context) => SoftCard(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Overline('Hasil teratas'),
                  const SizedBox(height: 6),
                  Text(
                    plant.scientific,
                    key: const Key('top_prediction'),
                    style: heading(22).copyWith(fontStyle: FontStyle.italic),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    plant.local,
                    style: const TextStyle(
                      color: inkMuted,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            _Ring(value: score.probability),
          ],
        ),
        const SizedBox(height: 14),
        StatusBadge(status: plant.status),
        const SizedBox(height: 6),
        Text(
          plant.status.description,
          style: const TextStyle(color: inkMuted, fontSize: 12),
        ),
        const SizedBox(height: 12),
        CategoryChips(plant: plant),
      ],
    ),
  );
}

/// Circular percentage, as on MeowScan's result card.
class _Ring extends StatelessWidget {
  const _Ring({required this.value});

  final double value;

  @override
  Widget build(BuildContext context) => SizedBox(
    width: 64,
    height: 64,
    child: CustomPaint(
      painter: _RingPainter(value),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '${(value * 100).round()}%',
              style: heading(16, weight: FontWeight.w800),
            ),
            const Text(
              'yakin',
              style: TextStyle(
                color: inkMuted,
                fontSize: 9,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    ),
  );
}

class _RingPainter extends CustomPainter {
  const _RingPainter(this.value);

  final double value;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = (Offset.zero & size).deflate(4);
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, 0, 2 * math.pi, false, stroke..color = sand);
    canvas.drawArc(
      rect,
      -math.pi / 2,
      2 * math.pi * value.clamp(0, 1),
      false,
      stroke..color = accent,
    );
  }

  @override
  bool shouldRepaint(_RingPainter old) => old.value != value;
}

class _CandidateRow extends StatelessWidget {
  const _CandidateRow({required this.plant, required this.probability});

  final Plant plant;
  final double probability;

  @override
  Widget build(BuildContext context) => InkWell(
    onTap: () => openPlant(context, plant),
    child: Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: line)),
      ),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: SizedBox(
              width: 44,
              height: 44,
              child: PlantPhoto(plant: plant),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  plant.local,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
                Text(
                  plant.scientific,
                  style: const TextStyle(
                    color: inkMuted,
                    fontSize: 11,
                    fontStyle: FontStyle.italic,
                  ),
                ),
                const SizedBox(height: 4),
                StatusBadge(status: plant.status, compact: true),
              ],
            ),
          ),
          Text(
            _pct(probability),
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13),
          ),
        ],
      ),
    ),
  );
}
