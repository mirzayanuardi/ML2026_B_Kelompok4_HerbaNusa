import 'package:flutter/material.dart';

import '../main.dart';
import '../theme.dart';
import '../widgets/ui.dart';
import 'about_screen.dart';
import 'onboarding_screen.dart';

/// Profil: on-device stats plus links to history, the model card, and guides.
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key, required this.historyVersion});

  final int historyVersion;

  void _push(BuildContext context, Widget page) =>
      Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => page));

  void _privacy(BuildContext context) => showModalBottomSheet<void>(
    context: context,
    backgroundColor: cream,
    showDragHandle: true,
    builder: (context) => Padding(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Privasi Penggunaan', style: heading(19)),
          const SizedBox(height: 12),
          for (final line in const [
            'Foto diproses oleh model di perangkat ini. Tidak ada foto yang dikirim ke server.',
            'Foto yang dianalisis disalin ke penyimpanan aplikasi agar muncul di Riwayat. Hapus kapan saja dari halaman Riwayat.',
            'Riwayat menyimpan paling banyak 50 hasil terakhir.',
            'Aplikasi hanya meminta izin kamera atau galeri saat kamu memilih foto.',
          ])
            Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 2),
                    child: Icon(
                      Icons.check_circle_outline,
                      size: 16,
                      color: success,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      line,
                      style: const TextStyle(color: inkMuted, height: 1.5),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    ),
  );

  @override
  Widget build(BuildContext context) {
    final services = AppServices.of(context);
    final catalog = services.catalog;
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        children: [
          Text('Profil', style: heading(24)),
          const SizedBox(height: 16),
          Center(
            child: Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: accent, width: 1.5),
              ),
              child: const CircleAvatar(
                radius: 40,
                backgroundColor: mint,
                child: Icon(Icons.eco, size: 36, color: accentDeep),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Center(child: Text('Sahabat Flora', style: heading(20))),
          const SizedBox(height: 4),
          const Center(
            child: Text(
              'Semua foto diproses di perangkat ini',
              style: TextStyle(color: inkMuted, fontSize: 12),
            ),
          ),
          const SizedBox(height: 18),
          FutureBuilder(
            key: ValueKey(historyVersion),
            future: services.history.load(),
            builder: (context, snapshot) => _Stats(
              items: [
                ('${snapshot.data?.length ?? 0}', 'Identifikasi'),
                ('${catalog.plants.length}', 'Spesies'),
                (
                  '${(catalog.modelCard.accuracyAnswered * 100).round()}%',
                  'Akurasi dijawab',
                ),
              ],
            ),
          ),
          const SizedBox(height: 22),
          Text('Data & Preferensi', style: heading(15)),
          const SizedBox(height: 10),
          MenuGroup(
            children: [
              MenuRow(
                icon: Icons.insights_outlined,
                title: 'Tentang Model',
                subtitle: 'Hasil pengujian dan keterbatasan model',
                onTap: () => _push(context, const AboutScreen()),
              ),
            ],
          ),
          const SizedBox(height: 22),
          Text('Lainnya', style: heading(15)),
          const SizedBox(height: 10),
          MenuGroup(
            children: [
              MenuRow(
                icon: Icons.shield_outlined,
                title: 'Privasi Penggunaan',
                subtitle: 'Cara HerbaNusa memakai fotomu',
                onTap: () => _privacy(context),
              ),
              const MenuRow(
                icon: Icons.info_outline,
                title: 'Tentang HerbaNusa',
                subtitle: 'Versi 1.0.0 · Project ML 2026',
              ),
            ],
          ),
          const SizedBox(height: 16),
          SoftCard(
            color: mint,
            padding: const EdgeInsets.symmetric(vertical: 14),
            onTap: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (page) =>
                    OnboardingScreen(onStart: () => Navigator.of(page).pop()),
              ),
            ),
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.auto_awesome, color: accentDeep, size: 18),
                SizedBox(width: 8),
                Text(
                  'Lihat Onboarding',
                  style: TextStyle(
                    color: accentDeep,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Stats extends StatelessWidget {
  const _Stats({required this.items});

  final List<(String, String)> items;

  @override
  Widget build(BuildContext context) => SoftCard(
    padding: const EdgeInsets.symmetric(vertical: 14),
    child: IntrinsicHeight(
      child: Row(
        children: [
          for (final (index, (value, label)) in items.indexed) ...[
            if (index > 0) const VerticalDivider(width: 1, color: line),
            Expanded(
              child: Column(
                children: [
                  Text(
                    value,
                    style: heading(
                      18,
                      weight: FontWeight.w800,
                      color: accentDeep,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    label,
                    style: const TextStyle(color: inkMuted, fontSize: 11),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    ),
  );
}
