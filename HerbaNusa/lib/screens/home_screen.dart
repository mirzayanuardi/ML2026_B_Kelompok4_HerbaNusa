import 'package:flutter/material.dart';

import '../main.dart';
import '../theme.dart';
import '../widgets/ui.dart';
import 'encyclopedia_screen.dart';

/// Beranda: greeting, scan shortcut, counts, featured species, poison reminder.
class HomeScreen extends StatelessWidget {
  const HomeScreen({
    super.key,
    required this.historyVersion,
    required this.onSelectTab,
  });

  /// Bumped whenever history changes, so the count reloads.
  final int historyVersion;
  final ValueChanged<int> onSelectTab;

  static const _featured = [
    'Curcuma Domestica (Kunyit)',
    'Zingiber Officinale (Jahe)',
    'Piper Betle (Sirih)',
    'Aloe Vera (Lidah Buaya)',
    'Gynura Segetum (Daun Dewa)',
    'Ricinus Communis (Jarak)',
  ];

  @override
  Widget build(BuildContext context) {
    final services = AppServices.of(context);
    final catalog = services.catalog;
    final featured = [for (final name in _featured) ?catalog.plants[name]];
    final poisonous = catalog.plants.values.where((p) => p.isPoisonous).length;
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        children: [
          Row(
            children: [
              Expanded(child: Text('HerbaNusa', style: heading(28))),
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: mint,
                  border: Border.all(color: Colors.white, width: 2),
                ),
                child: const Icon(Icons.eco, color: accentDeep, size: 22),
              ),
            ],
          ),
          const SizedBox(height: 18),
          _HeroCard(onTap: () => onSelectTab(2)),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: FutureBuilder(
                  key: ValueKey(historyVersion),
                  future: services.history.load(),
                  builder: (context, snapshot) => _StatCard(
                    icon: Icons.history,
                    color: mint,
                    value: '${snapshot.data?.length ?? 0}',
                    label: 'Teridentifikasi',
                    onTap: () => onSelectTab(3),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _StatCard(
                  icon: Icons.local_florist_outlined,
                  color: butter,
                  value: '${catalog.plants.length}',
                  label: 'Spesies',
                  onTap: () => onSelectTab(1),
                ),
              ),
            ],
          ),
          const SizedBox(height: 22),
          SectionHeader(
            'Jelajahi Tanaman',
            action: 'Lihat semua',
            onAction: () => onSelectTab(1),
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 200,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: featured.length,
              separatorBuilder: (_, _) => const SizedBox(width: 10),
              itemBuilder: (context, index) => SizedBox(
                width: 140,
                child: PlantCard(plant: featured[index]),
              ),
            ),
          ),
          const SizedBox(height: 22),
          const SectionHeader('Keselamatan'),
          const SizedBox(height: 8),
          SoftCard(
            color: dangerSoft,
            padding: const EdgeInsets.all(14),
            onTap: () => onSelectTab(1),
            child: Row(
              children: [
                const IconBadge(
                  icon: Icons.warning_amber_rounded,
                  color: Colors.white,
                  iconColor: danger,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Overline('Waspada', color: danger),
                      const SizedBox(height: 3),
                      Text(
                        '$poisonous spesies di katalog beracun',
                        style: const TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 2),
                      const Text(
                        'Beberapa mirip sayur atau bumbu. Cek kartunya sebelum dikonsumsi.',
                        style: TextStyle(color: inkMuted, fontSize: 11),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right, color: ink),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroCard extends StatelessWidget {
  const _HeroCard({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      borderRadius: BorderRadius.circular(24),
      clipBehavior: Clip.antiAlias,
      child: Ink(
        decoration: const BoxDecoration(
          gradient: LinearGradient(colors: heroGradient),
        ),
        child: InkWell(
          key: const Key('hero_scan'),
          onTap: onTap,
          child: SizedBox(
            height: 158,
            child: Stack(
              children: [
                Positioned(
                  right: -24,
                  top: 10,
                  bottom: -10,
                  child: ClipOval(
                    child: Opacity(
                      opacity: 0.55,
                      child: Image.asset(
                        plantPhotoAsset('Curcuma Domestica (Kunyit)'),
                        width: 170,
                        fit: BoxFit.cover,
                        filterQuality: FilterQuality.medium,
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Pill(
                        '✦ Pengenalan foto',
                        color: Color(0x24FFFFFF),
                        textColor: Colors.white,
                      ),
                      const SizedBox(height: 10),
                      Text(
                        'Tanaman apa ini?',
                        style: heading(
                          20,
                          weight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 4),
                      const SizedBox(
                        width: 190,
                        child: Text(
                          'Foto daunnya, lalu lihat apakah tanaman itu pangan, obat, atau beracun.',
                          style: TextStyle(
                            color: Color(0xE6FFFFFF),
                            fontSize: 11.5,
                          ),
                        ),
                      ),
                      const Spacer(),
                      const Row(
                        children: [
                          Text(
                            'Identifikasi Tanaman',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              fontSize: 12,
                            ),
                          ),
                          SizedBox(width: 6),
                          Icon(
                            Icons.photo_camera_outlined,
                            color: Colors.white,
                            size: 16,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({
    required this.icon,
    required this.color,
    required this.value,
    required this.label,
    required this.onTap,
  });

  final IconData icon;
  final Color color;
  final String value;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => SoftCard(
    padding: const EdgeInsets.all(12),
    onTap: onTap,
    child: Row(
      children: [
        IconBadge(icon: icon, color: color, size: 38),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(value, style: heading(17, weight: FontWeight.w800)),
              Text(
                label,
                style: const TextStyle(color: inkMuted, fontSize: 11),
              ),
            ],
          ),
        ),
        const Icon(Icons.chevron_right, color: inkFaint, size: 20),
      ],
    ),
  );
}
