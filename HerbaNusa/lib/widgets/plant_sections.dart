import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/plant.dart';
import '../theme.dart';
import 'ui.dart';

(Color, Color, IconData) statusStyle(SafetyStatus status) => switch (status) {
  SafetyStatus.dilarang => (danger, dangerSoft, Icons.dangerous_outlined),
  SafetyStatus.harusDiolah => (
    warning,
    warningSoft,
    Icons.soup_kitchen_outlined,
  ),
  SafetyStatus.pangan => (success, successSoft, Icons.restaurant_outlined),
  SafetyStatus.hatiHati => (warning, warningSoft, Icons.warning_amber_rounded),
  SafetyStatus.obat => (herbal, herbalSoft, Icons.healing_outlined),
  SafetyStatus.unknown => (neutral, neutralSoft, Icons.help_outline),
};

(Color, Color, IconData) categoryStyle(PlantCategory category) =>
    switch (category) {
      PlantCategory.pangan => (success, successSoft, Icons.restaurant_outlined),
      PlantCategory.obat => (herbal, herbalSoft, Icons.healing_outlined),
      PlantCategory.beracun => (danger, dangerSoft, Icons.coronavirus_outlined),
    };

/// Large status badge: "DILARANG", "PANGAN", ...
class StatusBadge extends StatelessWidget {
  const StatusBadge({super.key, required this.status, this.compact = false});

  final SafetyStatus status;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final (color, soft, icon) = statusStyle(status);
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 8 : 12,
        vertical: compact ? 4 : 8,
      ),
      decoration: BoxDecoration(
        color: soft,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: compact ? 12 : 16, color: color),
          SizedBox(width: compact ? 4 : 6),
          Text(
            status.label,
            style: TextStyle(
              color: color,
              fontSize: compact ? 9.5 : 12,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.6,
            ),
          ),
        ],
      ),
    );
  }
}

/// Every category of a species; a plant can be food, medicine and poisonous at once.
class CategoryChips extends StatelessWidget {
  const CategoryChips({super.key, required this.plant});

  final Plant plant;

  @override
  Widget build(BuildContext context) => Wrap(
    spacing: 6,
    runSpacing: 6,
    children: [
      for (final category in PlantCategory.values)
        if (plant.categories.contains(category))
          () {
            final (color, soft, icon) = categoryStyle(category);
            return Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: soft,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, size: 14, color: color),
                  const SizedBox(width: 5),
                  Text(
                    category.label,
                    style: TextStyle(
                      color: color,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            );
          }(),
    ],
  );
}

/// Red warning box used for poisonous candidates and known mix-ups.
class DangerBanner extends StatelessWidget {
  const DangerBanner({
    super.key,
    required this.title,
    required this.lines,
    this.color = danger,
    this.soft = dangerSoft,
  });

  final String title;
  final List<String> lines;
  final Color color;
  final Color soft;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: soft,
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: color.withValues(alpha: 0.5), width: 1.5),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.warning_rounded, color: color, size: 26),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.w800,
                  fontSize: 14,
                ),
              ),
              for (final line in lines) ...[
                const SizedBox(height: 4),
                Text(
                  line,
                  style: const TextStyle(
                    color: ink,
                    fontSize: 12.5,
                    height: 1.45,
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    ),
  );
}

class PlantSafetySections extends StatelessWidget {
  const PlantSafetySections({super.key, required this.plant});

  final Plant plant;

  @override
  Widget build(BuildContext context) {
    final groups = <String, List<PlantFinding>>{};
    for (final finding in plant.findings) {
      groups.putIfAbsent(_section(finding.heading), () => []).add(finding);
    }
    final sources = {
      for (final finding in plant.findings)
        for (final source in finding.sources) source.url: source,
    }.values;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text('Bagian aman & berbahaya', style: heading(18)),
        const SizedBox(height: 10),
        _PartsTable(plant: plant),
        const SizedBox(height: 14),
        Text('Penjelasan', style: heading(16)),
        const SizedBox(height: 8),
        SoftCard(
          padding: EdgeInsets.zero,
          child: Column(
            children: [
              for (final (index, section) in const [
                ('Pangan', Icons.restaurant_outlined, success),
                ('Yang perlu dihindari', Icons.warning_amber_rounded, danger),
                ('Penggunaan tradisional', Icons.healing_outlined, herbal),
                ('Informasi lain', Icons.local_florist_outlined, inkMuted),
              ].where((section) => groups.containsKey(section.$1)).indexed) ...[
                if (index > 0) const Divider(height: 1, color: line),
                Padding(
                  padding: const EdgeInsets.all(14),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(section.$2, color: section.$3, size: 20),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(section.$1, style: heading(14)),
                            const SizedBox(height: 4),
                            Text(
                              groups[section.$1]!
                                  .map((finding) => finding.text)
                                  .toSet()
                                  .join(' '),
                              style: const TextStyle(
                                fontSize: 13,
                                height: 1.45,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
        if (plant.lookAlikes.isNotEmpty) ...[
          const SizedBox(height: 10),
          DangerBanner(
            title: 'Spesies yang mirip',
            lines: [for (final alike in plant.lookAlikes) alike.name],
          ),
        ],
        const SizedBox(height: 12),
        Text('Bantuan medis', style: heading(16)),
        const Text(
          'Jika terjadi kegawatdaruratan medis di Indonesia, hubungi 119.',
        ),
        const SizedBox(height: 10),
        ExpansionTile(
          title: const Text('Sumber informasi'),
          children: [
            for (final source in sources)
              _Citation(
                title: source.title,
                date: source.date,
                url: source.url,
              ),
            _Citation(
              title: 'Kementerian Kesehatan RI — Layanan darurat medis 119',
              date: '2024',
              url: Uri.parse(
                'https://kemkes.go.id/id/akses-darurat-medis-119-kini-bisa-melalui-satusehat-mobile',
              ),
            ),
          ],
        ),
      ],
    );
  }

  static String _section(String heading) => switch (heading) {
    'Pangan' => 'Pangan',
    'Bahaya dan keamanan' => 'Yang perlu dihindari',
    'Penggunaan tradisional' ||
    'Sediaan obat tradisional' => 'Penggunaan tradisional',
    _ => 'Informasi lain',
  };
}

class _PartsTable extends StatelessWidget {
  const _PartsTable({required this.plant});

  final Plant plant;

  @override
  Widget build(BuildContext context) => SoftCard(
    padding: EdgeInsets.zero,
    child: Column(
      children: [
        _PartRow(
          icon: Icons.check_circle_outline,
          color: success,
          label: 'Bagian yang biasa dikonsumsi',
          value:
              plant.safeParts ??
              (plant.isPoisonous
                  ? 'Tidak ada, jangan dikonsumsi'
                  : 'Bukan tanaman pangan'),
        ),
        const Divider(height: 1, color: line),
        _PartRow(
          icon: Icons.block,
          color: danger,
          label: 'Bagian berbahaya',
          value: plant.dangerousParts ?? 'Tidak tercatat',
        ),
      ],
    ),
  );
}

class _PartRow extends StatelessWidget {
  const _PartRow({
    required this.icon,
    required this.color,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final Color color;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.all(14),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(color: inkMuted, fontSize: 11),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 14,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

class _Citation extends StatelessWidget {
  const _Citation({required this.title, required this.date, required this.url});

  final String title;
  final String date;
  final Uri url;

  @override
  Widget build(BuildContext context) => TextButton.icon(
    icon: const Icon(Icons.open_in_new, size: 15),
    label: Text('$title ($date)', textAlign: TextAlign.start),
    onPressed: () async {
      try {
        if (await launchUrl(url, mode: LaunchMode.externalApplication)) return;
      } catch (_) {
        // The source address remains available on devices without a browser.
      }
      if (!context.mounted) return;
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Tautan sumber'),
          content: SelectableText(url.toString()),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Tutup'),
            ),
          ],
        ),
      );
    },
  );
}
