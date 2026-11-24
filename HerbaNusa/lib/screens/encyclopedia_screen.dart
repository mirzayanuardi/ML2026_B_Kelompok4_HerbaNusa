import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../main.dart';
import '../models/plant.dart';
import '../theme.dart';
import '../widgets/plant_sections.dart';
import '../widgets/ui.dart';

/// Katalog: offline search and category filter over the 100 species.
class EncyclopediaScreen extends StatefulWidget {
  const EncyclopediaScreen({super.key});

  @override
  State<EncyclopediaScreen> createState() => _EncyclopediaScreenState();
}

class _EncyclopediaScreenState extends State<EncyclopediaScreen> {
  String _query = '';
  PlantCategory? _category;

  @override
  Widget build(BuildContext context) {
    final catalog = AppServices.of(context).catalog;
    final query = _query.toLowerCase();
    final plants = catalog.sorted.where((plant) {
      final matchesCategory =
          _category == null || plant.categories.contains(_category);
      return matchesCategory &&
          (plant.local.toLowerCase().contains(query) ||
              plant.scientific.toLowerCase().contains(query));
    }).toList();

    return SafeArea(
      child: CustomScrollView(
        slivers: [
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 0),
            sliver: SliverList.list(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Overline('Flora Nusantara'),
                          const SizedBox(height: 6),
                          Text('Katalog Tanaman', style: heading(24)),
                          const SizedBox(height: 4),
                          Text(
                            '${catalog.plants.length} spesies tanaman obat Indonesia yang dikenali model.',
                            style: const TextStyle(
                              color: inkMuted,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Icon(Icons.local_florist, color: accent, size: 30),
                  ],
                ),
                const SizedBox(height: 16),
                TextField(
                  onChanged: (value) => setState(() => _query = value),
                  decoration: const InputDecoration(
                    hintText: 'Cari nama lokal atau ilmiah...',
                    prefixIcon: Icon(Icons.search),
                  ),
                ),
                const SizedBox(height: 12),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      _filter('Semua', null),
                      for (final category in PlantCategory.values)
                        _filter(category.label, category),
                    ],
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  '${plants.length} spesies ditemukan',
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 10),
              ],
            ),
          ),
          if (plants.isEmpty)
            const SliverFillRemaining(
              hasScrollBody: false,
              child: Center(
                child: Text(
                  'Tidak ditemukan',
                  style: TextStyle(color: inkMuted),
                ),
              ),
            )
          else
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
              sliver: SliverGrid.builder(
                itemCount: plants.length,
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  childAspectRatio: 0.78,
                ),
                itemBuilder: (context, index) =>
                    PlantCard(plant: plants[index]),
              ),
            ),
        ],
      ),
    );
  }

  Widget _filter(String label, PlantCategory? category) {
    final selected = _category == category;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: Material(
        color: selected ? ink : Colors.white,
        shape: StadiumBorder(side: BorderSide(color: selected ? ink : line)),
        child: InkWell(
          customBorder: const StadiumBorder(),
          onTap: () => setState(() => _category = category),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
            child: Text(
              label,
              style: TextStyle(
                color: selected ? Colors.white : inkMuted,
                fontSize: 12,
                fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

void openPlant(BuildContext context, Plant plant) => Navigator.of(context).push(
  MaterialPageRoute<void>(builder: (_) => PlantDetailScreen(plant: plant)),
);

/// Photo card for a species, opening its detail page.
class PlantCard extends StatelessWidget {
  const PlantCard({super.key, required this.plant});

  final Plant plant;

  @override
  Widget build(BuildContext context) => SoftCard(
    padding: EdgeInsets.zero,
    onTap: () => openPlant(context, plant),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              PlantPhoto(plant: plant),
              Positioned(
                left: 8,
                top: 8,
                child: StatusBadge(status: plant.status, compact: true),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                plant.local,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                plant.scientific,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  color: inkMuted,
                  fontSize: 10.5,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

/// Kartu spesies: scientific name first, then status, parts, first aid.
class PlantDetailScreen extends StatelessWidget {
  const PlantDetailScreen({super.key, required this.plant});

  final Plant plant;

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.paddingOf(context).top;
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        body: ListView(
          padding: EdgeInsets.zero,
          children: [
            Stack(
              clipBehavior: Clip.none,
              children: [
                SizedBox(
                  height: 300 + top,
                  width: double.infinity,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      PlantPhoto(plant: plant),
                      const DecoratedBox(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Color(0x33000000),
                              Color(0x00000000),
                              Color(0xDD10271A),
                            ],
                            stops: [0, 0.4, 1],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Positioned(
                  top: top + 12,
                  left: 16,
                  child: SquareIconButton(
                    icon: Icons.chevron_left,
                    tooltip: 'Kembali',
                    dark: true,
                    onPressed: () => Navigator.of(context).maybePop(),
                  ),
                ),
                Positioned(
                  left: 20,
                  right: 20,
                  bottom: 22,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      StatusBadge(status: plant.status),
                      const SizedBox(height: 10),
                      Text(
                        plant.scientific,
                        style: heading(
                          28,
                          weight: FontWeight.w800,
                          color: Colors.white,
                        ).copyWith(fontStyle: FontStyle.italic),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        plant.local,
                        style: const TextStyle(
                          color: Color(0xE6FFFFFF),
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    plant.status.description,
                    style: const TextStyle(color: inkMuted, fontSize: 12.5),
                  ),
                  const SizedBox(height: 10),
                  CategoryChips(plant: plant),
                  const SizedBox(height: 22),
                  PlantSafetySections(plant: plant),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
