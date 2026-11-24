import 'dart:io';

import 'package:flutter/material.dart';

import '../main.dart';
import '../models/analysis.dart';
import '../theme.dart';
import '../widgets/ui.dart';
import 'encyclopedia_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key, this.historyVersion = 0, this.onChanged});

  /// Bumped by the shell whenever another tab adds or removes an entry.
  final int historyVersion;
  final VoidCallback? onChanged;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  Future<List<HistoryEntry>>? _entries;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _entries ??= AppServices.of(context).history.load();
  }

  @override
  void didUpdateWidget(HistoryScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.historyVersion != widget.historyVersion) {
      _entries = AppServices.of(context).history.load();
    }
  }

  Future<void> _remove(HistoryEntry entry) async {
    final history = AppServices.of(context).history;
    await history.remove(entry);
    if (mounted) setState(() => _entries = history.load());
    widget.onChanged?.call();
  }

  String _date(DateTime value) {
    final local = value.toLocal();
    String two(int number) => number.toString().padLeft(2, '0');
    return '${two(local.day)}/${two(local.month)}/${local.year} ${two(local.hour)}:${two(local.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    final catalog = AppServices.of(context).catalog;
    final plants = catalog.plants;
    return Scaffold(
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const PageHeader(title: 'Riwayat Identifikasi'),
            Expanded(
              child: FutureBuilder<List<HistoryEntry>>(
                future: _entries,
                builder: (context, snapshot) {
                  if (!snapshot.hasData)
                    return const Center(child: CircularProgressIndicator());
                  final entries = snapshot.data!;
                  if (entries.isEmpty)
                    return const Center(
                      child: Text(
                        'Belum ada foto yang dianalisis',
                        style: TextStyle(color: inkMuted),
                      ),
                    );
                  return ListView.separated(
                    padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
                    itemCount: entries.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 8),
                    itemBuilder: (context, index) {
                      final entry = entries[index];
                      final plant = plants[entry.species];
                      return SoftCard(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: ListTile(
                          leading: ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.file(
                              File(entry.imagePath),
                              width: 52,
                              height: 52,
                              fit: BoxFit.cover,
                              cacheWidth: 156,
                              errorBuilder: (_, _, _) => const SizedBox(
                                width: 52,
                                height: 52,
                                child: Icon(Icons.image_not_supported_outlined),
                              ),
                            ),
                          ),
                          title: Text(
                            plant?.local ?? entry.species,
                            style: const TextStyle(fontWeight: FontWeight.w700),
                          ),
                          isThreeLine: true,
                          subtitle: Text(
                            '${plant?.scientific ?? ''}\n'
                            '${(entry.probability * 100).toStringAsFixed(1)}% yakin · '
                            '${_date(entry.createdAt)}',
                            style: const TextStyle(
                              color: inkMuted,
                              fontSize: 12,
                            ),
                          ),
                          trailing: IconButton(
                            tooltip: 'Hapus',
                            icon: const Icon(
                              Icons.delete_outline,
                              color: inkMuted,
                            ),
                            onPressed: () => _remove(entry),
                          ),
                          onTap: plant == null
                              ? null
                              : () => openPlant(context, plant),
                        ),
                      );
                    },
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
