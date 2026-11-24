import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/analysis.dart';

/// Keeps past analyses on the device. Photos are copied into app storage
/// because picker results live in a cache the system may clear.
class HistoryStore {
  static const _key = 'history';
  static const _maxEntries = 50;

  Future<List<HistoryEntry>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final entries = prefs.getStringList(_key) ?? const [];
    return [
      for (final entry in entries)
        HistoryEntry.fromJson(jsonDecode(entry) as Map<String, dynamic>),
    ];
  }

  Future<void> add({
    required String sourcePath,
    required String species,
    required double probability,
  }) async {
    final directory = await getApplicationDocumentsDirectory();
    final createdAt = DateTime.now();
    final target = p.join(
      directory.path,
      'scan_${createdAt.millisecondsSinceEpoch}${p.extension(sourcePath)}',
    );
    await File(sourcePath).copy(target);

    final entries = await load();
    entries.insert(
      0,
      HistoryEntry(
        imagePath: target,
        species: species,
        probability: probability,
        createdAt: createdAt,
      ),
    );
    for (final stale in entries.skip(_maxEntries)) {
      await _deleteImage(stale);
    }
    await _save(entries.take(_maxEntries).toList());
  }

  Future<void> remove(HistoryEntry entry) async {
    final entries = await load();
    entries.removeWhere((item) => item.imagePath == entry.imagePath);
    await _deleteImage(entry);
    await _save(entries);
  }

  Future<void> _deleteImage(HistoryEntry entry) async {
    final file = File(entry.imagePath);
    if (await file.exists()) await file.delete();
  }

  Future<void> _save(List<HistoryEntry> entries) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_key, [
      for (final entry in entries) jsonEncode(entry.toJson()),
    ]);
  }
}
