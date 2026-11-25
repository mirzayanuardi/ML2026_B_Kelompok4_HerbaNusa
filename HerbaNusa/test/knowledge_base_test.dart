import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:herbanusa/main.dart';
import 'package:herbanusa/screens/history_screen.dart';
import 'package:herbanusa/screens/result_screen.dart';
import 'package:herbanusa/services/history_store.dart';
import 'package:herbanusa/widgets/plant_sections.dart';
import 'package:herbanusa/models/analysis.dart';
import 'package:herbanusa/models/model_card.dart';
import 'package:herbanusa/models/plant.dart';
import 'package:herbanusa/services/plant_catalog.dart';
import 'package:herbanusa/services/plant_classifier.dart';
import 'package:herbanusa/widgets/ui.dart';

dynamic _json(String path) => jsonDecode(File(path).readAsStringSync());

void main() {
  final labels = (_json('assets/model/labels.json') as List).cast<String>();
  final kb = _json('assets/data/knowledge_base.json') as Map<String, dynamic>;
  final card = ModelCard.fromJson(
    _json('assets/model/model_card.json') as Map<String, dynamic>,
  );
  final catalog = PlantCatalog(
    {
      for (final l in labels)
        l: Plant.fromJson(l, kb[l] as Map<String, dynamic>),
    },
    labels,
    card,
  );
  Plant plant(String label) => catalog.plants[label]!;

  test('every label has an entry, a photo and test results', () {
    expect(labels.length, 100);
    for (final name in labels) {
      expect(kb.containsKey(name), isTrue, reason: name);
      expect(File(plantPhotoAsset(name)).existsSync(), isTrue, reason: name);
      expect(card.perClassSupport[name], greaterThan(0), reason: name);
      expect(plant(name).firstAid, isNotEmpty, reason: name);
    }
    expect(card.threshold, 0.27);
  });

  test(
    'each model label has offline findings with public source addresses',
    () {
      final evidence =
          _json('assets/data/research.json') as Map<String, dynamic>;
      expect(evidence.keys.toSet(), labels.toSet());
      for (final label in labels) {
        final rows = evidence[label] as List<dynamic>;
        expect(rows, isNotEmpty, reason: label);
        for (final row in rows) {
          final finding = PlantFinding.fromJson(row as Map<String, dynamic>);
          expect(finding.text.trim(), isNotEmpty, reason: label);
          expect(finding.sources, isNotEmpty, reason: label);
          for (final source in finding.sources) {
            expect(source.url.scheme, 'https');
            expect(source.url.host, isNotEmpty);
            expect(source.title.trim(), isNotEmpty);
          }
        }
      }
    },
  );
  testWidgets('catnip detail presents culinary, traditional and risk sources', (
    tester,
  ) async {
    final loaded = (await tester.runAsync(PlantCatalog.load))!;
    final catnip = loaded.plants['Nepeta Cataria (Catnip)']!;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: PlantSafetySections(plant: catnip),
          ),
        ),
      ),
    );
    for (final text in [
      'Bagian aman & berbahaya',
      'Bagian yang biasa dikonsumsi',
      'Yang perlu dihindari',
      'Daun dan batang untuk teh',
      'Penjelasan',
      'Tidak tercatat',
      'Penggunaan tradisional',
      'Bagian berbahaya',
    ]) {
      expect(find.text(text), findsOneWidget);
    }
    expect(find.textContaining('kram perut dan kembung'), findsOneWidget);
    expect(find.text('Catatan'), findsNothing);
    expect(find.byType(BottomSheet), findsNothing);
    final sources = find.text('Sumber informasi');
    await tester.ensureVisible(sources);
    await tester.tap(sources);
    await tester.pumpAndSettle();
    expect(find.byIcon(Icons.open_in_new), findsWidgets);
  });
  testWidgets(
    'history refreshes after a save completes even if result was closed',
    (tester) async {
      const photo = 'test/fixtures/sample_0.jpg';
      final label = labels.first;
      final history = _DelayedHistory();
      final navigator = GlobalKey<NavigatorState>();
      var historyVersion = 0;

      await tester.pumpWidget(
        AppServices(
          catalog: catalog,
          classifier: _TestClassifier(label),
          history: history,
          child: MaterialApp(
            navigatorKey: navigator,
            home: StatefulBuilder(
              builder: (context, setState) => Column(
                children: [
                  TextButton(
                    onPressed: () => navigator.currentState!.push(
                      MaterialPageRoute<void>(
                        builder: (_) => ResultScreen(
                          imagePath: photo,
                          onHistorySaved: () =>
                              setState(() => historyVersion++),
                        ),
                      ),
                    ),
                    child: const Text('Buka hasil'),
                  ),
                  Expanded(
                    child: HistoryScreen(historyVersion: historyVersion),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.text('Buka hasil'));
      await tester.pumpAndSettle();
      expect(find.text('Hasil Identifikasi'), findsOneWidget);
      expect(history.addCalls, 1);
      expect(historyVersion, 0);

      navigator.currentState!.pop();
      await tester.pumpAndSettle();
      expect(find.text('Belum ada foto yang dianalisis'), findsOneWidget);

      history.finishSave.complete();
      await tester.pumpAndSettle();
      expect(find.text(plant(label).local), findsOneWidget);
      expect(historyVersion, 1);
      expect(history.addCalls, 1);
    },
  );

  test('every look-alike names a known poisonous species', () {
    for (final p in catalog.plants.values) {
      for (final alike in p.lookAlikes) {
        expect(plant(alike.name).isPoisonous, isTrue, reason: alike.name);
      }
    }
  });

  // Release checklist from the ML handoff (README section 5).
  group('badge matrix', () {
    test('sirih is medicinal only', () {
      final p = plant('Piper Betle (Sirih)');
      expect(p.status, SafetyStatus.obat);
      expect(p.categories, {PlantCategory.obat});
    });

    test('jarak is forbidden', () {
      expect(plant('Ricinus Communis (Jarak)').status, SafetyStatus.dilarang);
    });

    test('pir is food with a note on its seeds', () {
      final p = plant('Pyrus Communis (Pir)');
      expect(p.status, SafetyStatus.pangan);
      expect(p.dangerousParts, 'biji');
    });

    test('gadung must be processed', () {
      expect(
        plant('Dioscorea Hispida (Gadung)').status,
        SafetyStatus.harusDiolah,
      );
    });

    test('kayu putih is a caution plus medicine', () {
      final p = plant('Melaleuca Leucadendra (Kayu Putih)');
      expect(p.status, SafetyStatus.hatiHati);
      expect(
        p.categories,
        containsAll([PlantCategory.obat, PlantCategory.beracun]),
      );
    });

    test('no species falls through to unknown', () {
      for (final p in catalog.plants.values) {
        expect(p.status, isNot(SafetyStatus.unknown), reason: p.name);
      }
    });
  });

  group('analysis rules', () {
    test('refuses below the threshold only', () {
      Analysis at(double p) => Analysis(
        top: [PlantScore(labels.first, p)],
        threshold: card.threshold,
      );
      expect(at(0.27).rejected, isFalse);
      expect(at(0.2699).rejected, isTrue);
    });

    test('top scores are the three largest, best first', () {
      final probs = List<double>.filled(labels.length, 0.001);
      probs[5] = 0.5;
      probs[41] = 0.3;
      probs[2] = 0.1;
      final top = topScores(probs, labels);
      expect(top.map((s) => s.name), [labels[5], labels[41], labels[2]]);
    });

    test('warns when a poisonous species is anywhere in the top 3', () {
      final analysis = Analysis(
        top: const [
          PlantScore('Piper Sarmentosum (Cabean)', 0.65),
          PlantScore('Pluchea Indica (Beluntas)', 0.2),
          PlantScore('Gynura Segetum (Daun Dewa)', 0.1),
        ],
        threshold: card.threshold,
      );
      final dangerous = catalog.dangerousIn(analysis);
      expect(dangerous.map((d) => d.$1.name), ['Gynura Segetum (Daun Dewa)']);
      expect(
        plant('Piper Sarmentosum (Cabean)').lookAlikes.single.name,
        'Gynura Segetum (Daun Dewa)',
      );
    });

    test('no warning when the top 3 are all harmless', () {
      final analysis = Analysis(
        top: const [
          PlantScore('Curcuma Domestica (Kunyit)', 0.8),
          PlantScore('Kaempferia Galanga (Kencur)', 0.1),
          PlantScore('Zingiber Officinale (Jahe)', 0.05),
        ],
        threshold: card.threshold,
      );
      expect(catalog.dangerousIn(analysis), isEmpty);
    });
  });
}

class _TestClassifier implements PlantClassifier {
  _TestClassifier(this.label);

  final String label;

  @override
  Future<Analysis?> analyze(String imagePath) async =>
      Analysis(top: [PlantScore(label, .9)], threshold: .27);

  @override
  void dispose() {}
}

class _DelayedHistory extends HistoryStore {
  final finishSave = Completer<void>();
  final _entries = <HistoryEntry>[];
  int addCalls = 0;

  @override
  Future<List<HistoryEntry>> load() async => List.of(_entries);

  @override
  Future<void> add({
    required String sourcePath,
    required String species,
    required double probability,
  }) async {
    addCalls++;
    await finishSave.future;
    _entries.add(
      HistoryEntry(
        imagePath: sourcePath,
        species: species,
        probability: probability,
        createdAt: DateTime(2026),
      ),
    );
  }
}
