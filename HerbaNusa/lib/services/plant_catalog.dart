import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/analysis.dart';
import '../models/model_card.dart';
import '../models/plant.dart';

/// Read-only data bundled with the app: knowledge base, labels, model results.
class PlantCatalog {
  PlantCatalog(this.plants, this.labels, this.modelCard);

  static Future<PlantCatalog> load() async {
    final labels = (jsonDecode(
      await rootBundle.loadString('assets/model/labels.json'),
    ) as List<dynamic>).cast<String>();
    final card = ModelCard.fromJson(
      jsonDecode(await rootBundle.loadString('assets/model/model_card.json'))
          as Map<String, dynamic>,
    );
    final kb = jsonDecode(
      await rootBundle.loadString('assets/data/knowledge_base.json'),
    ) as Map<String, dynamic>;
    final research = jsonDecode(
      await rootBundle.loadString('assets/data/research.json'),
    ) as Map<String, dynamic>;
    return PlantCatalog(
      {
        for (final name in labels)
          name: Plant.fromJson(
            name,
            kb[name] as Map<String, dynamic>,
            findings: [
              for (final row in research[name] as List<dynamic>)
                PlantFinding.fromJson(row as Map<String, dynamic>),
            ],
          ),
      },
      labels,
      card,
    );
  }

  /// Knowledge base entries keyed by model label.
  final Map<String, Plant> plants;

  /// Class names in model output order.
  final List<String> labels;
  final ModelCard modelCard;

  /// Candidates with toxicity >= 2 anywhere in the top 3: these trigger the
  /// red warning even when the model ranks a harmless species first.
  List<(PlantScore, Plant)> dangerousIn(Analysis analysis) => [
    for (final score in analysis.top)
      if (plants[score.name]!.isDangerous) (score, plants[score.name]!),
  ];

  /// All species sorted by local name.
  List<Plant> get sorted =>
      plants.values.toList()..sort((a, b) => a.local.compareTo(b.local));
}
