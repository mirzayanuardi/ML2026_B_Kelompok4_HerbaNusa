/// One candidate species and its model probability.
class PlantScore {
  const PlantScore(this.name, this.probability);

  final String name;
  final double probability;
}

/// Outcome of running the model on a single photo.
class Analysis {
  const Analysis({required this.top, required this.threshold});

  /// The three highest-probability species, best first.
  final List<PlantScore> top;
  final double threshold;

  PlantScore get best => top.first;

  /// The model is unsure, so the app refuses to name the plant.
  bool get rejected => best.probability < threshold;
}

/// A saved analysis shown on the history tab.
class HistoryEntry {
  const HistoryEntry({
    required this.imagePath,
    required this.species,
    required this.probability,
    required this.createdAt,
  });

  factory HistoryEntry.fromJson(Map<String, dynamic> json) => HistoryEntry(
    imagePath: json['imagePath'] as String,
    species: json['species'] as String,
    probability: (json['probability'] as num).toDouble(),
    createdAt: DateTime.parse(json['createdAt'] as String),
  );

  final String imagePath;
  final String species;
  final double probability;
  final DateTime createdAt;

  Map<String, dynamic> toJson() => {
    'imagePath': imagePath,
    'species': species,
    'probability': probability,
    'createdAt': createdAt.toIso8601String(),
  };
}
