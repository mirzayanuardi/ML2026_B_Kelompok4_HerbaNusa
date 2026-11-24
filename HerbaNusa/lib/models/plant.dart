/// Plant categories from the knowledge base. A species can have several.
enum PlantCategory {
  pangan('Pangan'),
  obat('Obat'),
  beracun('Beracun');

  const PlantCategory(this.label);
  final String label;
}

/// The single status shown most prominently on a species, from the badge
/// matrix in the ML handoff (05_bahan_aplikasi/README.md section 3.3).
enum SafetyStatus {
  dilarang('DILARANG', 'Beracun, jangan dikonsumsi'),
  harusDiolah('HARUS DIOLAH', 'Hanya aman setelah diolah dengan benar'),
  pangan('PANGAN', 'Bagian tertentu biasa dikonsumsi'),
  hatiHati('HATI-HATI', 'Ada bagian atau dosis yang berbahaya'),
  obat('OBAT', 'Digunakan secara tradisional'),
  unknown('TIDAK DIKENALI', 'Belum ada data keamanan');

  const SafetyStatus(this.label, this.description);
  final String label;
  final String description;
}

/// A poisonous species the model has been measured to mistake for this one.
class LookAlike {
  const LookAlike(this.name, this.confidence);

  final String name;

  /// Model confidence of the wrong answer on the test photo.
  final double confidence;
}

/// One entry of assets/data/knowledge_base.json, keyed by the model label.
class Plant {
  const Plant({
    this.findings = const [],
    required this.name,
    required this.scientific,
    required this.local,
    required this.categories,
    required this.toxicity,
    required this.safeParts,
    required this.dangerousParts,
    required this.note,
    required this.firstAid,
    required this.lookAlikes,
  });

  factory Plant.fromJson(
    String name,
    Map<String, dynamic> json, {
    List<PlantFinding> findings = const [],
  }) => Plant(
    findings: findings,
    name: name,
    scientific: json['ilmiah'] as String,
    local: json['lokal'] as String,
    categories: {
      if (json['pangan'] == 1) PlantCategory.pangan,
      if (json['obat'] == 1) PlantCategory.obat,
      if (json['beracun'] == 1) PlantCategory.beracun,
    },
    toxicity: json['level_racun'] as int,
    safeParts: _part(json['bagian_aman'] as String),
    dangerousParts: _part(json['bagian_bahaya'] as String),
    note: json['catatan'] as String,
    firstAid: (json['p3k'] as List<dynamic>).cast<String>(),
    lookAlikes: [
      for (final item in json['tertukar_dengan'] as List<dynamic>)
        LookAlike(
          (item as Map<String, dynamic>)['nama'] as String,
          (item['keyakinan'] as num).toDouble(),
        ),
    ],
  );

  static String? _part(String value) => value == '-' ? null : value;

  /// Model label, e.g. "Zingiber Officinale (Jahe)".
  final String name;
  final List<PlantFinding> findings;
  final String scientific;
  final String local;
  final Set<PlantCategory> categories;

  /// 0 = not poisonous, 1 = some part or dose, 2 = needs processing or
  /// supervision, 3 = dangerous.
  final int toxicity;

  /// Parts usually consumed or used; null when none.
  final String? safeParts;
  final String? dangerousParts;
  final String note;
  final List<String> firstAid;
  final List<LookAlike> lookAlikes;

  bool get isFood => categories.contains(PlantCategory.pangan);
  bool get isMedicinal => categories.contains(PlantCategory.obat);
  bool get isPoisonous => categories.contains(PlantCategory.beracun);

  /// Triggers the red warning when this species appears anywhere in the top 3.
  bool get isDangerous => toxicity >= 2;

  SafetyStatus get status {
    if (toxicity >= 3) return SafetyStatus.dilarang;
    if (isFood && toxicity == 2) return SafetyStatus.harusDiolah;
    if (isFood) return SafetyStatus.pangan;
    if (isPoisonous) return SafetyStatus.hatiHati;
    if (isMedicinal) return SafetyStatus.obat;
    return SafetyStatus.unknown;
  }
}

class PlantFinding {
  const PlantFinding(this.heading, this.text, this.sources);

  factory PlantFinding.fromJson(Map<String, dynamic> json) =>
      PlantFinding(json['heading'] as String, json['text'] as String, [
        for (final source in json['sources'] as List<dynamic>)
          (
            title: source['title'] as String,
            date: source['date'] as String,
            url: Uri.parse(source['url'] as String),
          ),
      ]);

  final String heading;
  final String text;
  final List<({String title, String date, Uri url})> sources;
}
