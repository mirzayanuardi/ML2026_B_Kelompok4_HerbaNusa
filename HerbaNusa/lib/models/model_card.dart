/// Test results of the shipped model, written by tool/build_assets.py.
class ModelCard {
  const ModelCard({
    required this.threshold,
    required this.top1,
    required this.top3,
    required this.macroF1,
    required this.testImages,
    required this.trainImages,
    required this.coverage,
    required this.accuracyAnswered,
    required this.fatalRate,
    required this.fatalRateAnswered,
    required this.perClassRecall,
    required this.perClassSupport,
  });

  factory ModelCard.fromJson(Map<String, dynamic> json) {
    final perClass = (json['per_class'] as Map<String, dynamic>)
        .cast<String, Map<String, dynamic>>();
    double number(String key) => (json[key] as num).toDouble();
    return ModelCard(
      threshold: number('threshold'),
      top1: number('top1'),
      top3: number('top3'),
      macroF1: number('macro_f1'),
      testImages: json['test_images'] as int,
      trainImages: json['train_images'] as int,
      coverage: number('coverage'),
      accuracyAnswered: number('accuracy_answered'),
      fatalRate: number('fatal_rate'),
      fatalRateAnswered: number('fatal_rate_answered'),
      perClassRecall: {
        for (final MapEntry(:key, :value) in perClass.entries)
          key: (value['recall'] as num).toDouble(),
      },
      perClassSupport: {
        for (final MapEntry(:key, :value) in perClass.entries)
          key: value['support'] as int,
      },
    );
  }

  /// Answers below this top-1 probability are refused (safety_rules.json).
  final double threshold;
  final double top1;
  final double top3;
  final double macroF1;
  final int testImages;
  final int trainImages;

  /// Share of test photos answered at [threshold], and accuracy on those.
  final double coverage;
  final double accuracyAnswered;

  /// Poisonous photos predicted as food, without and with the threshold.
  final double fatalRate;
  final double fatalRateAnswered;
  final Map<String, double> perClassRecall;
  final Map<String, int> perClassSupport;
}
