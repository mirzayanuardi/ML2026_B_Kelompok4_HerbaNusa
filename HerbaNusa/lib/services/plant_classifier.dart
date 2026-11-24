import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/analysis.dart';
import 'preprocess.dart';

/// Runs the MobileNetV2 species model in a background isolate, on the phone,
/// so no photo is uploaded and the app works offline.
class PlantClassifier {
  PlantClassifier._(this._model, this._runner, this._labels, this._threshold);

  final Interpreter _model;
  final IsolateInterpreter _runner;
  final List<String> _labels;
  final double _threshold;

  static Future<PlantClassifier> load({
    required List<String> labels,
    required double threshold,
  }) async {
    final model = await Interpreter.fromAsset('assets/model/model.tflite');
    final runner = await IsolateInterpreter.create(address: model.address);
    return PlantClassifier._(model, runner, labels, threshold);
  }

  /// Returns the top 3 species, or null when the file is not an image.
  Future<Analysis?> analyze(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final input = await compute(preprocessImage, bytes);
    if (input == null) return null;

    final output = Float32List(_labels.length);
    await _runner.run(input.buffer, output.buffer);
    return Analysis(top: topScores(output, _labels), threshold: _threshold);
  }

  void dispose() {
    _runner.close();
    _model.close();
  }
}

/// The [k] most probable labels, best first.
List<PlantScore> topScores(List<double> probs, List<String> labels, {int k = 3}) {
  final scores = [
    for (var i = 0; i < labels.length; i++) PlantScore(labels[i], probs[i]),
  ]..sort((a, b) => b.probability.compareTo(a.probability));
  return scores.take(k).toList();
}
