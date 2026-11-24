import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:herbanusa/services/preprocess.dart';
import 'package:image/image.dart' as img;

void main() {
  const tensorLength = modelInputSize * modelInputSize * 3;

  // Fixtures come from tool/build_assets.py: dataset photos and the tensor that
  // the training code (tf.io.decode_image + tf.image.resize) produced for them.
  test('matches the TensorFlow preprocessing for dataset photos', () {
    for (var index = 0; index < 4; index++) {
      final bytes = File('test/fixtures/sample_$index.jpg').readAsBytesSync();
      final expected = File('test/fixtures/sample_$index.bin')
          .readAsBytesSync()
          .buffer
          .asFloat32List();

      final actual = preprocessImage(bytes)!;
      expect(actual.length, expected.length);
      var total = 0.0;
      for (var i = 0; i < actual.length; i++) {
        total += (actual[i] - expected[i]).abs();
      }
      // Values are 0..255. What remains is JPEG decoder noise (measured 1.4
      // gray levels). A PIL-decoded pipeline with the same noise scored 89.3%
      // vs 89.2% for TF on the full test set, so this does not hurt accuracy.
      expect(total / actual.length, lessThan(2.5), reason: 'sample_$index');
    }
  });

  test('keeps pixels in 0..255 instead of normalizing', () {
    final white = img.Image(width: 900, height: 600)
      ..clear(img.ColorRgb8(255, 255, 255));
    final input = preprocessImage(img.encodeJpg(white))!;
    expect(input.length, tensorLength);
    expect(input.reduce((a, b) => a < b ? a : b), greaterThan(250));
  });

  test('returns null for bytes that are not an image', () {
    expect(preprocessImage(Uint8List.fromList([1, 2, 3])), isNull);
  });
}
