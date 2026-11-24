import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Side length of the square input expected by the TFLite model.
const int modelInputSize = 224;

/// Resolution of every training photo in the dataset.
const int datasetImageSize = 128;

/// Converts encoded image bytes into the model's input tensor.
///
/// Training (`01_train_cnn.py`) upscaled the 128 x 128 dataset photos to
/// 224 x 224 with bilinear resize, so a phone photo is first squashed (not
/// cropped) to 128 x 128 with an anti-aliasing filter and then upscaled the
/// same way. Values stay in 0..255: the model normalizes internally, and
/// dividing by 255 here would silently ruin accuracy. EXIF rotation is applied
/// first because phone cameras often store photos sideways. Returns null when
/// the bytes are not a readable image.
Float32List? preprocessImage(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return null;

  final upright = img.bakeOrientation(decoded);
  final rgb = upright
      .convert(format: img.Format.uint8, numChannels: 3)
      .getBytes(order: img.ChannelOrder.rgb);
  final small = resizeLikePil(
    rgb,
    upright.width,
    upright.height,
    datasetImageSize,
    datasetImageSize,
  );
  // When upscaling, PIL's triangle filter equals TF's half-pixel bilinear resize.
  final resized = resizeLikePil(
    small,
    datasetImageSize,
    datasetImageSize,
    modelInputSize,
    modelInputSize,
  );
  final input = Float32List(resized.length);
  for (var i = 0; i < resized.length; i++) {
    input[i] = resized[i].toDouble();
  }
  return input;
}

/// Sample positions and weights contributing to one output pixel.
class _Weights {
  const _Weights(this.start, this.values);

  final int start;
  final Float64List values;
}

/// Weights of PIL's triangle filter, which widens with the reduction factor.
/// A plain bilinear lookup would skip pixels and alias on large camera photos.
List<_Weights> _triangleWeights(int inSize, int outSize) {
  final scale = inSize / outSize;
  final filterScale = math.max(scale, 1.0);
  return [
    for (var out = 0; out < outSize; out++)
      () {
        final center = (out + 0.5) * scale;
        final start = math.max((center - filterScale + 0.5).toInt(), 0);
        final end = math.min((center + filterScale + 0.5).toInt(), inSize);
        final values = Float64List(end - start);
        var sum = 0.0;
        for (var x = start; x < end; x++) {
          final distance = ((x - center + 0.5) / filterScale).abs();
          values[x - start] = distance < 1 ? 1 - distance : 0;
          sum += values[x - start];
        }
        for (var i = 0; i < values.length; i++) {
          values[i] /= sum;
        }
        return _Weights(start, values);
      }(),
  ];
}

/// Resizes interleaved RGB the way PIL's `Image.resize(BILINEAR)` does, so the
/// app sees the same pixels the model saw during training.
Uint8List resizeLikePil(
  Uint8List rgb,
  int width,
  int height,
  int outWidth,
  int outHeight,
) {
  final horizontal = _triangleWeights(width, outWidth);
  final vertical = _triangleWeights(height, outHeight);

  // Horizontal pass first, rounding to 8 bits between passes as PIL does.
  final firstPass = Uint8List(outWidth * height * 3);
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < outWidth; x++) {
      final weights = horizontal[x];
      for (var channel = 0; channel < 3; channel++) {
        var value = 0.0;
        for (var i = 0; i < weights.values.length; i++) {
          value +=
              weights.values[i] *
              rgb[(y * width + weights.start + i) * 3 + channel];
        }
        firstPass[(y * outWidth + x) * 3 + channel] = (value + 0.5)
            .toInt()
            .clamp(0, 255);
      }
    }
  }

  final output = Uint8List(outWidth * outHeight * 3);
  for (var y = 0; y < outHeight; y++) {
    final weights = vertical[y];
    for (var x = 0; x < outWidth; x++) {
      for (var channel = 0; channel < 3; channel++) {
        var value = 0.0;
        for (var i = 0; i < weights.values.length; i++) {
          value +=
              weights.values[i] *
              firstPass[((weights.start + i) * outWidth + x) * 3 + channel];
        }
        output[(y * outWidth + x) * 3 + channel] = (value + 0.5).toInt().clamp(
          0,
          255,
        );
      }
    }
  }
  return output;
}
