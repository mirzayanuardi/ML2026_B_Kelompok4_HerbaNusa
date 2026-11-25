import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../theme.dart';
import '../widgets/ui.dart';
import 'result_screen.dart';

final _picker = ImagePicker();

/// Opens the camera or gallery; returns the photo path, or null if cancelled or refused.
Future<String?> pickPhoto(BuildContext context, ImageSource source) async {
  try {
    final photo = await _picker.pickImage(
      source: source,
      maxWidth: 1280,
      maxHeight: 1280,
      imageQuality: 95,
    );
    return photo?.path;
  } on PlatformException catch (error) {
    if (context.mounted) {
      final blocked = error.code.contains('access_denied');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            blocked
                ? 'Izin ditolak. Aktifkan izin kamera atau galeri di pengaturan perangkat.'
                : 'Gagal membuka ${source == ImageSource.camera ? 'kamera' : 'galeri'}.',
          ),
        ),
      );
    }
    return null;
  }
}

/// Identifikasi, step 1 of 2: choose a photo.
class IdentifyScreen extends StatefulWidget {
  const IdentifyScreen({super.key, required this.onHistorySaved});

  /// Called when an accepted result has been saved to history.
  final VoidCallback onHistorySaved;

  @override
  State<IdentifyScreen> createState() => _IdentifyScreenState();
}

class _IdentifyScreenState extends State<IdentifyScreen> {
  @override
  void initState() {
    super.initState();
    _recoverLostPhoto();
  }

  /// Android may kill the app while the camera is open; recover that photo.
  Future<void> _recoverLostPhoto() async {
    final lost = await _picker.retrieveLostData();
    if (!lost.isEmpty && lost.file != null && mounted) {
      _preview(lost.file!.path);
    }
  }

  Future<void> _pick(ImageSource source) async {
    final path = await pickPhoto(context, source);
    if (path != null && mounted) await _preview(path);
  }

  /// Copies the bundled sample photo to a file, since results work on file paths.
  Future<void> _sample() async {
    final bytes = await rootBundle.load('assets/images/sample_plant.jpg');
    final file = File(
      p.join((await getTemporaryDirectory()).path, 'contoh_tanaman.jpg'),
    );
    await file.writeAsBytes(bytes.buffer.asUint8List());
    if (mounted) await _preview(file.path);
  }

  Future<void> _preview(String path) async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => PreviewScreen(
          imagePath: path,
          onHistorySaved: widget.onHistorySaved,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        children: [
          const PageHeader(title: 'Identifikasi Tanaman'),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
              children: [
                const Center(child: Pill('Langkah 1 dari 2')),
                const SizedBox(height: 14),
                Text(
                  'Unggah foto tanaman',
                  textAlign: TextAlign.center,
                  style: heading(24),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Foto daun atau bagian tanaman dari dekat agar bentuk dan teksturnya terlihat jelas.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: inkMuted, fontSize: 13, height: 1.5),
                ),
                const SizedBox(height: 20),
                _DropZone(onTap: () => _pick(ImageSource.gallery)),
                const SizedBox(height: 18),
                const Row(
                  children: [
                    Expanded(child: Divider(color: line)),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 12),
                      child: Text(
                        'ATAU',
                        style: TextStyle(
                          color: inkFaint,
                          fontSize: 10,
                          letterSpacing: 1,
                        ),
                      ),
                    ),
                    Expanded(child: Divider(color: line)),
                  ],
                ),
                const SizedBox(height: 18),
                OutlinedButton.icon(
                  key: const Key('camera_button'),
                  onPressed: () => _pick(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Ambil Foto'),
                ),
                const SizedBox(height: 6),
                Center(
                  child: TextButton(
                    key: const Key('sample_button'),
                    onPressed: _sample,
                    child: const Text(
                      'Coba dengan foto contoh',
                      style: TextStyle(
                        decoration: TextDecoration.underline,
                        decorationColor: accentDeep,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                const Divider(color: line),
                const SizedBox(height: 14),
                Text('Tips foto yang baik', style: heading(15)),
                const SizedBox(height: 12),
                const Row(
                  children: [
                    Expanded(
                      child: _Tip(
                        icon: Icons.wb_sunny_outlined,
                        colors: [Color(0xFFF3E7B8), Color(0xFFE6D48A)],
                        label: 'Pencahayaan\nterang',
                      ),
                    ),
                    SizedBox(width: 10),
                    Expanded(
                      child: _Tip(
                        icon: Icons.eco_outlined,
                        colors: [Color(0xFFD5EAD8), Color(0xFFA9D3B2)],
                        label: 'Daun terlihat\njelas',
                      ),
                    ),
                    SizedBox(width: 10),
                    Expanded(
                      child: _Tip(
                        icon: Icons.center_focus_strong_outlined,
                        colors: [Color(0xFFDDEEDC), Color(0xFFBBDABE)],
                        label: 'Satu tanaman\nsaja',
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DropZone extends StatelessWidget {
  const _DropZone({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DashedBorder(),
      child: Material(
        color: const Color(0xFFEFF7EC),
        borderRadius: BorderRadius.circular(22),
        child: InkWell(
          key: const Key('gallery_button'),
          borderRadius: BorderRadius.circular(22),
          onTap: onTap,
          child: const Padding(
            padding: EdgeInsets.symmetric(vertical: 30, horizontal: 16),
            child: Column(
              children: [
                IconBadge(icon: Icons.image_outlined, size: 56),
                SizedBox(height: 14),
                Text(
                  'Pilih foto dari galeri',
                  style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                ),
                SizedBox(height: 4),
                Text(
                  'JPG atau PNG',
                  style: TextStyle(color: inkMuted, fontSize: 11),
                ),
                SizedBox(height: 12),
                Text(
                  'Pilih Foto',
                  style: TextStyle(
                    color: accentDeep,
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _DashedBorder extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = accent.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    final path = Path()
      ..addRRect(
        RRect.fromRectAndRadius(Offset.zero & size, const Radius.circular(22)),
      );
    for (final metric in path.computeMetrics()) {
      for (double d = 0; d < metric.length; d += 10) {
        canvas.drawPath(metric.extractPath(d, d + 6), paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _Tip extends StatelessWidget {
  const _Tip({required this.icon, required this.colors, required this.label});

  final IconData icon;
  final List<Color> colors;
  final String label;

  @override
  Widget build(BuildContext context) => Column(
    children: [
      Container(
        height: 60,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          gradient: LinearGradient(
            colors: colors,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(child: Icon(icon, color: Colors.white, size: 26)),
      ),
      const SizedBox(height: 8),
      Text(
        label,
        textAlign: TextAlign.center,
        style: const TextStyle(color: inkMuted, fontSize: 11, height: 1.5),
      ),
    ],
  );
}

/// Identifikasi, step 2 of 2: confirm the photo before running the model.
class PreviewScreen extends StatefulWidget {
  const PreviewScreen({
    super.key,
    required this.imagePath,
    required this.onHistorySaved,
  });

  final VoidCallback onHistorySaved;

  final String imagePath;

  @override
  State<PreviewScreen> createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  late String _path = widget.imagePath;

  Future<void> _replace() async {
    final path = await pickPhoto(context, ImageSource.gallery);
    if (path != null && mounted) setState(() => _path = path);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const PageHeader(title: 'Pratinjau Foto'),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                children: [
                  const Center(child: Pill('Langkah 2 dari 2')),
                  const SizedBox(height: 14),
                  Text(
                    'Foto sudah siap?',
                    textAlign: TextAlign.center,
                    style: heading(24),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Periksa fotonya, lalu mulai identifikasi.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: inkMuted, fontSize: 13),
                  ),
                  const SizedBox(height: 18),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(24),
                    child: AspectRatio(
                      aspectRatio: 1,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          const ColoredBox(color: Color(0xFF16241B)),
                          Image.file(
                            File(_path),
                            key: ValueKey(_path),
                            fit: BoxFit.contain,
                          ),
                          const CustomPaint(painter: _CornerBrackets()),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  SoftCard(
                    color: successSoft,
                    padding: const EdgeInsets.all(12),
                    child: const Row(
                      children: [
                        CircleAvatar(
                          radius: 14,
                          backgroundColor: success,
                          child: Icon(Icons.eco, size: 14, color: Colors.white),
                        ),
                        SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'Jika model ragu (keyakinan di bawah 27%), aplikasi akan menolak menjawab.',
                            style: TextStyle(
                              color: Color(0xFF3E6B44),
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              child: Row(
                children: [
                  Expanded(
                    flex: 2,
                    child: OutlinedButton.icon(
                      onPressed: _replace,
                      style: OutlinedButton.styleFrom(
                        foregroundColor: ink,
                        side: const BorderSide(color: line),
                      ),
                      icon: const Icon(Icons.image_outlined, size: 20),
                      label: const Text('Ganti Foto'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    flex: 3,
                    child: FilledButton.icon(
                      key: const Key('identify_button'),
                      onPressed: () => Navigator.of(context).pushReplacement(
                        MaterialPageRoute<void>(
                          builder: (_) => ResultScreen(
                            imagePath: _path,
                            onHistorySaved: widget.onHistorySaved,
                          ),
                        ),
                      ),
                      icon: const Icon(Icons.auto_awesome, size: 20),
                      label: const Text('Identifikasi'),
                    ),
                  ),
                ],
              ),
            ),
            const Padding(
              padding: EdgeInsets.only(bottom: 12),
              child: Text(
                'Foto diproses di perangkat ini dan tidak dikirim ke server.',
                style: TextStyle(color: inkMuted, fontSize: 11),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CornerBrackets extends CustomPainter {
  const _CornerBrackets();

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    const inset = 14.0, arm = 22.0;
    for (final (x, y, dx, dy) in [
      (inset, inset, 1.0, 1.0),
      (size.width - inset, inset, -1.0, 1.0),
      (inset, size.height - inset, 1.0, -1.0),
      (size.width - inset, size.height - inset, -1.0, -1.0),
    ]) {
      canvas.drawLine(Offset(x, y), Offset(x + arm * dx, y), paint);
      canvas.drawLine(Offset(x, y), Offset(x, y + arm * dy), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
