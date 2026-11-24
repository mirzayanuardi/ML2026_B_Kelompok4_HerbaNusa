import 'package:flutter/material.dart';

import '../theme.dart';
import '../widgets/ui.dart';

/// First-launch welcome page, also reachable from Profil > Lihat Onboarding.
class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key, required this.onStart});

  final VoidCallback onStart;

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.sizeOf(context);
    return Scaffold(
      body: Column(
        children: [
          Container(
            height: size.height * 0.46,
            width: double.infinity,
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFFD9EEDC), cream],
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Stack(
                alignment: Alignment.bottomCenter,
                children: [
                  const Positioned(
                    top: 28,
                    left: 32,
                    child: _Sparkle(size: 14),
                  ),
                  const Positioned(
                    top: 90,
                    right: 36,
                    child: _Sparkle(size: 10),
                  ),
                  const Positioned(top: 52, right: 70, child: _Dot()),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(56, 24, 56, 0),
                    child: ClipRRect(
                      borderRadius: const BorderRadius.vertical(
                        top: Radius.circular(400),
                        bottom: Radius.circular(28),
                      ),
                      child: AspectRatio(
                        aspectRatio: 0.82,
                        child: Image.asset(
                          plantPhotoAsset('Centella Asiatica (Pegagan)'),
                          fit: BoxFit.cover,
                          filterQuality: FilterQuality.medium,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          Expanded(
            child: SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(28, 20, 28, 20),
                child: Column(
                  children: [
                    const _Logo(),
                    const SizedBox(height: 16),
                    Text(
                      'Kenali Tanaman\ndi Sekitarmu',
                      textAlign: TextAlign.center,
                      style: heading(28, weight: FontWeight.w700),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Ambil atau pilih foto tanaman. Lihat hasil identifikasi, bagian yang biasa dimakan atau perlu dihindari, dan penggunaan tradisionalnya.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: inkMuted,
                        fontSize: 12,
                        height: 1.5,
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Spacer(),
                    const _Disclaimer(),
                    const SizedBox(height: 12),
                    FilledButton(
                      key: const Key('start_button'),
                      onPressed: onStart,
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text('Saya Mengerti, Mulai'),
                          SizedBox(width: 8),
                          Icon(Icons.chevron_right, size: 20),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.shield_outlined, size: 14, color: inkMuted),
                        SizedBox(width: 6),
                        Flexible(
                          child: Text(
                            'Foto diproses di perangkat ini dan tidak dikirim ke server',
                            style: TextStyle(color: inkMuted, fontSize: 11),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Logo extends StatelessWidget {
  const _Logo();

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      const Icon(Icons.eco, color: accent, size: 22),
      const SizedBox(width: 6),
      Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: 'Herba',
              style: heading(16, weight: FontWeight.w800),
            ),
            TextSpan(
              text: 'Nusa',
              style: heading(16, weight: FontWeight.w800, color: accent),
            ),
          ],
        ),
      ),
    ],
  );
}

class _Sparkle extends StatelessWidget {
  const _Sparkle({required this.size});

  final double size;

  @override
  Widget build(BuildContext context) => Icon(
    Icons.auto_awesome,
    size: size,
    color: accent.withValues(alpha: 0.6),
  );
}

class _Dot extends StatelessWidget {
  const _Dot();

  @override
  Widget build(BuildContext context) => Container(
    width: 8,
    height: 8,
    decoration: BoxDecoration(
      color: accent.withValues(alpha: 0.25),
      shape: BoxShape.circle,
    ),
  );
}

/// Shown once before first use (ML handoff: Disclaimer screen).
class _Disclaimer extends StatelessWidget {
  const _Disclaimer();

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: warningSoft,
      borderRadius: BorderRadius.circular(14),
    ),
    child: const Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.info_outline, color: warning, size: 18),
        SizedBox(width: 8),
        Expanded(
          child: Text(
            'HerbaNusa hanya mengenal 100 spesies dan bisa salah. Bukan alat diagnosis, '
            'bukan pengganti dokter atau apoteker. Anda bertanggung jawab atas keputusan untuk mengonsumsi tanaman dan risikonya.',
            style: TextStyle(color: ink, fontSize: 11.5, height: 1.45),
          ),
        ),
      ],
    ),
  );
}
