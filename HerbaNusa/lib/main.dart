import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'screens/encyclopedia_screen.dart';
import 'screens/history_screen.dart';
import 'screens/home_screen.dart';
import 'screens/identify_screen.dart';
import 'screens/onboarding_screen.dart';
import 'screens/profile_screen.dart';
import 'services/plant_catalog.dart';
import 'services/plant_classifier.dart';
import 'services/history_store.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const HerbaNusaApp());
}

/// Services shared by every screen, created once at startup.
class AppServices extends InheritedWidget {
  const AppServices({
    super.key,
    required this.catalog,
    required this.classifier,
    required this.history,
    required super.child,
  });

  final PlantCatalog catalog;
  final PlantClassifier classifier;
  final HistoryStore history;

  static AppServices of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<AppServices>()!;

  @override
  bool updateShouldNotify(AppServices oldWidget) => false;
}

class HerbaNusaApp extends StatefulWidget {
  const HerbaNusaApp({super.key});

  @override
  State<HerbaNusaApp> createState() => _HerbaNusaAppState();
}

class _HerbaNusaAppState extends State<HerbaNusaApp> {
  late Future<(PlantCatalog, PlantClassifier)> _startup = _load();

  Future<(PlantCatalog, PlantClassifier)> _load() async {
    final catalog = await PlantCatalog.load();
    final classifier = await PlantClassifier.load(
      labels: catalog.labels,
      threshold: catalog.modelCard.threshold,
    );
    return (catalog, classifier);
  }

  final _history = HistoryStore();

  MaterialApp _app(Widget home) => MaterialApp(
    title: 'HerbaNusa',
    debugShowCheckedModeBanner: false,
    theme: buildTheme(),
    home: home,
    // Dark status-bar icons by default; the species page overrides this over its photo.
    builder: (context, child) => AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.dark,
      child: child!,
    ),
  );

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: _startup,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return _app(
            _StartupError(
              message: '${snapshot.error}',
              onRetry: () => setState(() => _startup = _load()),
            ),
          );
        }
        if (!snapshot.hasData) {
          return _app(
            const Scaffold(body: Center(child: CircularProgressIndicator())),
          );
        }
        final (catalog, classifier) = snapshot.data!;
        // Above MaterialApp so that pushed routes can reach the services too:
        // routes are children of the Navigator, not of `home`.
        return AppServices(
          catalog: catalog,
          classifier: classifier,
          history: _history,
          child: _app(const _Shell()),
        );
      },
    );
  }
}

class _StartupError extends StatelessWidget {
  const _StartupError({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 48),
              const SizedBox(height: 12),
              const Text(
                'Model gagal dimuat',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
              ),
              const SizedBox(height: 8),
              Text(message, textAlign: TextAlign.center),
              const SizedBox(height: 16),
              FilledButton(onPressed: onRetry, child: const Text('Coba lagi')),
            ],
          ),
        ),
      ),
    );
  }
}

class _Shell extends StatefulWidget {
  const _Shell();

  @override
  State<_Shell> createState() => _ShellState();
}

class _ShellState extends State<_Shell> {
  static const _onboardedKey = 'onboarded';
  int _tab = 0;
  int _historyVersion = 0;
  bool? _onboarded;

  @override
  void initState() {
    super.initState();
    SharedPreferences.getInstance().then((prefs) {
      if (mounted) {
        setState(() => _onboarded = prefs.getBool(_onboardedKey) ?? false);
      }
    });
  }

  Future<void> _finishOnboarding() async {
    setState(() => _onboarded = true);
    (await SharedPreferences.getInstance()).setBool(_onboardedKey, true);
  }

  void _historyChanged() => setState(() => _historyVersion++);

  @override
  Widget build(BuildContext context) {
    if (_onboarded == null) return const Scaffold();
    if (!_onboarded!) return OnboardingScreen(onStart: _finishOnboarding);
    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: [
          HomeScreen(
            historyVersion: _historyVersion,
            onSelectTab: (tab) => setState(() => _tab = tab),
          ),
          const EncyclopediaScreen(),
          IdentifyScreen(onAnalyzed: _historyChanged),
          HistoryScreen(
            historyVersion: _historyVersion,
            onChanged: _historyChanged,
          ),
          ProfileScreen(historyVersion: _historyVersion),
        ],
      ),
      bottomNavigationBar: _BottomBar(
        selected: _tab,
        onSelect: (tab) => setState(() => _tab = tab),
      ),
    );
  }
}

/// Tab bar with the raised scan button in the middle.
class _BottomBar extends StatelessWidget {
  const _BottomBar({required this.selected, required this.onSelect});

  final int selected;
  final ValueChanged<int> onSelect;

  static const _tabs = [
    (Icons.home_outlined, Icons.home_rounded, 'Beranda'),
    (Icons.local_florist_outlined, Icons.local_florist, 'Katalog'),
    (Icons.document_scanner_outlined, Icons.document_scanner, 'Identifikasi'),
    (Icons.history, Icons.history, 'Riwayat'),
    (Icons.person_outline, Icons.person_rounded, 'Profil'),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        border: const Border(top: BorderSide(color: line)),
        boxShadow: [
          BoxShadow(
            color: ink.withValues(alpha: 0.05),
            blurRadius: 16,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 68,
          child: Row(
            children: [
              for (final (index, (icon, activeIcon, label)) in _tabs.indexed)
                Expanded(
                  child: Semantics(
                    selected: index == selected,
                    button: true,
                    label: label,
                    child: InkResponse(
                      onTap: () => onSelect(index),
                      child: index == 2
                          ? _scanTab(index == selected, label)
                          : _tab(index == selected, icon, activeIcon, label),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _label(String label, bool active) => Text(
    label,
    style: TextStyle(
      fontSize: 11,
      fontWeight: active ? FontWeight.w700 : FontWeight.w500,
      color: active ? accentDeep : inkMuted,
    ),
  );

  Widget _tab(bool active, IconData icon, IconData activeIcon, String label) =>
      Column(
        children: [
          Container(
            height: 3,
            width: 28,
            color: active ? accent : Colors.transparent,
          ),
          const Spacer(),
          Icon(
            active ? activeIcon : icon,
            color: active ? accentDeep : inkMuted,
            size: 24,
          ),
          const SizedBox(height: 4),
          _label(label, active),
          const Spacer(),
        ],
      );

  Widget _scanTab(bool active, String label) => Column(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      Container(
        width: 46,
        height: 46,
        decoration: BoxDecoration(
          color: active ? accentDeep : accent,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: accent.withValues(alpha: 0.35),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: const Icon(
          Icons.document_scanner_outlined,
          color: Colors.white,
          size: 24,
        ),
      ),
      const SizedBox(height: 3),
      _label(label, active),
    ],
  );
}
