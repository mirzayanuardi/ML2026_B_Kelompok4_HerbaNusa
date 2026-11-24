import 'package:flutter/material.dart';

// Layout and type follow MeowScan; the palette is retuned to forest greens.
const Color ink = Color(0xFF1E3326);
const Color inkMuted = Color(0xFF66766B);
const Color inkFaint = Color(0xFF93A197);
const Color accent = Color(0xFF2F8F5B);
const Color accentDeep = Color(0xFF1F6B42);
const Color cream = Color(0xFFF6FAF2);
const Color sand = Color(0xFFE8F1E2);
const Color mint = Color(0xFFDDEFE1);
const Color butter = Color(0xFFF3EDCC);
const Color line = Color(0xFFDFE9DA);
const Color success = Color(0xFF3E8E52);
const Color successSoft = Color(0xFFE6F3E8);
const Color warning = Color(0xFFC0661A);
const Color warningSoft = Color(0xFFFBEEDB);
const Color danger = Color(0xFFC0392B);
const Color dangerSoft = Color(0xFFF9E3E0);
const Color herbal = Color(0xFF9A7A0C);
const Color herbalSoft = Color(0xFFF7F0D2);
const Color neutral = Color(0xFF6F7A72);
const Color neutralSoft = Color(0xFFEDF0EE);
const List<Color> heroGradient = [Color(0xFF1F5C3B), Color(0xFF3C8F5E)];

const String headingFont = 'Nunito';
const String bodyFont = 'DMSans';

/// Heading style in Nunito, the prototype's display face.
TextStyle heading(
  double size, {
  FontWeight weight = FontWeight.w700,
  Color color = ink,
}) => TextStyle(
  fontFamily: headingFont,
  fontSize: size,
  fontWeight: weight,
  color: color,
  height: 1.2,
);

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: accent,
    primary: accent,
    onPrimary: Colors.white,
    secondary: accentDeep,
    surface: cream,
    onSurface: ink,
    onSurfaceVariant: inkMuted,
    outline: line,
    outlineVariant: line,
  );
  final base = ThemeData(
    colorScheme: scheme,
    fontFamily: bodyFont,
    useMaterial3: true,
  );
  final shape = RoundedRectangleBorder(borderRadius: BorderRadius.circular(18));
  return base.copyWith(
    scaffoldBackgroundColor: cream,
    textTheme: base.textTheme
        .apply(bodyColor: ink, displayColor: ink)
        .copyWith(
          headlineMedium: heading(26, weight: FontWeight.w800),
          headlineSmall: heading(22, weight: FontWeight.w800),
          titleLarge: heading(19),
          titleMedium: heading(16),
          bodySmall: const TextStyle(
            fontFamily: bodyFont,
            fontSize: 12,
            color: inkMuted,
            height: 1.4,
          ),
        ),
    appBarTheme: AppBarTheme(
      backgroundColor: cream,
      foregroundColor: ink,
      scrolledUnderElevation: 0,
      centerTitle: true,
      titleTextStyle: heading(17),
    ),
    cardTheme: CardThemeData(
      color: Colors.white,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: const BorderSide(color: line),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: accent,
        foregroundColor: Colors.white,
        minimumSize: const Size.fromHeight(54),
        textStyle: const TextStyle(
          fontFamily: bodyFont,
          fontWeight: FontWeight.w700,
          fontSize: 15,
        ),
        shape: shape,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: accentDeep,
        backgroundColor: Colors.white,
        minimumSize: const Size.fromHeight(54),
        side: const BorderSide(color: accent),
        textStyle: const TextStyle(
          fontFamily: bodyFont,
          fontWeight: FontWeight.w700,
          fontSize: 15,
        ),
        shape: shape,
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: accentDeep,
        textStyle: const TextStyle(
          fontFamily: bodyFont,
          fontWeight: FontWeight.w700,
          fontSize: 13,
        ),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Colors.white,
      hintStyle: const TextStyle(color: inkFaint),
      prefixIconColor: inkMuted,
      contentPadding: const EdgeInsets.symmetric(vertical: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: line),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: line),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: accent),
      ),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: accent,
      linearTrackColor: sand,
    ),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: ink,
      behavior: SnackBarBehavior.floating,
    ),
  );
}
