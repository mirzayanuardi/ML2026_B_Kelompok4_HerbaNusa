import 'package:flutter/material.dart';

import '../models/plant.dart';
import '../theme.dart';

/// Bundled photo for a species, as written by tool/build_assets.py: the label
/// lowercased, non-alphanumeric runs as `_`, edges trimmed.
String plantPhotoAsset(String name) {
  final slug = name
      .toLowerCase()
      .replaceAll(RegExp('[^a-z0-9]+'), '_')
      .replaceAll(RegExp(r'^_+|_+$'), '');
  return 'assets/plants/$slug.jpg';
}

class PlantPhoto extends StatelessWidget {
  const PlantPhoto({super.key, required this.plant, this.fit = BoxFit.cover});

  final Plant plant;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) => Image.asset(
    plantPhotoAsset(plant.name),
    fit: fit,
    // Dataset photos are 128 px; smooth upscaling avoids blocky catalog cards.
    filterQuality: FilterQuality.medium,
    errorBuilder: (_, _, _) => const ColoredBox(
      color: sand,
      child: Center(child: Icon(Icons.eco, color: inkFaint)),
    ),
  );
}

/// Rounded tinted square holding an icon, as on the prototype's list rows.
class IconBadge extends StatelessWidget {
  const IconBadge({
    super.key,
    required this.icon,
    this.color = mint,
    this.iconColor = accentDeep,
    this.size = 44,
  });

  final IconData icon;
  final Color color;
  final Color iconColor;
  final double size;

  @override
  Widget build(BuildContext context) => Container(
    width: size,
    height: size,
    decoration: BoxDecoration(
      color: color,
      borderRadius: BorderRadius.circular(size * 0.32),
    ),
    child: Icon(icon, color: iconColor, size: size * 0.5),
  );
}

/// Small uppercase label in a soft pill ("LANGKAH 1 DARI 2").
class Pill extends StatelessWidget {
  const Pill(
    this.text, {
    super.key,
    this.color = mint,
    this.textColor = accentDeep,
  });

  final String text;
  final Color color;
  final Color textColor;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
      color: color,
      borderRadius: BorderRadius.circular(999),
    ),
    child: Text(
      text.toUpperCase(),
      style: TextStyle(
        color: textColor,
        fontSize: 10,
        fontWeight: FontWeight.w800,
        letterSpacing: 0.8,
      ),
    ),
  );
}

/// Uppercase overline in the accent colour.
class Overline extends StatelessWidget {
  const Overline(this.text, {super.key, this.color = accentDeep});

  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) => Text(
    text.toUpperCase(),
    style: TextStyle(
      color: color,
      fontSize: 10,
      fontWeight: FontWeight.w800,
      letterSpacing: 1,
    ),
  );
}

/// White rounded square button with a border, used for back/info actions.
class SquareIconButton extends StatelessWidget {
  const SquareIconButton({
    super.key,
    required this.icon,
    required this.onPressed,
    this.tooltip,
    this.dark = false,
  });

  final IconData icon;
  final VoidCallback onPressed;
  final String? tooltip;
  final bool dark;

  @override
  Widget build(BuildContext context) => Material(
    color: dark ? Colors.black.withValues(alpha: 0.5) : Colors.white,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(14),
      side: dark ? BorderSide.none : const BorderSide(color: line),
    ),
    child: IconButton(
      tooltip: tooltip,
      onPressed: onPressed,
      icon: Icon(icon, size: 20, color: dark ? Colors.white : ink),
    ),
  );
}

/// Centered title with a square back button, the prototype's page header.
class PageHeader extends StatelessWidget {
  const PageHeader({
    super.key,
    required this.title,
    this.trailing,
    this.onBack,
  });

  final String title;
  final Widget? trailing;
  final VoidCallback? onBack;

  @override
  Widget build(BuildContext context) {
    final canPop = onBack != null || Navigator.of(context).canPop();
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
      child: Row(
        children: [
          SizedBox(
            width: 48,
            child: canPop
                ? SquareIconButton(
                    icon: Icons.chevron_left,
                    tooltip: 'Kembali',
                    onPressed: onBack ?? () => Navigator.of(context).maybePop(),
                  )
                : null,
          ),
          Expanded(
            child: Text(title, textAlign: TextAlign.center, style: heading(17)),
          ),
          SizedBox(width: 48, child: trailing),
        ],
      ),
    );
  }
}

/// Section heading with an optional "Lihat semua" action.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.title, {super.key, this.action, this.onAction});

  final String title;
  final String? action;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Expanded(child: Text(title, style: heading(18))),
      if (action != null)
        TextButton(
          onPressed: onAction,
          style: TextButton.styleFrom(
            padding: EdgeInsets.zero,
            minimumSize: const Size(0, 32),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(action!),
              const Icon(Icons.chevron_right, size: 18),
            ],
          ),
        ),
    ],
  );
}

/// White card with the prototype's soft border.
class SoftCard extends StatelessWidget {
  const SoftCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.onTap,
    this.color = Colors.white,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final VoidCallback? onTap;
  final Color color;

  @override
  Widget build(BuildContext context) => Material(
    color: color,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
      side: color == Colors.white
          ? const BorderSide(color: line)
          : BorderSide.none,
    ),
    clipBehavior: Clip.antiAlias,
    child: InkWell(
      onTap: onTap,
      child: Padding(padding: padding, child: child),
    ),
  );
}

/// Row inside a grouped settings-style card.
class MenuRow extends StatelessWidget {
  const MenuRow({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.onTap,
    this.trailing,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) => InkWell(
    onTap: onTap,
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Row(
        children: [
          IconBadge(icon: icon, size: 40),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 2),
                Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
          trailing ??
              (onTap == null
                  ? const SizedBox.shrink()
                  : const Icon(Icons.chevron_right, color: ink)),
        ],
      ),
    ),
  );
}

/// Card holding MenuRows separated by thin lines.
class MenuGroup extends StatelessWidget {
  const MenuGroup({super.key, required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) => SoftCard(
    padding: EdgeInsets.zero,
    child: Column(
      children: [
        for (final (index, child) in children.indexed) ...[
          if (index > 0)
            const Divider(height: 1, color: line, indent: 14, endIndent: 14),
          child,
        ],
      ],
    ),
  );
}

/// Three stats separated by thin vertical lines ("Tenang | 12–17 th | Sedang").
class StatStrip extends StatelessWidget {
  const StatStrip({
    super.key,
    required this.items,
    this.color = sand,
    this.labelFirst = false,
  });

  final List<(String value, String label)> items;
  final Color color;
  final bool labelFirst;

  @override
  Widget build(BuildContext context) {
    final small = Theme.of(context).textTheme.bodySmall?.copyWith(fontSize: 11);
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 14),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(18),
        border: color == Colors.white ? Border.all(color: line) : null,
        boxShadow: color == Colors.white
            ? [
                BoxShadow(
                  color: ink.withValues(alpha: 0.06),
                  blurRadius: 18,
                  offset: const Offset(0, 6),
                ),
              ]
            : null,
      ),
      child: IntrinsicHeight(
        child: Row(
          children: [
            for (final (index, (value, label)) in items.indexed) ...[
              if (index > 0)
                const VerticalDivider(
                  width: 1,
                  color: line,
                  indent: 4,
                  endIndent: 4,
                ),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 6),
                  child: Column(
                    children: [
                      if (labelFirst)
                        Text(label, style: small, textAlign: TextAlign.center),
                      if (labelFirst) const SizedBox(height: 4),
                      Text(
                        value,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 14,
                        ),
                      ),
                      if (!labelFirst) const SizedBox(height: 4),
                      if (!labelFirst)
                        Text(label, style: small, textAlign: TextAlign.center),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
