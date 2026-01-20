from __future__ import annotations

from typing import Dict, List, Optional

from .checks import (
    ASPECT_OUTLIER_THRESHOLD,
    RGBA_SHARE_THRESHOLD,
    SMALL_RES_THRESHOLD,
)

from .models import CheckResult, ScanConfig, ScanResult


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    config: ScanConfig,
    scan: ScanResult,
    checks: CheckResult,
    warnings: Dict[str, List[str]],
    top_n: int = 20,
) -> str:
    lines: List[str] = []
    lines.append("# Dataset Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- Mode: %s" % config.mode)
    lines.append("- Dataset root: %s" % config.dataset_root)
    lines.append("- Images folder: %s" % config.images_dir)
    lines.append("- Allowed extensions: %s" % " ".join(config.allowed_exts))
    lines.append("- Hashing method: %s" % config.hash_method)
    lines.append("- Compute stats: %s" % ("yes" if config.compute_stats else "no"))
    if config.mode == "Images Only":
        lines.append("- Infer classes: %s" % ("yes" if config.infer_classes else "no"))
    else:
        lines.append("- CSV path: %s" % (config.csv_path or ""))
        lines.append("- Filename column: %s" % (scan.filename_col or ""))
        lines.append("- Label column: %s" % (scan.label_col or ""))
        lines.append("- IDs without extension: %s" % ("yes" if config.ids_without_ext else "no"))
        lines.append(
            "- Normalize labels: %s" % ("yes" if config.normalize_labels else "no")
        )

    lines.append("")
    lines.append("## Config Fingerprint")
    lines.append("")
    mode_flag = "A" if config.mode == "Images Only" else "B"
    lines.append("- Mode: %s" % mode_flag)
    lines.append("- Images path: %s" % config.images_dir)
    if config.mode == "Images + Labels (CSV)":
        lines.append("- CSV path: %s" % (config.csv_path or ""))
        lines.append("- Filename column: %s" % (scan.filename_col or ""))
        lines.append("- Label column: %s" % (scan.label_col or ""))
    lines.append("- Normalize labels: %s" % ("yes" if config.normalize_labels else "no"))
    lines.append("- Extensions: %s" % " ".join(config.allowed_exts))
    lines.append("- Duplicate method: %s" % config.hash_method)
    lines.append("- IDs without extension: %s" % ("yes" if config.ids_without_ext else "no"))
    lines.append(
        "- Thresholds: small_res=%d, aspect_outlier=%.1f, rgba_share=%.2f"
        % (SMALL_RES_THRESHOLD, ASPECT_OUTLIER_THRESHOLD, RGBA_SHARE_THRESHOLD)
    )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Total images: %d" % len(scan.images))
    lines.append("- Corrupted images: %d" % len(checks.corrupted))
    lines.append("- Duplicate groups: %d" % len(checks.duplicates))
    if config.mode == "Images Only" and not config.infer_classes:
        lines.append("- Unlabeled dataset: yes")
    else:
        lines.append("- Missing labels: %d" % scan.missing_label_count)
    if config.mode == "Images + Labels (CSV)":
        lines.append("- Missing images: %d" % len(scan.missing_images))
        lines.append("- Orphan images: %d" % len(scan.orphan_images))

    lines.append("")
    lines.append("## Top Warnings")
    lines.append("")
    has_warnings = False
    for category in sorted(warnings.keys()):
        items = warnings[category]
        if not items:
            continue
        has_warnings = True
        lines.append("### %s" % category)
        for item in items:
            lines.append("- %s" % item)
        lines.append("")
    if not has_warnings:
        lines.append("- None")

    if config.mode == "Images + Labels (CSV)":
        lines.append("")
        lines.append("## Coverage / Orphans")
        lines.append("")
        coverage_ratio = (
            scan.resolved_images / scan.csv_row_count if scan.csv_row_count else 0.0
        )
        orphan_rate = (
            len(scan.orphan_images) / scan.total_images_scanned
            if scan.total_images_scanned
            else 0.0
        )
        lines.append(
            "- Coverage: %.1f%% (%d/%d)"
            % (coverage_ratio * 100, scan.resolved_images, scan.csv_row_count)
        )
        lines.append(
            "- Orphan rate: %.1f%% (%d/%d)"
            % (
                orphan_rate * 100,
                len(scan.orphan_images),
                scan.total_images_scanned,
            )
        )

    if checks.label_counts:
        lines.append("")
        lines.append("## Class Distribution")
        lines.append("")
        rows = [[label, str(count)] for label, count in checks.label_counts.items()]
        lines.append(_markdown_table(["Label", "Count"], rows))

    lines.append("")
    lines.append("## Corrupted Images")
    lines.append("")
    if checks.corrupted:
        for path, error in checks.corrupted:
            lines.append("- %s | %s" % (path, error))
    else:
        lines.append("- None")

    if config.mode == "Images + Labels (CSV)":
        lines.append("")
        lines.append("## Missing Images")
        lines.append("")
        if scan.missing_images:
            for entry in scan.missing_images:
                lines.append("- %s | row %d" % (entry.reference, entry.row_index))
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Orphan Images")
        lines.append("")
        if scan.orphan_images:
            for path in scan.orphan_images:
                lines.append("- %s" % path)
        else:
            lines.append("- None")

    lines.append("")
    lines.append("## Duplicate Groups (Method: %s, Top %d)" % (config.hash_method, top_n))
    lines.append("")
    if checks.duplicates:
        for group in checks.duplicates[:top_n]:
            lines.append("- Hash: %s" % group.hash_value)
            for path in group.paths:
                lines.append("  - %s" % path)
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)
