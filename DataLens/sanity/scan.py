from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import ImageEntry, MissingImageEntry, ScanResult


FILENAME_CANDIDATES = [
    "filename",
    "file",
    "image",
    "image_path",
    "path",
    "img",
    "id",
]

LABEL_CANDIDATES = [
    "label",
    "class",
    "category",
    "target",
    "y",
]


def _normalize_exts(allowed_exts: List[str]) -> List[str]:
    return sorted({ext.lower() for ext in allowed_exts})


def list_images(images_dir: str, allowed_exts: List[str]) -> List[str]:
    exts = set(_normalize_exts(allowed_exts))
    root = Path(images_dir)
    if not root.exists():
        return []
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts:
            paths.append(str(path))
    return sorted(paths)


def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1", dtype=str)


def _choose_column(
    columns: List[str], candidates: List[str]
) -> Tuple[Optional[str], List[str]]:
    lowered = [col.lower() for col in columns]
    matches = []
    for cand in candidates:
        if cand in lowered:
            idx = lowered.index(cand)
            matches.append(columns[idx])
    if matches:
        return matches[0], matches

    partials = []
    for i, col in enumerate(lowered):
        for cand in candidates:
            if cand in col:
                partials.append(columns[i])
                break
    if partials:
        return partials[0], partials

    return None, []


def _resolve_relative(root: str, images_dir: str, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    if raw.parent != Path("."):
        return Path(root) / raw
    return Path(images_dir) / raw


def normalize_label(value: object, enabled: bool) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    if enabled:
        return text.lower()
    return text




def _extract_ext(value: str) -> Optional[str]:
    suffix = Path(value).suffix.lower()
    if suffix:
        return suffix
    return None


def scan_mode_a(
    dataset_root: str,
    images_dir: str,
    allowed_exts: List[str],
    infer_classes: bool,
) -> ScanResult:
    images_root = os.path.join(dataset_root, images_dir)
    image_paths = list_images(images_root, allowed_exts)
    images = []
    missing_label_count = 0
    for path in image_paths:
        rel = os.path.relpath(path, images_root)
        label = None
        if infer_classes:
            parts = rel.split(os.sep)
            if len(parts) > 1:
                label = parts[0]
            else:
                missing_label_count += 1
        images.append(ImageEntry(path=path, rel_path=rel, label=label))

    return ScanResult(
        images=images,
        missing_images=[],
        orphan_images=[],
        csv_warnings=[],
        filename_col=None,
        label_col=None,
        missing_label_count=missing_label_count,
        csv_row_count=0,
        resolved_images=len(images),
        total_images_scanned=len(image_paths),
        csv_ext_counts={},
        csv_no_ext_count=0,
    )


def scan_mode_b(
    dataset_root: str,
    images_dir: str,
    allowed_exts: List[str],
    csv_path: str,
    filename_col_override: Optional[str],
    label_col_override: Optional[str],
    ids_without_ext: bool,
    normalize_labels: bool,
) -> ScanResult:
    csv_full_path = csv_path
    if not os.path.isabs(csv_path):
        csv_full_path = os.path.join(dataset_root, csv_path)

    df = read_csv_with_fallback(csv_full_path)
    columns = list(df.columns)
    warnings: List[str] = []

    filename_col = filename_col_override or None
    label_col = label_col_override or None

    if not filename_col:
        filename_col, filename_matches = _choose_column(columns, FILENAME_CANDIDATES)
        if len(filename_matches) > 1:
            warnings.append(
                "Multiple possible filename columns found. Using '%s'." % filename_col
            )
    if not label_col:
        label_col, label_matches = _choose_column(columns, LABEL_CANDIDATES)
        if len(label_matches) > 1:
            warnings.append(
                "Multiple possible label columns found. Using '%s'." % label_col
            )

    if not filename_col:
        raise ValueError("Could not detect filename column. Use the override input.")

    images_root = os.path.join(dataset_root, images_dir)
    all_images = list_images(images_root, allowed_exts)
    allowed = {ext.lower() for ext in allowed_exts}

    stem_map: Dict[str, List[str]] = {}
    if ids_without_ext:
        for path in all_images:
            stem = Path(path).stem.lower()
            stem_map.setdefault(stem, []).append(path)
        for key in stem_map:
            stem_map[key] = sorted(stem_map[key])

    images: List[ImageEntry] = []
    missing_images: List[MissingImageEntry] = []
    referenced = set()
    csv_ext_counts: Dict[str, int] = {}
    csv_no_ext_count = 0
    missing_label_count = 0

    for row_index, row in df.iterrows():
        raw_value = row[filename_col]
        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
            missing_images.append(MissingImageEntry(row_index=int(row_index), reference="<empty>"))
            missing_label = normalize_label(row[label_col], normalize_labels) if label_col else None
            if missing_label is None:
                missing_label_count += 1
            continue
        value = str(raw_value).strip()
        if value == "" or value.lower() == "nan":
            missing_images.append(MissingImageEntry(row_index=int(row_index), reference="<empty>"))
            missing_label = normalize_label(row[label_col], normalize_labels) if label_col else None
            if missing_label is None:
                missing_label_count += 1
            continue

        ext = _extract_ext(value)
        if ext:
            csv_ext_counts[ext] = csv_ext_counts.get(ext, 0) + 1
        else:
            csv_no_ext_count += 1

        resolved: Optional[Path] = None
        candidate = _resolve_relative(dataset_root, images_root, value)
        if candidate.suffix == "" and ids_without_ext:
            stem = candidate.stem.lower()
            matches = stem_map.get(stem, [])
            if len(matches) > 1:
                warnings.append(
                    "Multiple files match id '%s'. Using '%s'." % (value, matches[0])
                )
            if matches:
                resolved = Path(matches[0])
        else:
            resolved = candidate

        if resolved is None:
            missing_images.append(MissingImageEntry(row_index=int(row_index), reference=value))
            missing_label = normalize_label(row[label_col], normalize_labels) if label_col else None
            if missing_label is None:
                missing_label_count += 1
            continue
        if resolved.suffix.lower() not in allowed or not resolved.exists():
            missing_images.append(MissingImageEntry(row_index=int(row_index), reference=value))
            missing_label = normalize_label(row[label_col], normalize_labels) if label_col else None
            if missing_label is None:
                missing_label_count += 1
            continue

        label = normalize_label(row[label_col], normalize_labels) if label_col else None
        if label is None:
            missing_label_count += 1
        rel = os.path.relpath(str(resolved), images_root) if str(resolved).startswith(images_root) else str(resolved)
        images.append(ImageEntry(path=str(resolved), rel_path=rel, label=label))
        referenced.add(str(resolved))

    orphan_images = sorted([path for path in all_images if path not in referenced])
    images = sorted(images, key=lambda item: item.path)

    return ScanResult(
        images=images,
        missing_images=sorted(
            missing_images, key=lambda entry: (entry.row_index, entry.reference)
        ),
        orphan_images=orphan_images,
        csv_warnings=warnings,
        filename_col=filename_col,
        label_col=label_col,
        missing_label_count=missing_label_count,
        csv_row_count=len(df),
        resolved_images=len(images),
        total_images_scanned=len(all_images),
        csv_ext_counts=dict(sorted(csv_ext_counts.items(), key=lambda item: item[0])),
        csv_no_ext_count=csv_no_ext_count,
    )
