from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .models import (
    CheckResult,
    DuplicateGroup,
    HygieneResult,
    ImageEntry,
    ImageStat,
    StatsResult,
)


IMBALANCE_THRESHOLD = 0.1
SMALL_RES_THRESHOLD = 64
ASPECT_OUTLIER_THRESHOLD = 3.0
RGBA_SHARE_THRESHOLD = 0.3
MODE_VARIANCE_THRESHOLD = 0.6
EXT_MISMATCH_THRESHOLD = 0.5


def _verify_image(path: str) -> Optional[str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return str(exc)


def find_corrupted(paths: List[str]) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    if not paths:
        return results
    with ThreadPoolExecutor() as executor:
        for path, error in zip(paths, executor.map(_verify_image, paths)):
            if error:
                results.append((path, error))
    return results


def _hash_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_quick(path: str) -> str:
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        first = handle.read(64 * 1024)
        if size > 128 * 1024:
            handle.seek(-64 * 1024, os.SEEK_END)
            last = handle.read(64 * 1024)
        else:
            last = b""
    hasher = hashlib.sha256()
    hasher.update(str(size).encode("utf-8"))
    hasher.update(first)
    hasher.update(last)
    return hasher.hexdigest()


def _hash_phash(path: str) -> str:
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            if hasattr(Image, "Resampling"):
                img = img.resize((9, 8), Image.Resampling.LANCZOS)
            else:
                img = img.resize((9, 8), Image.ANTIALIAS)
            pixels = list(img.getdata())
        diff = []
        for row in range(8):
            row_start = row * 9
            for col in range(8):
                left = pixels[row_start + col]
                right = pixels[row_start + col + 1]
                diff.append(1 if left > right else 0)
        value = 0
        for bit in diff:
            value = (value << 1) | bit
        return "%016x" % value
    except Exception:  # pylint: disable=broad-except
        return "error-%s" % hashlib.sha256(path.encode("utf-8")).hexdigest()


def find_duplicates(paths: List[str], method: str) -> List[DuplicateGroup]:
    if not paths:
        return []
    if method not in {"sha256", "quick", "phash"}:
        raise ValueError("Unknown hash method: %s" % method)

    if method == "sha256":
        hash_fn = _hash_sha256
    elif method == "quick":
        hash_fn = _hash_quick
    else:
        hash_fn = _hash_phash
    hashes: Dict[str, List[str]] = {}
    with ThreadPoolExecutor() as executor:
        for path, digest in zip(paths, executor.map(hash_fn, paths)):
            hashes.setdefault(digest, []).append(path)

    groups = []
    for digest, items in hashes.items():
        if len(items) > 1:
            groups.append(DuplicateGroup(hash_value=digest, paths=sorted(items)))
    groups.sort(key=lambda group: (-len(group.paths), group.hash_value))
    return groups


def extension_counts(paths: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def collect_image_details(paths: List[str]) -> List[ImageStat]:
    details: List[ImageStat] = []

    def _stats_for_path(path: str) -> Optional[ImageStat]:
        try:
            with Image.open(path) as img:
                return ImageStat(path=path, width=img.width, height=img.height, mode=img.mode)
        except Exception:  # pylint: disable=broad-except
            return None

    with ThreadPoolExecutor() as executor:
        for result in executor.map(_stats_for_path, paths):
            if result is None:
                continue
            details.append(result)
    return details


def build_stats(details: List[ImageStat]) -> Optional[StatsResult]:
    widths: List[int] = [item.width for item in details]
    heights: List[int] = [item.height for item in details]
    mode_counts: Dict[str, int] = {}
    for item in details:
        mode_counts[item.mode] = mode_counts.get(item.mode, 0) + 1

    if not widths or not heights:
        return None

    widths.sort()
    heights.sort()

    return StatsResult(
        width_min=widths[0],
        width_median=int(median(widths)),
        width_max=widths[-1],
        height_min=heights[0],
        height_median=int(median(heights)),
        height_max=heights[-1],
        mode_counts=dict(sorted(mode_counts.items(), key=lambda item: item[0])),
    )


def analyze_hygiene(details: List[ImageStat]) -> Optional[HygieneResult]:
    if not details:
        return None
    small_res_paths: List[str] = []
    aspect_outliers: List[Tuple[float, str]] = []
    mode_counts: Dict[str, int] = {}
    rgba_count = 0

    for item in details:
        if min(item.width, item.height) < SMALL_RES_THRESHOLD:
            small_res_paths.append(item.path)
        ratio = max(item.width / item.height, item.height / item.width)
        if ratio > ASPECT_OUTLIER_THRESHOLD:
            aspect_outliers.append((ratio, item.path))
        mode_counts[item.mode] = mode_counts.get(item.mode, 0) + 1
        if item.mode == "RGBA":
            rgba_count += 1

    total = len(details)
    rgba_share = (rgba_count / total) if total else 0.0
    sorted_outliers = sorted(aspect_outliers, key=lambda item: (-item[0], item[1]))
    aspect_examples = [path for _, path in sorted_outliers[:10]]
    top_mode_share = max(mode_counts.values()) / total if total else 0.0
    mode_variance_warning = len(mode_counts) >= 3 and top_mode_share < MODE_VARIANCE_THRESHOLD

    return HygieneResult(
        small_res_count=len(small_res_paths),
        small_res_examples=sorted(small_res_paths)[:10],
        aspect_outlier_count=len(aspect_outliers),
        aspect_outlier_examples=aspect_examples,
        rgba_share=rgba_share,
        mode_variance_warning=mode_variance_warning,
    )


def label_counts(entries: List[ImageEntry]) -> Optional[Dict[str, int]]:
    counts: Dict[str, int] = {}
    for entry in entries:
        if entry.label is None or str(entry.label).strip() == "":
            continue
        counts[entry.label] = counts.get(entry.label, 0) + 1
    if not counts:
        return None
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def check_imbalance(counts: Optional[Dict[str, int]]) -> bool:
    if not counts or len(counts) < 2:
        return False
    values = list(counts.values())
    min_count = min(values)
    max_count = max(values)
    if max_count == 0:
        return False
    return (min_count / max_count) < IMBALANCE_THRESHOLD


def run_checks(
    entries: List[ImageEntry],
    hash_method: str,
    compute_stats: bool,
) -> CheckResult:
    paths = sorted({entry.path for entry in entries})
    corrupted = find_corrupted(paths)
    duplicates = find_duplicates(paths, hash_method)
    ext_counts = extension_counts(paths)
    details = collect_image_details(paths) if compute_stats else []
    stats = build_stats(details) if compute_stats else None
    hygiene = analyze_hygiene(details) if compute_stats else None
    counts = label_counts(entries)
    imbalance = check_imbalance(counts)

    return CheckResult(
        corrupted=corrupted,
        duplicates=duplicates,
        ext_counts=ext_counts,
        stats=stats,
        label_counts=counts,
        imbalance_warning=imbalance,
        hygiene=hygiene,
    )


def extension_mismatch_warnings(
    image_ext_counts: Dict[str, int],
    csv_ext_counts: Dict[str, int],
    allowed_exts: List[str],
) -> List[str]:
    warnings: List[str] = []
    total_csv = sum(csv_ext_counts.values())
    if total_csv == 0:
        return warnings
    total_images = sum(image_ext_counts.values())
    top_csv_ext, top_csv_count = max(csv_ext_counts.items(), key=lambda item: item[1])
    top_img_ext, top_img_count = ("", 0)
    if image_ext_counts:
        top_img_ext, top_img_count = max(
            image_ext_counts.items(), key=lambda item: item[1]
        )
    csv_share = top_csv_count / total_csv if total_csv else 0.0
    img_share = top_img_count / total_images if total_images else 0.0
    if (
        top_img_ext
        and top_csv_ext != top_img_ext
        and csv_share >= EXT_MISMATCH_THRESHOLD
        and img_share >= EXT_MISMATCH_THRESHOLD
    ):
        warnings.append(
            "CSV references '%s' heavily (%.0f%%) but images are mostly '%s' (%.0f%%)."
            % (top_csv_ext, csv_share * 100, top_img_ext, img_share * 100)
        )

    allowed = {ext.lower() for ext in allowed_exts}
    excluded_count = sum(
        count for ext, count in csv_ext_counts.items() if ext not in allowed
    )
    if excluded_count / total_csv >= EXT_MISMATCH_THRESHOLD:
        warnings.append(
            "CSV references extensions not in allowed list (%.0f%% of rows)."
            % (excluded_count / total_csv * 100)
        )
    return warnings
