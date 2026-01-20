from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from sanity.checks import (
    ASPECT_OUTLIER_THRESHOLD,
    IMBALANCE_THRESHOLD,
    RGBA_SHARE_THRESHOLD,
    SMALL_RES_THRESHOLD,
    analyze_hygiene,
    build_stats,
    check_imbalance,
    collect_image_details,
    extension_mismatch_warnings,
    find_corrupted,
    find_duplicates,
    label_counts,
)
from sanity.models import CheckResult, HygieneResult, ScanConfig
from sanity.report import build_report
from sanity.scan import read_csv_with_fallback, scan_mode_a, scan_mode_b


APP_TITLE = "Image Dataset Sanity Checker"


def _parse_extensions(text: str) -> List[str]:
    parts = [item.strip().lower() for item in text.replace(",", " ").split()]
    exts = []
    for part in parts:
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        exts.append(part)
    return sorted(set(exts))


@st.cache_data
def load_csv_columns(csv_path: str) -> List[str]:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, nrows=5)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", dtype=str, nrows=5)
    return list(df.columns)


@st.cache_data
def cached_scan(config: Dict[str, object]) -> Tuple[object, ScanConfig]:
    scan_config = ScanConfig(
        mode=config["mode"],
        dataset_root=config["dataset_root"],
        images_dir=config["images_dir"],
        allowed_exts=config["allowed_exts"],
        hash_method=config["hash_method"],
        compute_stats=config["compute_stats"],
        infer_classes=config["infer_classes"],
        csv_path=config.get("csv_path"),
        filename_col=config.get("filename_col"),
        label_col=config.get("label_col"),
        ids_without_ext=config.get("ids_without_ext", False),
        normalize_labels=config.get("normalize_labels", False),
    )
    if scan_config.mode == "Images Only":
        scan = scan_mode_a(
            dataset_root=scan_config.dataset_root,
            images_dir=scan_config.images_dir,
            allowed_exts=scan_config.allowed_exts,
            infer_classes=scan_config.infer_classes,
        )
    else:
        scan = scan_mode_b(
            dataset_root=scan_config.dataset_root,
            images_dir=scan_config.images_dir,
            allowed_exts=scan_config.allowed_exts,
            csv_path=scan_config.csv_path or "",
            filename_col_override=scan_config.filename_col,
            label_col_override=scan_config.label_col,
            ids_without_ext=scan_config.ids_without_ext,
            normalize_labels=scan_config.normalize_labels,
        )
    return scan, scan_config


@st.cache_data
def cached_corrupted(paths: List[str]):
    return find_corrupted(paths)


@st.cache_data
def cached_duplicates(paths: List[str], hash_method: str):
    return find_duplicates(paths, hash_method)


@st.cache_data
def cached_image_details(paths: List[str]):
    return collect_image_details(paths)


@st.cache_data
def cached_read_csv(csv_path: str) -> pd.DataFrame:
    return read_csv_with_fallback(csv_path)


def _image_thumbnail(path: str, label: Optional[str]) -> Tuple[Image.Image, str]:
    with Image.open(path) as img:
        img_copy = img.copy()
    img_copy.thumbnail((200, 200))
    caption = os.path.basename(path)
    if label is not None:
        caption = "%s | %s" % (caption, label)
    return img_copy, caption


def _build_issues_csv(scan, checks, hash_method: str) -> bytes:
    rows: List[Dict[str, object]] = []
    for entry in scan.missing_images:
        rows.append(
            {
                "type": "missing",
                "path": "",
                "row_index": entry.row_index,
                "reference": entry.reference,
                "error": "",
                "group_id": "",
                "hash": "",
                "method": "",
            }
        )
    for path in scan.orphan_images:
        rows.append(
            {
                "type": "orphan",
                "path": path,
                "row_index": "",
                "reference": "",
                "error": "",
                "group_id": "",
                "hash": "",
                "method": "",
            }
        )
    for path, error in checks.corrupted:
        rows.append(
            {
                "type": "corrupted",
                "path": path,
                "row_index": "",
                "reference": "",
                "error": error,
                "group_id": "",
                "hash": "",
                "method": "",
            }
        )
    for idx, group in enumerate(checks.duplicates, start=1):
        for path in group.paths:
            rows.append(
                {
                    "type": "duplicate",
                    "path": path,
                    "row_index": "",
                    "reference": "",
                    "error": "",
                    "group_id": idx,
                    "hash": group.hash_value,
                    "method": hash_method,
                }
            )
    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.sidebar.header("Controls")
    mode = st.sidebar.selectbox("Mode", ["Images Only", "Images + Labels (CSV)"])
    dataset_root = st.sidebar.text_input("Dataset root folder", "")
    images_dir = st.sidebar.text_input("Images folder path", "images")
    allowed_exts_text = st.sidebar.text_input(
        "Allowed extensions", ".png .jpg .jpeg .webp"
    )
    hash_method = st.sidebar.selectbox(
        "Duplicate detection method", ["sha256", "quick", "phash"]
    )

    compute_stats = st.sidebar.checkbox("Compute image stats (resolution + mode)", False)
    show_preview = st.sidebar.checkbox("Show sample preview", True)
    export_report = st.sidebar.checkbox("Export report", True)
    top_n = st.sidebar.number_input(
        "Top duplicate groups in report", min_value=1, max_value=200, value=20
    )

    infer_classes = False
    csv_path = None
    filename_override = None
    label_override = None
    ids_without_ext = False
    normalize_labels = False

    if mode == "Images Only":
        infer_classes = st.sidebar.checkbox("Infer classes from subfolders", True)
    else:
        csv_path = st.sidebar.text_input("CSV path", "labels.csv")
        csv_full_path = csv_path
        if dataset_root:
            csv_full_path = os.path.join(dataset_root, csv_path)
        if os.path.exists(csv_full_path):
            try:
                columns = load_csv_columns(csv_full_path)
            except Exception as exc:  # pylint: disable=broad-except
                columns = []
                st.sidebar.warning("Could not read CSV columns: %s" % exc)
        else:
            columns = []
        options = [""] + columns
        filename_override = st.sidebar.selectbox(
            "Filename column override", options, index=0
        )
        label_override = st.sidebar.selectbox(
            "Label column override", options, index=0
        )
        ids_without_ext = st.sidebar.checkbox("IDs without extension resolution", True)
        normalize_labels = st.sidebar.checkbox("Normalize labels", False)

    run_clicked = st.sidebar.button("Run checks")
    clear_clicked = st.sidebar.button("Clear cache")

    if clear_clicked:
        st.cache_data.clear()
        st.sidebar.success("Cache cleared")

    if not run_clicked:
        st.info("Configure the dataset and click 'Run checks'.")
        return

    root = dataset_root if dataset_root else os.getcwd()
    allowed_exts = _parse_extensions(allowed_exts_text)

    config_dict: Dict[str, object] = {
        "mode": mode,
        "dataset_root": root,
        "images_dir": images_dir,
        "allowed_exts": allowed_exts,
        "hash_method": hash_method,
        "compute_stats": compute_stats,
        "infer_classes": infer_classes,
        "csv_path": csv_path,
        "filename_col": filename_override or None,
        "label_col": label_override or None,
        "ids_without_ext": ids_without_ext,
        "normalize_labels": normalize_labels,
    }

    progress = st.progress(0)
    status = st.empty()
    timings: Dict[str, float] = {}

    status.text("Scanning dataset...")
    scan_start = time.perf_counter()
    try:
        scan, config = cached_scan(config_dict)
    except Exception as exc:  # pylint: disable=broad-except
        st.error("Scan failed: %s" % exc)
        return
    timings["scan"] = time.perf_counter() - scan_start
    progress.progress(0.2)

    paths = sorted({entry.path for entry in scan.images})

    status.text("Verifying images...")
    verify_start = time.perf_counter()
    corrupted = cached_corrupted(paths)
    timings["verify"] = time.perf_counter() - verify_start
    progress.progress(0.45)

    status.text("Hashing images...")
    hash_start = time.perf_counter()
    duplicates = cached_duplicates(paths, hash_method)
    timings["hash"] = time.perf_counter() - hash_start
    progress.progress(0.7)

    stats = None
    hygiene: Optional[HygieneResult] = None
    if compute_stats:
        status.text("Computing image stats...")
        stats_start = time.perf_counter()
        details = cached_image_details(paths)
        stats = build_stats(details)
        hygiene = analyze_hygiene(details)
        timings["stats"] = time.perf_counter() - stats_start
        progress.progress(0.9)
    else:
        timings["stats"] = 0.0

    status.text("Finalizing...")
    counts = label_counts(scan.images)
    imbalance = check_imbalance(counts)
    ext_counts = {}
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    check_result = CheckResult(
        corrupted=corrupted,
        duplicates=duplicates,
        ext_counts=dict(sorted(ext_counts.items(), key=lambda item: item[0])),
        stats=stats,
        label_counts=counts,
        imbalance_warning=imbalance,
        hygiene=hygiene,
    )
    progress.progress(1.0)
    status.empty()

    warnings_by_group: Dict[str, List[str]] = {
        "CSV": [],
        "Labels": [],
        "Data Hygiene": [],
        "Extensions": [],
        "Duplicates": [],
    }
    warnings_by_group["CSV"].extend(scan.csv_warnings)

    if not (config.mode == "Images Only" and not config.infer_classes):
        if scan.missing_label_count:
            warnings_by_group["Labels"].append(
                "Missing labels: %d" % scan.missing_label_count
            )
    if config.mode == "Images + Labels (CSV)":
        if scan.missing_images:
            warnings_by_group["CSV"].append(
                "Missing images: %d" % len(scan.missing_images)
            )
        if scan.orphan_images:
            warnings_by_group["CSV"].append(
                "Orphan images: %d" % len(scan.orphan_images)
            )

    if check_result.imbalance_warning:
        warnings_by_group["Labels"].append(
            "Class imbalance detected (min/max < %.2f)." % IMBALANCE_THRESHOLD
        )

    if check_result.hygiene:
        if check_result.hygiene.small_res_count:
            warnings_by_group["Data Hygiene"].append(
                "Small resolution (<%dpx): %d"
                % (SMALL_RES_THRESHOLD, check_result.hygiene.small_res_count)
            )
        if check_result.hygiene.aspect_outlier_count:
            examples = ", ".join(check_result.hygiene.aspect_outlier_examples)
            warnings_by_group["Data Hygiene"].append(
                "Aspect ratio outliers (>%.1f): %d. Examples: %s"
                % (
                    ASPECT_OUTLIER_THRESHOLD,
                    check_result.hygiene.aspect_outlier_count,
                    examples,
                )
            )
        if check_result.hygiene.rgba_share > RGBA_SHARE_THRESHOLD:
            warnings_by_group["Data Hygiene"].append(
                "RGBA share %.0f%% exceeds %.0f%%."
                % (
                    check_result.hygiene.rgba_share * 100,
                    RGBA_SHARE_THRESHOLD * 100,
                )
            )
        if check_result.hygiene.mode_variance_warning:
            warnings_by_group["Data Hygiene"].append(
                "Mixed image modes show high variance across the dataset."
            )

    if config.mode == "Images + Labels (CSV)":
        ext_warnings = extension_mismatch_warnings(
            check_result.ext_counts, scan.csv_ext_counts, allowed_exts
        )
        warnings_by_group["Extensions"].extend(ext_warnings)

    st.subheader("Dashboard")
    if config.mode == "Images + Labels (CSV)":
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("Total images scanned", scan.total_images_scanned)
        col2.metric("Resolved images", scan.resolved_images)
        col3.metric("Corrupted", len(check_result.corrupted))
        col4.metric("Duplicate groups", len(check_result.duplicates))
        col5.metric("Missing labels", scan.missing_label_count)
        coverage_ratio = (
            scan.resolved_images / scan.csv_row_count if scan.csv_row_count else 0.0
        )
        orphan_rate = (
            len(scan.orphan_images) / scan.total_images_scanned
            if scan.total_images_scanned
            else 0.0
        )
        col6.metric(
            "Coverage",
            "%.1f%%" % (coverage_ratio * 100),
            "%d/%d" % (scan.resolved_images, scan.csv_row_count),
        )
        col7.metric(
            "Orphan rate",
            "%.1f%%" % (orphan_rate * 100),
            "%d/%d" % (len(scan.orphan_images), scan.total_images_scanned),
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total images", len(scan.images))
        col2.metric("Corrupted", len(check_result.corrupted))
        col3.metric("Duplicate groups", len(check_result.duplicates))
        if config.infer_classes:
            col4.metric("Missing labels", scan.missing_label_count)
        else:
            col4.metric("Unlabeled dataset", "yes")

    timing_rows = [
        {"step": "scan", "seconds": "%.3f" % timings["scan"]},
        {"step": "verify", "seconds": "%.3f" % timings["verify"]},
        {"step": "hash", "seconds": "%.3f" % timings["hash"]},
        {"step": "stats", "seconds": "%.3f" % timings["stats"]},
    ]
    st.dataframe(pd.DataFrame(timing_rows), use_container_width=True)

    if check_result.label_counts:
        st.subheader("Class Distribution")
        data = [
            {"label": label, "count": count}
            for label, count in check_result.label_counts.items()
        ]
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    st.subheader("Warnings")
    has_warning = any(warnings_by_group[key] for key in warnings_by_group)
    if not has_warning:
        st.success("No warnings")
    else:
        for category in sorted(warnings_by_group.keys()):
            items = warnings_by_group[category]
            if not items:
                continue
            st.markdown("**%s**" % category)
            st.markdown("\n".join(["- %s" % item for item in items]))

    st.subheader("Duplicates Viewer")
    if check_result.duplicates:
        options = [
            "Group %d (%d files, %s)" % (i + 1, len(group.paths), hash_method)
            for i, group in enumerate(check_result.duplicates)
        ]
        selection = st.selectbox("Duplicate groups", options)
        index = options.index(selection)
        group = check_result.duplicates[index]
        images = []
        captions = []
        for path in group.paths:
            thumb, caption = _image_thumbnail(path, None)
            images.append(thumb)
            captions.append(caption)
        st.image(images, caption=captions, width=200)
    else:
        st.write("No duplicates found.")

    st.subheader("Issues Viewer")
    if check_result.corrupted:
        st.write("Corrupted images")
        st.dataframe(
            pd.DataFrame(check_result.corrupted, columns=["path", "error"]),
            use_container_width=True,
        )
    if config.mode == "Images + Labels (CSV)" and scan.missing_images:
        st.write("Missing images from CSV")
        st.dataframe(
            pd.DataFrame(
                [
                    {"row_index": entry.row_index, "reference": entry.reference}
                    for entry in scan.missing_images
                ]
            ),
            use_container_width=True,
        )
    if config.mode == "Images + Labels (CSV)" and scan.orphan_images:
        st.write("Orphan images")
        st.dataframe(pd.DataFrame(scan.orphan_images, columns=["path"]))

    if show_preview and scan.images:
        st.subheader("Preview")
        sample_size = min(12, len(scan.images))
        rng = random.Random(1234)
        sample = rng.sample(scan.images, sample_size)
        images = []
        captions = []
        for entry in sample:
            thumb, caption = _image_thumbnail(entry.path, entry.label)
            images.append(thumb)
            captions.append(caption)
        st.image(images, caption=captions, width=200)

    if export_report:
        st.subheader("Export")
        report = build_report(
            config,
            scan,
            check_result,
            warnings_by_group,
            top_n=int(top_n),
        )
        report_path = Path("dataset_report.md")
        report_path.write_text(report, encoding="utf-8")
        st.download_button(
            "Download report",
            data=report,
            file_name="dataset_report.md",
            mime="text/markdown",
        )
        issues_csv = _build_issues_csv(scan, check_result, hash_method)
        st.download_button(
            "Download issues.csv",
            data=issues_csv,
            file_name="issues.csv",
            mime="text/csv",
        )
        st.caption("Report saved to dataset_report.md")


if __name__ == "__main__":
    main()
