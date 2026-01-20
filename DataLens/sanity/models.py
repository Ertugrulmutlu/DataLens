from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ScanConfig:
    mode: str
    dataset_root: str
    images_dir: str
    allowed_exts: List[str]
    hash_method: str
    compute_stats: bool
    infer_classes: bool
    csv_path: Optional[str]
    filename_col: Optional[str]
    label_col: Optional[str]
    ids_without_ext: bool
    normalize_labels: bool


@dataclass(frozen=True)
class ImageEntry:
    path: str
    rel_path: str
    label: Optional[str]


@dataclass(frozen=True)
class MissingImageEntry:
    row_index: int
    reference: str


@dataclass(frozen=True)
class ImageStat:
    path: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class ScanResult:
    images: List[ImageEntry]
    missing_images: List[MissingImageEntry]
    orphan_images: List[str]
    csv_warnings: List[str]
    filename_col: Optional[str]
    label_col: Optional[str]
    missing_label_count: int
    csv_row_count: int
    resolved_images: int
    total_images_scanned: int
    csv_ext_counts: Dict[str, int]
    csv_no_ext_count: int


@dataclass(frozen=True)
class DuplicateGroup:
    hash_value: str
    paths: List[str]


@dataclass(frozen=True)
class StatsResult:
    width_min: int
    width_median: int
    width_max: int
    height_min: int
    height_median: int
    height_max: int
    mode_counts: Dict[str, int]


@dataclass(frozen=True)
class HygieneResult:
    small_res_count: int
    small_res_examples: List[str]
    aspect_outlier_count: int
    aspect_outlier_examples: List[str]
    rgba_share: float
    mode_variance_warning: bool


@dataclass(frozen=True)
class CheckResult:
    corrupted: List[Tuple[str, str]]
    duplicates: List[DuplicateGroup]
    ext_counts: Dict[str, int]
    stats: Optional[StatsResult]
    label_counts: Optional[Dict[str, int]]
    imbalance_warning: bool
    hygiene: Optional[HygieneResult]
