# DataLens

🔗 Read the full write-up on dev.to:  
https://dev.to/ertugrulmutlu/datalens-a-read-only-image-dataset-sanity-checker-2do0

**DataLens** is a lightweight, read-only dataset sanity checker for image datasets. It helps you detect silent data quality issues *before* training by scanning images and (optionally) label metadata, then surfacing actionable insights through an interactive Streamlit dashboard and a shareable Markdown report.

> **Philosophy:** Inspect, don’t mutate. DataLens never modifies your dataset.

---

## Why DataLens?

Many model failures originate from data problems that go unnoticed:

* Corrupted or unreadable images
* Duplicate or near-duplicate samples
* Broken CSV-to-image mappings
* Severe class imbalance
* Inconsistent resolutions, aspect ratios, or image modes

DataLens is designed as a **pre-training audit tool**: fast, deterministic, and safe to run on any dataset.

---

## Features

### Core Checks (All Modes)

* Corrupted image detection (PIL verification)
* Duplicate detection:

  * `sha256` (exact duplicates)
  * `quick` (fast approximate hashing)
  * `phash` (near-duplicate / visually similar images)
* Image statistics (optional):

  * Resolution min / median / max
  * Image mode distribution (RGB / RGBA / L, etc.)
* Data hygiene warnings:

  * Very small images
  * Extreme aspect ratio outliers
  * High RGBA share
  * High mode variance

### Mode A — Images Only

* Scan an image folder recursively
* Optional class inference from subfolder structure
* Correct handling of **unlabeled datasets** (no false "missing label" warnings)

### Mode B — Images + Labels (CSV)

* Robust CSV reading (UTF-8 with Latin-1 fallback)
* Automatic filename and label column detection (with manual override)
* Support for IDs without file extensions
* Label normalization (`strip + lower`) toggle
* Coverage & orphan analysis:

  * **Coverage ratio:** resolved images / CSV rows
  * **Orphan rate:** unreferenced images / total images
* Extension mismatch detection between CSV and image files

### Reporting & Export

* Interactive Streamlit dashboard
* `dataset_report.md` (shareable, deterministic)
* `issues.csv` export:

  * missing images
  * orphan images
  * corrupted files
  * duplicate groups

---

## Project Structure

```
.
├── app.py                 # Streamlit UI & orchestration
├── sanity/
│   ├── scan.py            # Dataset scanning (Mode A / B)
│   ├── checks.py          # Corruption, duplicates, stats, hygiene
│   ├── report.py          # Markdown report generation
│   └── models.py          # Typed data containers
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended.

---

## Usage

```bash
streamlit run app.py
```

### Dataset Layout Examples

**Images Only**

```
dataset/
└── images/
    ├── cat/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── dog/
        └── img3.jpg
```

**Images + Labels (CSV)**

```
dataset/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
└── labels.csv
```

`labels.csv` must contain:

* one column referencing the image (filename, path, or ID)
* one column containing labels/classes

Common column names are auto-detected.

---

## Design Principles

* **Read-only:** no auto-fix, no mutation, no silent writes
* **Deterministic:** stable ordering, reproducible reports
* **Transparent:** all assumptions and thresholds are visible
* **Pre-training focused:** catch issues *before* they affect models

---

## When to Use DataLens

* Before starting model training
* When receiving datasets from external sources
* When debugging unstable training or strange metrics
* As a lightweight QA step in ML pipelines
---

## Links

- **dev.to Article:** https://dev.to/ertugrulmutlu/datalens-a-read-only-image-dataset-sanity-checker-2do0  
- **LinkedIn:** https://www.linkedin.com/in/ertugrul-mutlu/  
- **Personal Website:** https://ertugrulmutlu.github.io
---

## License

MIT License

---

## Author

Built by Ertuğrul Mutlu.

If you find this useful, feel free to open an issue or start a discussion.
