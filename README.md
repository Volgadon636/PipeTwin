# PipeTwin — 10-Meter Pipeline Segmentation & Preprocessing Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Clean Architecture](https://img.shields.io/badge/architecture-clean-brightgreen.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end data preprocessing and spatial segmentation framework designed for the **PipeTwin** oilfield pipeline wall-thinning prediction challenge (Nedra Hackathon).

The pipeline ingests raw field datasets, validates integrity, safely cleans anomalies without mutating raw files, constructs a continuous 10-meter spatial grid ($\text{segment\_id} = \lfloor \text{distance} / 10 \rfloor$), and maps in-line inspection (VTD/MFL) defects, manual ultrasonic thickness (UZT) measurements, elevation profiles, and hydrodynamic operating regimes.

---

## Quick Start (How to Run in 3 Steps)

### Step 1: Clone & Set Up Virtual Environment

#### On Windows (PowerShell):
```powershell
# Navigate into the project folder
cd pipetwins

# Create virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS (Bash):
```bash
# Navigate into the project folder
cd pipetwins

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
# Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r requirements.txt
```

Core libraries installed:
* `pandas` (>= 3.1) — DataFrame operations and tabular transformation
* `numpy` (>= 2.5) — Vectorized mathematical computations and array handling
* `pyarrow` (>= 26.0) — High-throughput Parquet read/write serialization
* `tqdm` (>= 4.66) — Progress tracking and batch utilities

---

### Step 3: Run the Pipeline

Execute the master orchestrator script with a single command:

```bash
python src/data_prep.py
```

The script automatically executes Steps 1 through 5, generates diagnostic logs, runs assertion tests, and populates the `data/cleaned/` and `processed/` folders.

---

## What Is Included Right Now

### 1. Ingested Datasets (`data/`)

Raw field data remains **100% immutable and intact** as the source of truth:
* **`passport.csv`** (16 rows, 286 columns) — Mechanical specifications, nominal wall thickness, steel grade, diameter, and commissioning dates for 16 pipelines.
* **`profile.csv`** (20,218 rows) — High-resolution GPS survey coordinates and elevation data ($Z$).
* **`vtd.csv`** (48,291 rows) — Magnetic flux leakage (MFL) in-line inspection defect logs (depth %, length, width, defect type).
* **`uzt.csv`** (67 rows) — Ultrasonic thickness measurements across reinforced field control points.
* **`regime.csv`** (495 rows) — Historical corrosion coupon (UKK) monitoring logs.
* **`tech_mode_*.csv`** (39,052 rows) — Daily hydrodynamic calculations (2012–2024): pressures, liquid/oil/gas rates, watercut %, mixture velocity, and pit corrosion rates.

---

### 2. Validated Clean Tables (`data/cleaned/`)

Cleaned and type-standardized datasets stored in both **`.csv`** (human-readable) and **`.parquet`** (fast binary format):
* **`passport.{csv,parquet}`** — 16 validated pipeline sections.
* **`profile.{csv,parquet}`** — 20,180 waypoints (38 exact duplicate survey points removed).
* **`vtd.{csv,parquet}`** — 48,043 defect records (4 negative odometer start offsets clipped to 0.0 m; 244 records beyond pipeline end traps excluded).
* **`uzt.{csv,parquet}`** — 67 valid wall thickness measurements across 3 field locations.
* **`regime.{csv,parquet}`** — 495 historical coupon logs.
* **`tech_mode_*.{csv,parquet}`** — 39,052 daily hydraulic computation records.

---

### 3. Target Segmented Datasets (`processed/`)

Deliverables produced by the 10-meter spatial segmentation engine:

| Output File | Rows | Columns | Purpose |
| :--- | :---: | :---: | :--- |
| **`segments_grid.{csv,parquet}`** | **4,668** | 5 | Master continuous 10m grid: `(id, segment_id, start_m, end_m, length_m)` |
| **`vtd_by_segment.{csv,parquet}`** | **1,483** | 10 | Aggregated MFL defect counts, max depth %, min residual wall thickness (mm), and defect dimensions |
| **`uzt_by_segment.{csv,parquet}`** | **3** | 7 | Ultrasonic thickness benchmarks per segment: min/mean thickness, angles, and measurement counts |
| **`profile_by_segment.{csv,parquet}`** | **4,668** | 8 | Elevation profile per segment: mean/min/max $Z$ and inter-segment slopes ($\Delta Z / \Delta L$ in m/m) |
| **`regime_mapped.{csv,parquet}`** | **4,668** | 16 | Operating conditions with linear hydrodynamic pressure interpolation ($p(x)$ between $p_{\text{start}}$ and $p_{\text{end}}$) |
| **`pipes_summary.csv`** | **16** | 6 | Pipeline-level summary report |
| **`step1_column_mappings.json`** | — | — | Full raw-to-canonical Latin snake_case column mapping dictionary |
| **`step2_cleaning_audit.json`** | — | — | Diagnostic log detailing all cleaning actions and excluded rows |

---

### 4. Network Summary Overview (`pipes_summary.csv`)

| Pipeline ID | Length (m) | 10m Segments | Segments with VTD | Segments with UZT | Segments with Regime |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1250013653** | 638.4 | 64 | **64** (100%) | 0 | 64 |
| **1250013655** | 545.2 | 55 | **55** (100%) | 0 | 55 |
| **1250022875** | 958.1 | 96 | **95** (99%) | 0 | 96 |
| **1250022876** | 524.1 | 53 | **51** (96%) | 0 | 53 |
| **1250013657** | 1,592.1 | 160 | **133** (83%) | 0 | 160 |
| **1250009490** | 44.7 | 5 | **5** (100%) | 0 | 5 |
| **1250022871** | 5,248.4 | 525 | **521** (99%) | **1** | 525 |
| **1250011604** | 493.4 | 50 | **50** (100%) | **1** | 50 |
| **1250028931** | 1,034.7 | 104 | 4 | 0 | 104 |
| **1250028932** | 4,450.8 | 446 | 9 | 0 | 446 |
| **1250028933** | 377.7 | 38 | 1 | 0 | 38 |
| **1250028934** | 7,252.0 | 726 | 31 | 0 | 726 |
| **1250028935** | 3,497.0 | 350 | 27 | 0 | 350 |
| **1250009583** | 14,745.1 | 1,475 | 0 | **1** | 1,475 |
| **1250009584** | 5,073.9 | 508 | 19 | 0 | 508 |
| **1250009587** | 129.1 | 13 | 3 | 0 | 13 |
| **TOTAL** | **46,510.6 m** | **4,668** | **1,068** | **3** | **4,668** |

---

## Clean Architecture Directory Layout

```text
pipetwins/
├── data/                                      # Raw immutable input datasets
│   ├── passport.csv                           # Pipeline technical passports
│   ├── profile.csv                            # Raw spatial GPS track
│   ├── vtd.csv                                # Raw MFL in-line inspection logs
│   ├── uzt.csv                                # Raw ultrasonic measurements
│   ├── regime.csv                             # Corrosion coupon monitoring
│   ├── tech_mode_*.csv                        # Daily hydraulic computation logs
│   └── cleaned/                               # Validated cleaned tables (.csv & .parquet)
│
├── processed/                                 # Target segmentation outputs & audit logs
│   ├── segments_grid.{csv,parquet}            # Master 10m grid (4,668 segments)
│   ├── vtd_by_segment.{csv,parquet}           # Aggregated VTD defect metrics
│   ├── uzt_by_segment.{csv,parquet}           # Aggregated UZT thickness benchmarks
│   ├── profile_by_segment.{csv,parquet}       # Elevation and slopes
│   ├── regime_mapped.{csv,parquet}            # Hydrodynamic regime features
│   ├── pipes_summary.csv                      # Pipeline summary report
│   ├── step1_column_mappings.json             # Column normalization registry
│   └── step2_cleaning_audit.json              # Cleaning action audit log
│
├── src/                                       # Modular Python source code
│   ├── __init__.py                            # Package declarations
│   ├── config.py                              # Central configuration & canonical mappings
│   ├── format_detector.py                     # Encoding sniffer, delimiter detector & date parser
│   ├── data_cleaner.py                        # Safe domain cleaning rules
│   ├── grid_builder.py                        # Continuous 10m spatial segmentation engine
│   ├── segment_mapper.py                      # Mapping VTD, UZT, Profile & Regime to segments
│   ├── validator.py                           # Integrity assertion tests & summary reporter
│   └── data_prep.py                           # Master CLI orchestrator running the full pipeline
│
├── .gitignore                                 # Git exclusions specification
├── requirements.txt                           # Pinned package dependencies
└── README.md                                  # Project documentation
```

---

## Key Domain Rules & Assumptions

1. **Pipeline Primary Key (`id`)**: The 10-digit simple section identifier (`simple_section_id`) serves as the unique universal foreign key connecting all datasets.
2. **Pipeline Lengths**: Pipeline length is defined as `length_km * 1000` from `passport.csv`. This value matches the maximum track distance in `profile.csv` within centimeters across all 16 pipelines.
3. **Negative VTD Distances**: 4 defect entries on pipe `1250011604` exhibited minor negative distances ($-0.095$ to $-0.252$ m) due to odometer zero-point calibration offset before the start weld. These were clipped to `0.0 m`, placing them into `segment_id = 0`.
4. **Tool Overshoot Beyond End Flange**: 244 VTD records exceeded nominal pipeline length (up to +7.9 m on pipe `1250011604` and +5.1 m on pipe `1250013655`) as the inspection pig traveled into receiving traps (KPСОД). These records were filtered from the line pipe body.
5. **Continuous 10-Meter Grid**: Each pipe is partitioned into continuous $10$ m segments:
   $$\text{segment\_id} \in \left[0, \, \left\lfloor \frac{L - \varepsilon}{10} \right\rfloor \right]$$
   The final segment covers the trailing fraction ($L - \text{start\_m} \le 10.0$ m).
6. **Hydrodynamic Pressure Modeling**: For each segment, operational parameters apply uniformly across the section, while pressure at the segment midpoint $x_{\text{center}}$ is modeled using linear hydrodynamic interpolation between inlet and outlet pressures:
   $$p(x) = p_{\text{start}} + \frac{x_{\text{center}}}{L} \cdot (p_{\text{end}} - p_{\text{start}})$$

---

## Verification & Integrity Assertions

The pipeline enforces strict runtime assertion checks:
* **Uniqueness**: Asserts that `(id, segment_id)` contains zero duplicates across all grid and profile tables.
* **Spatial Continuity**: Asserts that $\text{start\_m}[i+1] == \text{end\_m}[i]$ with no gaps or overlaps.
* **Boundary Confinement**: Asserts that all segment indices satisfy $0 \le \text{segment\_id} \le N_{\text{pipe}} - 1$.
* **Length Conservation**: Asserts that $\sum \text{length\_m}$ equals total pipeline length down to millimeters.
