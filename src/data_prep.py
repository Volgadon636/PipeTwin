"""
PipeTwin Data Preparation Pipeline
Main orchestrator module executing Steps 1 through 5:
- Step 1: Format detection, ingestion, and column standardization.
- Step 2: Safe cleaning and validation without mutating raw files.
- Step 3: Continuous 10-meter segmentation grid generation.
- Step 4: Spatial and temporal mapping of VTD, UZT, Profile, and Regime.
- Step 5: Assert validations, final artifacts persistence, and reporting.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import DATA_DIR, CLEANED_DIR, PROCESSED_DIR
    from src.format_detector import read_and_standardize_all
    from src.data_cleaner import clean_datasets
    from src.grid_builder import build_segments_grid
    from src.segment_mapper import map_all_to_segments
    from src.validator import validate_and_summarize
else:
    from .config import DATA_DIR, CLEANED_DIR, PROCESSED_DIR
    from .format_detector import read_and_standardize_all
    from .data_cleaner import clean_datasets
    from .grid_builder import build_segments_grid
    from .segment_mapper import map_all_to_segments
    from .validator import validate_and_summarize

# Ensure UTF-8 output across all operating systems
sys.stdout.reconfigure(encoding='utf-8')


def run_pipeline(
    data_dir: Path = DATA_DIR,
    cleaned_dir: Path = CLEANED_DIR,
    processed_dir: Path = PROCESSED_DIR
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Executes the complete PipeTwin data preparation workflow.
    """
    print("=" * 75)
    print("STEP 1: READING, FORMAT DETECTION & COLUMN STANDARDIZATION")
    print("=" * 75)
    raw_dfs, summary = read_and_standardize_all(data_dir)

    # Save column mappings audit
    mappings_file = processed_dir / "step1_column_mappings.json"
    with open(mappings_file, "w", encoding="utf-8") as f:
        json.dump({fn: s['column_mapping'] for fn, s in summary.items()}, f, indent=2, ensure_ascii=False)
    print(f"[OK] Standardized column mappings saved to: {mappings_file}")

    print("\n" + "=" * 75)
    print("STEP 2: DOMAIN CLEANING & PERSISTENCE TO data/cleaned/")
    print("=" * 75)
    cleaned_dfs, audit = clean_datasets(raw_dfs, cleaned_dir)

    audit_file = processed_dir / "step2_cleaning_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    for component, details in audit.items():
        print(f"[{component.upper()}] {details['status']}")
    print(f"[OK] Cleaned datasets saved to: {cleaned_dir} (.csv & .parquet)")
    print(f"[OK] Cleaning audit log saved to: {audit_file}")

    print("\n" + "=" * 75)
    print("STEP 3: CONTINUOUS 10-METER SEGMENTATION GRID BUILDER")
    print("=" * 75)
    grid_df = build_segments_grid(cleaned_dfs['passport.csv'], processed_dir)
    print(f"[OK] Generated {len(grid_df)} segments across {grid_df['id'].nunique()} pipelines")
    print(f"[OK] Grid saved to: {processed_dir / 'segments_grid.csv'}")

    print("\n" + "=" * 75)
    print("STEP 4: MAPPING VTD, UZT, PROFILE & REGIME TO SEGMENTS")
    print("=" * 75)
    tables = map_all_to_segments(cleaned_dfs, grid_df, processed_dir)
    for table_name, table_df in tables.items():
        print(f"  - {table_name}: {len(table_df):5d} rows x {len(table_df.columns):2d} columns")

    print("\n" + "=" * 75)
    print("STEP 5: ASSERTION VALIDATIONS & SUMMARY REPORTING")
    print("=" * 75)
    summary_df = validate_and_summarize(grid_df, tables, cleaned_dfs['passport.csv'], processed_dir)
    print("[OK] All integrity assertion tests passed successfully!")
    print("\nPIPELINE-LEVEL SUMMARY REPORT:")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Summary report saved to: {processed_dir / 'pipes_summary.csv'}")

    return cleaned_dfs, grid_df, tables, summary_df


if __name__ == '__main__':
    run_pipeline()
