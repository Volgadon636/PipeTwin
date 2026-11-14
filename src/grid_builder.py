"""
Continuous 10-meter segmentation grid builder.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from .config import SEGMENT_LENGTH_M, EPSILON_M


def build_segments_grid(passport_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Builds a continuous 10-meter segmentation grid for each pipeline:
      segment_id from 0 to floor((length_m - eps) / 10).
    The last segment represents the remaining distance (length <= 10m).
    
    Validates:
      1. No duplicates by (id, segment_id).
      2. No missing segment indices (0, 1, ..., N-1).
      3. Precise spatial continuity (start of next == end of previous).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []

    for _, r in passport_df.iterrows():
        pid = int(r['id'])
        length_m = float(r['length_km']) * 1000.0
        num_segments = int(np.floor((length_m - EPSILON_M) / SEGMENT_LENGTH_M)) + 1

        for seg_id in range(num_segments):
            start_m = round(seg_id * SEGMENT_LENGTH_M, 4)
            end_m = round(min((seg_id + 1) * SEGMENT_LENGTH_M, length_m), 4)
            len_m = round(end_m - start_m, 4)

            grid_rows.append({
                'id': pid,
                'segment_id': seg_id,
                'start_m': start_m,
                'end_m': end_m,
                'length_m': len_m
            })

    grid_df = pd.DataFrame(grid_rows)

    # Integrity assertions
    assert not grid_df.duplicated(subset=['id', 'segment_id']).any(), "Duplicate (id, segment_id) found in grid!"

    for pid, grp in grid_df.groupby('id'):
        seg_ids = grp['segment_id'].values
        expected_ids = np.arange(len(seg_ids))
        assert np.array_equal(seg_ids, expected_ids), f"Missing segment_id sequence in pipe {pid}!"
        assert grp['start_m'].iloc[0] == 0.0, f"Pipe {pid} grid does not start at 0.0m!"
        if len(grp) > 1:
            diffs = grp['start_m'].iloc[1:].values - grp['end_m'].iloc[:-1].values
            assert np.allclose(diffs, 0.0, atol=1e-5), f"Discontinuity gap found in pipe {pid} grid!"

    # Save to CSV and Parquet
    grid_csv = output_dir / 'segments_grid.csv'
    grid_parquet = output_dir / 'segments_grid.parquet'
    grid_df.to_csv(grid_csv, index=False, encoding='utf-8')
    grid_df.to_parquet(grid_parquet, index=False)

    return grid_df
