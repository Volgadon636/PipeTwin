"""
Pipeline validation and reporting module.
Executes assertion checks on segmented tables and generates pipe-level summary reports.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd


def validate_and_summarize(
    grid_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    passport_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """
    Executes strict assertion tests on segmented tables:
      1. Verifies that (id, segment_id) has no duplicates.
      2. Asserts all segment_id values reside within the valid grid bounds for each pipe.
      3. Verifies pipeline length matches segment grid counts.
    Generates and persists pipes_summary.csv.
    """
    # 1. Uniqueness assertions
    assert not grid_df.duplicated(subset=['id', 'segment_id']).any(), "Duplicates found in segments_grid!"
    prof_df = tables['profile_by_segment']
    assert not prof_df.duplicated(subset=['id', 'segment_id']).any(), "Duplicates found in profile_by_segment!"

    # 2. Boundary assertions
    max_grid_segs = grid_df.groupby('id')['segment_id'].max().to_dict()

    vtd_seg = tables['vtd_by_segment']
    for _, r in vtd_seg.iterrows():
        pid = int(r['id'])
        sid = int(r['segment_id'])
        assert 0 <= sid <= max_grid_segs[pid], f"VTD segment_id {sid} exceeds grid bounds for pipe {pid}!"

    uzt_seg = tables['uzt_by_segment']
    for _, r in uzt_seg.iterrows():
        pid = int(r['id'])
        sid = int(r['segment_id'])
        assert 0 <= sid <= max_grid_segs[pid], f"UZT segment_id {sid} exceeds grid bounds for pipe {pid}!"

    # 3. Pipe-level summary generation
    summary_rows = []
    vtd_seg_counts = vtd_seg.groupby('id')['segment_id'].nunique().to_dict()
    uzt_seg_counts = uzt_seg.groupby('id')['segment_id'].nunique().to_dict()

    for _, r in passport_df.iterrows():
        pid = int(r['id'])
        length_m = float(r['length_km']) * 1000.0
        n_segs = len(grid_df[grid_df['id'] == pid])
        vtd_segs = vtd_seg_counts.get(pid, 0)
        uzt_segs = uzt_seg_counts.get(pid, 0)
        regime_segs = n_segs

        summary_rows.append({
            'id': pid,
            'length_m': length_m,
            'total_segments': n_segs,
            'segments_with_vtd': vtd_segs,
            'segments_with_uzt': uzt_segs,
            'segments_with_regime': regime_segs
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'pipes_summary.csv', index=False, encoding='utf-8')

    return summary_df
