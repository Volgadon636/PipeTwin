"""Complete half-open 10 m grid with the pipe endpoint assigned to the last segment."""
import numpy as np
import pandas as pd
from .config import SEGMENT_LENGTH_M


def build_segments_grid(passport):
    rows = []
    for row in passport.sort_values('id').itertuples():
        length = round(float(row.length_km)*1000, 6)
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f'Invalid length for {row.id}')
        count = int(np.ceil(length/SEGMENT_LENGTH_M))
        for sid in range(count):
            start = sid*SEGMENT_LENGTH_M
            end = min(start+SEGMENT_LENGTH_M, length)
            rows.append((int(row.id), sid, start, end, round(end-start, 6)))
    return pd.DataFrame(rows, columns=['id','segment_id','start_m','end_m','length_m'])


def assign_segments(df, coordinate, grid):
    lengths = grid.groupby('id').end_m.max()
    max_segments = grid.groupby('id').segment_id.max()
    values = df[coordinate]
    valid = df.id.isin(lengths.index) & values.notna() & values.ge(0) & values.le(df.id.map(lengths)+1e-9)
    if not valid.all():
        raise ValueError(f'Invalid positions in {coordinate}: {int((~valid).sum())}')
    result = df.copy()
    result['segment_id'] = np.minimum(np.floor(values/SEGMENT_LENGTH_M), df.id.map(max_segments)).astype('int64')
    return result
