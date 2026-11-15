"""Geometry and signed-correlation checks for the S2 alternatives."""
import unittest

import pandas as pd

from src.ml.s2_terrain import BASINS, RANKS, VARIANTS, select, terrain_features
from src.ml.targeted_s2 import SPATIAL_FEATURES


class TerrainTests(unittest.TestCase):
    def test_basin_depth_and_ranks_use_the_complete_geometry(self):
        height = [5, 4, 1, 3, 6]
        grid = pd.DataFrame({
            'id': 7, 'segment_id': range(5), 'start_m': range(0, 50, 10),
            'end_m': range(10, 60, 10), 'z_center': height,
            **{name: height for name in SPATIAL_FEATURES},
        })
        base = terrain_features(grid)
        modified = terrain_features(grid.iloc[::-1].assign(next_mm=100))
        pd.testing.assert_frame_equal(base[BASINS + RANKS], modified[BASINS + RANKS])
        self.assertEqual(base.basin_depth_30m.tolist(), [0, 1, 4, 2, 0])
        self.assertEqual(base.z_rank.tolist(), [0.7, 0.5, 0.1, 0.3, 0.9])
        basin = base.loc[base.segment_id.eq(2), 'basin_depth_30m'].item()
        self.assertEqual(basin, 4)

    def test_selection_rejects_negative_r_and_dominant_section_regression(self):
        def record(scheme, candidate, r):
            return {'scheme': scheme, 'candidate': candidate, 'pearson_r': r,
                    'pearson_squared': r * r}

        values = [(0.1, 0.1), (-0.8, -0.8), (0.4, 0.05), (0.2, 0.2)]
        rows = [record(scheme, name, r) for name, pair in zip(VARIANTS, values)
                for scheme, r in zip(['section_LOO', 'section_7'], pair)]
        self.assertEqual(select(rows, 7), 'basin_ranks')
