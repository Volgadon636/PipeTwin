"""Read published artifacts, preserving their manifest schema and provenance."""
import json
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TECH_FILE
from src.storage import sha256


class RepositoryData:
    def __init__(self, root=BASE_DIR):
        self.root = Path(root)
        # Explicit UTF-8: the default Windows locale (cp1251) cannot decode the
        # Russian source names stored in the manifest.
        self.config = json.loads((self.root / 'configs/preprocessing.json').read_text(encoding='utf-8'))
        self.manifest = json.loads((self.root / 'reports/run_manifest.json').read_text(encoding='utf-8'))
        if self.config != self.manifest['config']:
            raise ValueError('Preprocessing config changed: rebuild artifacts in a separate output root')
        self.hashes = {}
        self._cache = {}

    def read(self, stem):
        """CSV is a supported published format, not a new raw-data parser.

        Reading explicit types also permits ML-only execution without PyArrow.
        Every CSV must match the preprocessing manifest before it can be used.
        """
        if stem not in self._cache:
            entry = self.manifest['outputs'][stem]
            path = self.root / (stem + '.csv')
            digest = sha256(path)
            if digest != entry['csv_sha256']:
                raise ValueError(f'Artifact checksum mismatch: {path}')
            types = entry['dtypes']
            dates = [c for c, t in types.items() if t.startswith('datetime')]
            dtypes = {c: ('string' if t in ('str', 'object') else t)
                      for c, t in types.items() if c not in dates}
            frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[''])
            for column in dates:
                frame[column] = pd.to_datetime(frame[column], format='ISO8601')
            if len(frame) != entry['rows'] or set(frame) != set(types):
                raise ValueError(f'Artifact schema mismatch: {path}')
            self.hashes[stem + '.csv'] = digest
            self._cache[stem] = frame
        return self._cache[stem].copy()

    @property
    def cleaned(self):
        # build_features_asof does not require raw VTD/UZT: it consumes their
        # existing mapped histories. Do not load target-pipe raw diagnostics.
        return {name: self.read('data/cleaned/' + Path(name).stem)
                for name in ('passport.csv', 'regime.csv', TECH_FILE)}

    def parent_groups(self):
        passport = self.read('data/cleaned/passport')
        groups = passport.set_index('id').main_pipeline_id
        if groups.isna().any():
            raise ValueError('Missing main_pipeline_id; independent groups cannot be inferred')
        return groups
