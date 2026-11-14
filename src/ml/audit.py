"""Machine-readable current schemas, missingness, time ranges and provenance."""
from pathlib import Path

import pandas as pd

from src.storage import sha256

JOIN_KEYS = {
    'target_features': ['id', 'segment_id', 'prediction_date', 'case'],
    'target_segments_grid': ['id', 'segment_id', 'prediction_date', 'case'],
    'training_targets': ['id', 'date', 'segment_id'],
    'vtd_by_segment': ['id', 'date', 'segment_id'],
    'profile_by_segment': ['id', 'segment_id'],
    'regime_mapped': ['id', 'segment_id', 'prediction_date'],
    'hydraulic_asof': ['id', 'date', 'prediction_date'],
    'hydraulic_quality': ['id', 'source_row', 'prediction_date', 'field', 'rule'],
    'monitoring_mapped': ['id', 'segment_id', 'prediction_date'],
    'quality_issues': ['file', 'source_row', 'field', 'rule'],
}


def source_audit(data):
    tables = {}
    for name, keys in JOIN_KEYS.items():
        frame = data.read('processed/' + name)
        dates = {}
        for col in frame:
            if pd.api.types.is_datetime64_any_dtype(frame[col]):
                dates[col] = {'min': None if frame[col].isna().all() else str(frame[col].min()),
                              'max': None if frame[col].isna().all() else str(frame[col].max())}
        tables['processed/' + name + '.csv'] = dict(
            evidence='observed_directly', rows=len(frame), keys=keys,
            columns=list(frame), dtypes={c: str(frame[c].dtype) for c in frame},
            missing_by_column={c: int(frame[c].isna().sum()) for c in frame}, dates=dates,
            representative_rows=frame.head(2).astype('string').where(frame.head(2).notna(), None).to_dict('records'))
    mismatches = [name for name, digest in data.manifest['code_sha256'].items()
                  if sha256(data.root/'src'/name) != digest]
    return dict(tables=tables, manifest_code_hash_mismatches=mismatches,
                published_validation=pd.read_json(data.root/'reports/validation.json', typ='series').to_dict(),
                warning='Published preprocessing checks are not tests rerun by this audit.',
                source_to_canonical_mappings={name: info['column_mapping']
                                             for name, info in data.manifest['inputs'].items()})
