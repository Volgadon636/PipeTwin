"""Semantic source identification and versioned schema contracts.

Filenames are hints, never proof.  Classification uses header fingerprints and
reports filename/schema conflicts.  Legacy ``src.config.FILE_ALIASES`` is kept for
callers of the old resolver; the pipeline uses :func:`discover_sources`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import DATA_DIR, FILE_ALIASES, TECH_FILE
from .parsing import ParseError, peek_headers, read_raw_table

PASSPORT = "passport"
PROFILE = "profile"
VTD = "vtd"
UZT = "uzt"
HYDRAULIC = "hydraulic_regime"
MONITORING = "corrosion_monitoring"
TEMPLATE = "submission_template"

SOURCE_TYPES = (PASSPORT, PROFILE, VTD, UZT, HYDRAULIC, MONITORING, TEMPLATE)

LEGACY_KEY = {
    PASSPORT: "passport.csv",
    PROFILE: "profile.csv",
    VTD: "vtd.csv",
    UZT: "uzt.csv",
    HYDRAULIC: TECH_FILE,
    MONITORING: "regime.csv",
}
LEGACY_TO_SOURCE = {value: key for key, value in LEGACY_KEY.items()}
DISPLAY_NAME = {
    PASSPORT: "passport",
    PROFILE: "route profile",
    VTD: "VTD in-line inspection",
    UZT: "UZT ultrasonic measurements",
    HYDRAULIC: "technology/hydraulic regime (Techmod)",
    MONITORING: "corrosion/chemistry monitoring (RepCorr)",
    TEMPLATE: "submission template",
}

# Filename hints are intentionally broader than the legacy aliases.
ALIAS_HINTS = {
    PASSPORT: ("passport.csv", "pot_south.csv"),
    PROFILE: ("profile.csv", "altitudes_south.csv"),
    VTD: ("vtd.csv", "vtd_south_train.csv"),
    UZT: ("uzt.csv", "uzt_south.csv"),
    HYDRAULIC: (TECH_FILE, "regime.csv", "tech_mode.csv", "techmod.csv", "sample_result.csv"),
    MONITORING: ("regime.csv", "repcorr_south.csv", "repcorr_south(1).csv",
                 "corrosion_monitoring.csv"),
    TEMPLATE: ("sample_result.csv",),
}


@dataclass(frozen=True)
class SchemaContract:
    source_type: str
    schema_version: str
    adapter_version: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]
    date_fields: tuple[str, ...]
    fingerprint: tuple[str, ...]
    description: str


# Fingerprints are canonical/raw header names that are distinctive in this case.
CONTRACTS = {
    PASSPORT: SchemaContract(
        PASSPORT, "passport-v1", "adapter-passport-1",
        required_fields=("id", "length_km", "diameter_mm", "nominal_thickness_mm"),
        optional_fields=("steel_grade", "internal_coating", "commissioning_date",
                         "main_pipeline_id", "direction_id", "direction_name",
                         "pipeline_name", "length_accounting_m", "scrap_wall_thickness_mm"),
        numeric_fields=("id", "length_km", "diameter_mm", "nominal_thickness_mm",
                        "scrap_wall_thickness_mm", "length_accounting_m"),
        date_fields=("commissioning_date",),
        fingerprint=("Общие данные_D, мм", "Общие данные_Дата ввода",
                     "Данные по протяжённости в структуре сети OISPIPE Паспортизация_ID простого участка трубопровода"),
        description="One row per simple section/passport record."),
    PROFILE: SchemaContract(
        PROFILE, "profile-v1", "adapter-profile-1",
        required_fields=("id", "total_distance", "z"),
        optional_fields=("lon", "lat"),
        numeric_fields=("id", "total_distance", "z", "lat", "lon"),
        date_fields=(),
        fingerprint=("total_distance", "z"),
        description="Route elevation samples along total distance."),
    VTD: SchemaContract(
        VTD, "vtd-v1", "adapter-vtd-1",
        required_fields=("id", "date", "distance", "residual_wall_thickness"),
        optional_fields=("row_id", "distance_raw", "wall_thickness", "defect_depth_pct",
                         "defect_depth", "angle", "defect_length", "defect_width",
                         "defect_type", "section_number", "feature_number"),
        numeric_fields=("id", "row_id", "distance", "section_length", "wall_thickness",
                        "residual_wall_thickness", "defect_depth_pct", "defect_depth",
                        "angle", "defect_length", "defect_width"),
        date_fields=("date",),
        fingerprint=("Distance", "Residual Wall Thickness", "Defect Type"),
        description="VTD defect/measurement observations; Distance is the coordinate."),
    UZT: SchemaContract(
        UZT, "uzt-v1", "adapter-uzt-1",
        required_fields=("id", "date", "distance", "residual_wall_thickness"),
        optional_fields=("control_section_num", "start_distance", "control_point_id",
                         "point_num", "measurement_id", "angle"),
        numeric_fields=("id", "distance", "start_distance", "angle",
                        "residual_wall_thickness"),
        date_fields=("date",),
        fingerprint=("residual_wall_thickness", "angle", "start_distance"),
        description="UZT control-point thickness measurements."),
    HYDRAULIC: SchemaContract(
        HYDRAULIC, "hydraulic-v1", "adapter-hydraulic-1",
        required_fields=("id", "date", "q_liquid", "watercut_pct"),
        optional_fields=("p_start_fact", "p_end_fact", "p_start", "p_end", "q_oil",
                         "q_gas", "gas_content_rate", "v_liquid_true", "v_mix",
                         "flow_regime", "fill_coefficient", "density_liquid_work",
                         "density_oil", "viscosity_liquid_work", "pit_corrosion_intensity",
                         "length", "diameter_mm", "t_start", "t_end", "t_mix"),
        numeric_fields=("id", "p_start_fact", "p_end_fact", "p_start", "p_end", "q_oil",
                        "q_liquid", "watercut_pct", "gas_content_rate", "v_liquid_true",
                        "fill_coefficient", "q_gas", "density_liquid_work",
                        "viscosity_liquid_work", "density_oil", "pit_corrosion_intensity",
                        "v_mix", "t_mix", "length", "diameter_mm", "t_start", "t_end"),
        date_fields=("date",),
        fingerprint=("simple_section_id", "calc_date", "watercut", "q"),
        description="Daily technology/hydraulic regime; simple_section_id is not a 10 m segment."),
    MONITORING: SchemaContract(
        MONITORING, "monitoring-v1", "adapter-monitoring-1",
        required_fields=("id",),
        optional_fields=("year", "sample_installation_date", "sample_removal_date",
                         "ukk_number", "ukk_distance", "length", "diameter",
                         "wall_thickness", "flow_liquid", "watercut_pct", "exposure_days",
                         "direction_name", "fluid_density", "protective_effect_pct",
                         "ph", "co2_water", "total_mineralization"),
        numeric_fields=("id", "year", "ukk_distance", "diameter", "wall_thickness",
                        "flow_liquid", "watercut_pct", "exposure_days", "fluid_density",
                        "protective_effect_pct", "ph", "co2_water", "total_mineralization"),
        date_fields=("sample_installation_date", "sample_removal_date"),
        fingerprint=("Simple Section ID", "UKK Number", "Sample Installation Date"),
        description="Corrosion/chemistry monitoring; legacy filename regime.csv."),
    TEMPLATE: SchemaContract(
        TEMPLATE, "template-v1", "adapter-template-1",
        required_fields=("id", "segment_id", "thickness_pred"),
        optional_fields=(),
        numeric_fields=("id", "segment_id", "thickness_pred"),
        date_fields=(),
        fingerprint=("id", "segment_id", "thickness_pred"),
        description="Authoritative submission template; filename alone is insufficient."),
}


@dataclass
class SourceCandidate:
    path: Path
    source_type: str
    schema_version: str
    adapter_version: str
    sha256: str
    size_bytes: int
    encoding: str
    delimiter: str
    headers: tuple[str, ...]
    header_fingerprint: str
    record_count: int | None = None
    filename_hint: tuple[str, ...] = ()
    filename_conflict: bool = False
    role: str = "input"
    limitations: tuple[str, ...] = ()


@dataclass
class DiscoveryResult:
    candidates: dict[str, list[SourceCandidate]]
    selected: dict[str, SourceCandidate]
    by_legacy_key: dict[str, Path]
    warnings: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    template_validation: dict | None = None

    def inventory(self) -> list[dict]:
        rows = []
        for source_type, candidates in self.candidates.items():
            for candidate in candidates:
                rows.append({
                    "source_type": source_type,
                    "path": str(candidate.path),
                    "sha256": candidate.sha256,
                    "size_bytes": candidate.size_bytes,
                    "encoding": candidate.encoding,
                    "delimiter": candidate.delimiter,
                    "schema_version": candidate.schema_version,
                    "adapter_version": candidate.adapter_version,
                    "schema_fingerprint": candidate.header_fingerprint,
                    "record_count": candidate.record_count,
                    "filename_hints": list(candidate.filename_hint),
                    "filename_schema_conflict": candidate.filename_conflict,
                    "selected": self.selected.get(source_type) is candidate,
                    "role": candidate.role,
                    "limitations": list(candidate.limitations),
                })
        return rows


class SourceConflictError(ValueError):
    """Multiple semantically different files claim the same source role."""


class SourceAmbiguityError(ValueError):
    """A file matches more than one source schema and cannot be classified."""


def _norm_header(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _fingerprint(headers: Iterable[str]) -> str:
    payload = "\n".join(sorted(_norm_header(h) for h in headers))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matches(source_type: str, headers: list[str]) -> bool:
    contract = CONTRACTS[source_type]
    normalized = {_norm_header(h) for h in headers}
    if source_type == TEMPLATE:
        return [_norm_header(h) for h in headers] == ["id", "segment_id", "thickness_pred"]
    if source_type == PASSPORT:
        return any(_norm_header(h) in normalized for h in contract.fingerprint)
    if source_type == PROFILE:
        return {"id", "total_distance", "z"} <= normalized
    if source_type == VTD:
        return {"distance", "residual wall thickness", "defect type", "date"} <= normalized
    if source_type == UZT:
        return ({"residual_wall_thickness", "angle", "start_distance"} <= normalized
                or {"id", "date", "angle", "residual_wall_thickness"} <= normalized)
    if source_type == HYDRAULIC:
        return ({"simple_section_id", "calc_date", "watercut", "q"} <= normalized
                or {"simple_section_id", "calc_date", "watercut_pct", "q_liquid"} <= normalized)
    if source_type == MONITORING:
        return ({"simple section id", "ukk number", "sample installation date"} <= normalized
                or {"id", "sample_removal_date", "ph"} <= normalized)
    return False


def classify_headers(headers: list[str]) -> list[str]:
    matches = [source_type for source_type in SOURCE_TYPES if _matches(source_type, headers)]
    # A three-column template is also a valid generic table; only exact schema counts.
    if TEMPLATE in matches and len(headers) != 3:
        matches.remove(TEMPLATE)
    return matches


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate(path: Path, source_type: str, header_info: dict, *,
               role: str = "input") -> SourceCandidate:
    contract = CONTRACTS[source_type]
    hints = tuple(alias for alias in ALIAS_HINTS[source_type]
                  if path.name.lower() == alias.lower())
    # A filename hint conflict exists when another source type claims this name.
    other_hints = []
    for other, aliases in ALIAS_HINTS.items():
        if other != source_type and path.name.lower() in {a.lower() for a in aliases}:
            other_hints.append(other)
    return SourceCandidate(
        path=path, source_type=source_type, schema_version=contract.schema_version,
        adapter_version=contract.adapter_version, sha256=_sha256(path),
        size_bytes=path.stat().st_size, encoding=header_info["encoding"],
        delimiter=header_info["delimiter"], headers=tuple(header_info["headers"]),
        header_fingerprint=_fingerprint(header_info["headers"]),
        filename_hint=hints, filename_conflict=bool(other_hints and not hints) or bool(other_hints),
        role=role,
        limitations=tuple())


def _semantic_equivalent(left: Path, right: Path) -> bool:
    """Identical copies are deduplicated even if only line endings differ."""
    try:
        return left.read_bytes().replace(b"\r\n", b"\n") == right.read_bytes().replace(b"\r\n", b"\n")
    except OSError:
        return False


def discover_sources(data_dir: Path | str = DATA_DIR, *,
                     required_types: Iterable[str] = (),
                     optional_types: Iterable[str] = (),
                     source_overrides: Mapping[str, str] | None = None,
                     check_template: bool = True) -> DiscoveryResult:
    """Scan CSV files and assign semantic roles using schema fingerprints.

    ``source_overrides`` maps a source type to an explicit filename.  It is the
    only way to resolve two semantically different copies; otherwise the run fails.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    required = set(required_types)
    optional = set(optional_types)
    overrides = {k: Path(v) for k, v in (source_overrides or {}).items()}
    candidates: dict[str, list[SourceCandidate]] = {s: [] for s in SOURCE_TYPES}
    warnings: list[str] = []
    for path in sorted(p for p in data_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".csv"):
        try:
            header_info = peek_headers(path)
        except (ParseError, OSError) as exc:
            warnings.append(f"could not inspect {path.name}: {exc}")
            continue
        matches = classify_headers(header_info["headers"])
        if not matches:
            warnings.append(f"unclassified CSV: {path.name}")
            continue
        if len(matches) > 1:
            raise SourceAmbiguityError(
                f"{path.name} matches multiple source schemas: {matches}; "
                "provide explicit source_overrides or correct the file")
        source_type = matches[0]
        candidate = _candidate(path, source_type, header_info)
        if candidate.filename_conflict:
            warnings.append(
                f"filename/schema conflict: {path.name} looks like {source_type} "
                "by contents; filename also appears in another source's alias list")
        candidates[source_type].append(candidate)

    selected: dict[str, SourceCandidate] = {}
    by_legacy: dict[str, Path] = {}
    for source_type in SOURCE_TYPES:
        items = candidates[source_type]
        override = overrides.get(source_type)
        if override is not None:
            matching = [c for c in items if c.path.name == override.name]
            if not matching:
                raise FileNotFoundError(f"configured override for {source_type} not found: {override}")
            selected[source_type] = matching[0]
            continue
        if not items:
            if source_type in required:
                aliases = ", ".join(ALIAS_HINTS[source_type])
                raise FileNotFoundError(
                    f"Missing required source {source_type}; accepted filename hints: {aliases}")
            continue
        if len(items) == 1:
            selected[source_type] = items[0]
            continue
        # Deduplicate semantically identical copies with provenance.
        primary = items[0]
        identical = [c for c in items[1:] if _semantic_equivalent(primary.path, c.path)]
        if len(identical) == len(items) - 1:
            warnings.append(
                f"deduplicated {len(identical)} identical copy/copies of {source_type}: "
                + ", ".join(c.path.name for c in items))
            selected[source_type] = primary
        else:
            names = ", ".join(c.path.name for c in items)
            raise SourceConflictError(
                f"Conflicting copies for {source_type}: {names}. "
                "Select one explicitly via source_overrides; no file was preferred by name or date.")
    if TEMPLATE in selected:
        selected[TEMPLATE].role = "validation_only"
    for source_type, candidate in selected.items():
        legacy_key = LEGACY_KEY.get(source_type)
        if legacy_key is not None:
            by_legacy[legacy_key] = candidate.path
    missing_optional = [s for s in optional if s not in selected]
    template_validation = None
    if check_template:
        template_validation = validate_template_candidate(selected.get(TEMPLATE))
    return DiscoveryResult(candidates=candidates, selected=selected,
                           by_legacy_key=by_legacy, warnings=warnings,
                           missing_optional=missing_optional,
                           template_validation=template_validation)


def validate_template_candidate(candidate: SourceCandidate | None) -> dict:
    if candidate is None:
        return {"available": False, "status": "no_authoritative_template",
                "detail": "No CSV with exact id,segment_id,thickness_pred schema was found."}
    table = read_raw_table(candidate.path)
    headers = [str(h).strip() for h in table.raw_headers]
    exact = headers == ["id", "segment_id", "thickness_pred"]
    if not exact:
        return {"available": True, "status": "not_a_submission_template",
                "path": str(candidate.path), "sha256": candidate.sha256,
                "headers": headers,
                "detail": "Filename matched a hint, but column contents do not match the required schema."}
    return {"available": True, "status": "authoritative_schema_match",
            "path": str(candidate.path), "sha256": candidate.sha256,
            "headers": headers, "rows": table.record_count}


def source_contract(source_type: str) -> SchemaContract:
    return CONTRACTS[source_type]


def legacy_source_name(source_type: str) -> str:
    return LEGACY_KEY.get(source_type, source_type)


def normalize_source_name(name: str) -> str:
    """Map either a semantic type or a legacy key to a semantic type."""
    if name in SOURCE_TYPES:
        return name
    if name in LEGACY_TO_SOURCE:
        return LEGACY_TO_SOURCE[name]
    return name
