"""Loss-aware, deterministic parsing primitives.

The functions here do not decide what a source means.  They preserve source text,
report ambiguity, and return explicit parse states so adapters can quarantine a
value instead of guessing.  Existing public helpers in ``format_detector`` remain
available and delegate here where behaviour is equivalent.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ABSENT_MARKERS = {"", "#n/a", "n/a", "na", "nan", "null", "none", "нет данных", "нет"}
NOT_APPLICABLE_MARKERS = {"-", "—", "–", "не применимо", "not applicable", "n/a"}
_NUMBER = r"[+-]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?)(?:[eE][+-]?\d+)?"
_CENSORED_RE = re.compile(rf"^\s*([<>]=?)\s*({_NUMBER})\s*$")
_NUMBER_RE = re.compile(rf"^\s*({_NUMBER})\s*$")
_ID_RE = re.compile(r"^\s*([0-9]+)(?:\.(0+))?\s*$")


class ParseError(ValueError):
    """Base class for deterministic parse failures."""


class AmbiguousDialectError(ParseError):
    """A delimiter or encoding cannot be selected without guessing."""


@dataclass(frozen=True)
class Dialect:
    encoding: str
    delimiter: str
    score: float
    candidates: tuple[tuple[str, str, float], ...] = ()


@dataclass
class RawTable:
    path: Path
    encoding: str
    delimiter: str
    raw_headers: list[str]
    unique_headers: list[str]
    rows: list[list[str]]
    record_numbers: list[int]
    physical_lines: list[int]
    malformed: list[dict]
    duplicate_headers: list[str]
    empty_headers: list[int]
    dialect_candidates: tuple[tuple[str, str, float], ...] = ()

    @property
    def record_count(self) -> int:
        return len(self.rows)


def _decode(blob: bytes) -> tuple[str, str]:
    if blob.startswith(b"\xef\xbb\xbf"):
        try:
            return blob.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise ParseError(f"invalid UTF-8 BOM payload: {exc}") from exc
    try:
        return blob.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return blob.decode("cp1251"), "cp1251"
    except UnicodeDecodeError as exc:
        raise ParseError("file is neither valid UTF-8 nor Windows-1251") from exc


def decode_text(path: Path | str) -> tuple[str, str]:
    """Decode a supplied file as UTF-8/BOM or Windows-1251 without locale guesses."""
    blob = Path(path).read_bytes()
    return _decode(blob)


def _score_delimiter(text: str, delimiter: str) -> tuple[float, int, int]:
    """Score a candidate by header width and consistent record widths.

    Decimal commas in a semicolon file make a comma candidate wider and
    inconsistent, so the score is not based on a single character count.
    """
    sample = text[:262144]
    try:
        reader = csv.reader(io.StringIO(sample), delimiter=delimiter, quotechar='"',
                            doublequote=True, skipinitialspace=True)
        widths = [len(row) for row in reader if row and any(cell != "" for cell in row)]
    except csv.Error:
        return -1e9, 0, 0
    if not widths:
        return -1e9, 0, 0
    mode = max(set(widths), key=widths.count)
    consistent = sum(1 for width in widths if width == mode) / len(widths)
    # A width of one is never a usable CSV table.
    if mode < 2:
        return -1e9, mode, len(widths)
    return consistent * 100.0 + min(mode, 50) + min(len(widths), 100) / 1000.0, mode, len(widths)


def detect_dialect(path: Path | str, text: str | None = None,
                   encoding: str | None = None) -> Dialect:
    if text is None:
        text, encoding = decode_text(path)
    elif encoding is None:
        encoding, _ = decode_text(path)
    candidates = []
    for delimiter in (",", ";", "\t", "|"):
        score, mode, nrows = _score_delimiter(text, delimiter)
        if score > -1e8:
            candidates.append((delimiter, mode, score, nrows))
    if not candidates:
        raise AmbiguousDialectError(f"{path}: no usable CSV delimiter found")
    candidates.sort(key=lambda item: (-item[2], item[0]))
    best = candidates[0]
    tied = [c for c in candidates if abs(c[2] - best[2]) < 1e-9]
    if len(tied) > 1 and len({c[1] for c in tied}) > 1:
        detail = ", ".join(f"{c[0]!r} width={c[1]} score={c[2]:.4f}" for c in tied)
        raise AmbiguousDialectError(f"{path}: ambiguous delimiter; {detail}")
    return Dialect(encoding=encoding or "utf-8", delimiter=best[0], score=best[2],
                   candidates=tuple((d, str(m), s) for d, m, s, _ in candidates))


def peek_headers(path: Path | str) -> dict:
    """Read only the header row while still detecting encoding and delimiter."""
    path = Path(path)
    text, encoding = decode_text(path)
    dialect = detect_dialect(path, text, encoding)
    reader = csv.reader(io.StringIO(text), delimiter=dialect.delimiter,
                        quotechar='"', doublequote=True, skipinitialspace=True)
    try:
        headers = next(reader)
    except StopIteration as exc:
        raise ParseError(f"{path}: empty file") from exc
    return {"path": path, "encoding": encoding, "delimiter": dialect.delimiter,
            "headers": headers, "dialect": dialect}


def _dedupe_headers(headers: Sequence[str]) -> tuple[list[str], list[str], list[int]]:
    unique, duplicate, empty = [], [], []
    counts: dict[str, int] = {}
    for index, header in enumerate(headers):
        name = str(header).strip()
        if not name:
            empty.append(index)
            name = f"__empty_column_{index + 1}"
        count = counts.get(name, 0) + 1
        counts[name] = count
        if count == 1:
            unique.append(name)
        else:
            duplicate.append(name)
            unique.append(f"{name}__duplicate_{count}")
    return unique, duplicate, empty


def read_raw_table(path: Path | str, *, max_records: int | None = None) -> RawTable:
    path = Path(path)
    text, encoding = decode_text(path)
    dialect = detect_dialect(path, text, encoding)
    reader = csv.reader(io.StringIO(text), delimiter=dialect.delimiter,
                        quotechar='"', doublequote=True, skipinitialspace=True)
    try:
        raw_headers = next(reader)
    except StopIteration as exc:
        raise ParseError(f"{path}: empty file") from exc
    unique, duplicates, empty = _dedupe_headers(raw_headers)
    rows, record_numbers, physical_lines, malformed = [], [], [], []
    record_number = 0
    previous_line = reader.line_num
    try:
        for row in reader:
            record_number += 1
            start_line = previous_line + 1
            previous_line = reader.line_num
            if not row or (len(row) == 1 and row[0] == ""):
                # Completely blank physical records are not data records.
                malformed.append({"record_number": record_number, "physical_line": start_line,
                                  "error_code": "blank_record", "original_content": ""})
                continue
            if len(row) != len(raw_headers):
                malformed.append({"record_number": record_number, "physical_line": start_line,
                                  "error_code": "field_count_mismatch",
                                  "original_content": "|".join(row)[:4000],
                                  "expected_fields": len(raw_headers), "actual_fields": len(row)})
                continue
            rows.append(["" if cell is None else str(cell) for cell in row])
            record_numbers.append(record_number)
            physical_lines.append(start_line)
            if max_records is not None and len(rows) >= max_records:
                break
    except csv.Error as exc:
        malformed.append({"record_number": record_number + 1, "physical_line": reader.line_num,
                          "error_code": "csv_syntax_error", "original_content": str(exc)})
    return RawTable(path=path, encoding=encoding, delimiter=dialect.delimiter,
                    raw_headers=raw_headers, unique_headers=unique, rows=rows,
                    record_numbers=record_numbers, physical_lines=physical_lines,
                    malformed=malformed, duplicate_headers=duplicates,
                    empty_headers=empty, dialect_candidates=dialect.candidates)


def _strip_number_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = value.replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ")
    return value


def parse_number_text(value, *, decimal: str | None = None,
                      thousands: str | None = None) -> tuple[float | None, str, str]:
    """Return ``(number, state, reason)``.

    States: ``absent``, ``not_applicable``, ``censored``, ``ok``, ``parse_failed``.
    A censored value is deliberately not converted to zero.  A one-comma integer
    such as ``1,234`` is ambiguous unless a decimal convention is declared.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "absent", "null"
    text = _strip_number_text(str(value))
    if not text:
        return None, "absent", "empty"
    lowered = text.lower()
    if lowered in NOT_APPLICABLE_MARKERS:
        return None, "not_applicable", "marker"
    if lowered in ABSENT_MARKERS:
        return None, "absent", "marker"
    censored = _CENSORED_RE.match(text)
    if censored:
        return None, "censored", f"{censored.group(1)}{censored.group(2)}"
    if not _NUMBER_RE.match(text):
        return None, "parse_failed", "not_numeric"
    # Remove confirmed regular/NBSP thousands groups, but only in a way that is
    # unambiguous: spaces must be followed by exactly three digits.
    if " " in text:
        parts = text.split(" ")
        def _group_is_three(part: str) -> bool:
            head = re.split(r"[.,]", part, maxsplit=1)[0]
            return len(head) == 3 and head.isdigit()
        if (len(parts) > 1 and all(_group_is_three(p) for p in parts[1:])
                and parts[0].lstrip("+-").isdigit()):
            text = "".join(parts)
            thousands = "space" if thousands is None else thousands
        else:
            return None, "parse_failed", "ambiguous_space_thousands"
    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        last_dot, last_comma = text.rfind("."), text.rfind(",")
        if last_dot > last_comma:
            if thousands in (None, "comma") and all(len(p) == 3 for p in text.split(",")[1:]):
                text = text.replace(",", "")
            else:
                return None, "parse_failed", "ambiguous_separators"
        else:
            if thousands in (None, "dot") and all(len(p) == 3 for p in text.split(".")[1:]):
                text = text.replace(".", "").replace(",", ".")
            else:
                return None, "parse_failed", "ambiguous_separators"
    elif has_comma:
        count = text.count(",")
        groups = text.split(",")
        if decimal == "," or (decimal is None and count == 1 and len(groups[1]) != 3):
            text = text.replace(",", ".")
        elif thousands == "," or (count > 1 and all(len(p) == 3 for p in groups[1:])):
            text = text.replace(",", "")
        elif count == 1 and len(groups[1]) == 3 and decimal is None:
            return None, "parse_failed", "ambiguous_decimal_separator"
        else:
            return None, "parse_failed", "ambiguous_separators"
    elif has_dot:
        groups = text.split(".")
        if len(groups) > 2:
            if all(len(p) == 3 for p in groups[1:]):
                text = text.replace(".", "")
            else:
                return None, "parse_failed", "ambiguous_dot_separators"
    try:
        result = float(text)
    except ValueError:
        return None, "parse_failed", "not_numeric"
    if not np.isfinite(result):
        return None, "parse_failed", "nonfinite"
    return result, "ok", ""


def parse_numeric_series(series: pd.Series, *, decimal: str | None = None,
                         thousands: str | None = None) -> tuple[pd.Series, list[dict]]:
    parsed, issues = [], []
    for index, value in series.items():
        number, state, reason = parse_number_text(value, decimal=decimal, thousands=thousands)
        parsed.append(number)
        if state in ("parse_failed", "censored", "not_applicable"):
            issues.append({"index": index, "state": state, "reason": reason,
                           "original_value": "" if value is None else str(value)})
    return pd.Series(parsed, index=series.index, dtype="float64"), issues


def parse_identifier_text(value) -> tuple[int | None, str, str]:
    """Parse an identifier without floating-point round trips.

    ``1250013653.0`` is accepted only when the fractional part is exactly zero.
    Scientific notation, fractions, signs, non-finite values and zero are rejected.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "absent", "null"
    text = str(value).strip()
    if not text:
        return None, "absent", "empty"
    if text.lower() in ABSENT_MARKERS:
        return None, "absent", "marker"
    if text.lower() in NOT_APPLICABLE_MARKERS:
        return None, "not_applicable", "marker"
    match = _ID_RE.match(text)
    if not match:
        return None, "parse_failed", "not_exact_integer_identifier"
    try:
        number = int(match.group(1))
    except ValueError:
        return None, "parse_failed", "integer_overflow"
    if number <= 0:
        return None, "parse_failed", "nonpositive_identifier"
    return number, "ok", "integer_like_float" if match.group(2) else "integer"


def parse_identifier_series(series: pd.Series) -> tuple[pd.Series, list[dict], pd.Series]:
    parsed, issues, leading = [], [], []
    for index, value in series.items():
        number, state, reason = parse_identifier_text(value)
        parsed.append(number)
        raw = "" if value is None else str(value).strip()
        leading.append(bool(re.fullmatch(r"0+[0-9]+", raw)))
        if state != "ok":
            issues.append({"index": index, "state": state, "reason": reason,
                           "original_value": raw})
    return (pd.Series(parsed, index=series.index, dtype="Int64"), issues,
            pd.Series(leading, index=series.index, dtype="bool"))


_ISO_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:(?P<sep>[T ])(?P<H>\d{1,2}):(?P<M>\d{1,2})(?::(?P<S>\d{1,2})(?:\.(?P<frac>\d+))?)?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?)?$")
_RU_RE = re.compile(
    r"^(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})"
    r"(?:(?P<sep>[ T])(?P<H>\d{1,2}):(?P<M>\d{1,2})(?::(?P<S>\d{1,2})(?:\.(?P<frac>\d+))?)?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?)?$")
_SLASH_RE = re.compile(
    r"^(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})"
    r"(?:(?P<sep>[ T])(?P<H>\d{1,2}):(?P<M>\d{1,2})(?::(?P<S>\d{1,2})(?:\.(?P<frac>\d+))?)?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?)?$")


def _tzinfo(value: str | None):
    if not value:
        return None
    if value == "Z":
        return timezone.utc
    sign = 1 if value[0] == "+" else -1
    body = value[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(body[:2]), minutes=int(body[2:4])))


def parse_date_text(value, *, locale: str | None = None) -> dict:
    """Parse one date value and preserve date-only/timestamp/timezone facts."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"timestamp": None, "state": "absent", "reason": "null",
                "date_only": None, "tz_offset": None}
    text = str(value).strip()
    if not text:
        return {"timestamp": None, "state": "absent", "reason": "empty",
                "date_only": None, "tz_offset": None}
    if text.lower() in NOT_APPLICABLE_MARKERS | ABSENT_MARKERS:
        return {"timestamp": None, "state": "not_applicable" if text.lower() in NOT_APPLICABLE_MARKERS else "absent",
                "reason": "marker", "date_only": None, "tz_offset": None}
    match = _ISO_RE.match(text) or _RU_RE.match(text)
    ambiguous = False
    if match is None:
        slash = _SLASH_RE.match(text)
        if slash is None:
            return {"timestamp": None, "state": "parse_failed", "reason": "unrecognized_date",
                    "date_only": None, "tz_offset": None}
        a, b = int(slash.group("a")), int(slash.group("b"))
        if locale in ("ru", "ru_ru", "dayfirst"):
            day, month = a, b
        elif locale in ("en", "en_us", "monthfirst"):
            month, day = a, b
        elif a > 12 and b <= 12:
            day, month = a, b
        elif b > 12 and a <= 12:
            month, day = a, b
        else:
            ambiguous = True
            day = month = None
        groups = slash.groupdict()
        groups["d"], groups["m"] = day, month
    else:
        groups = match.groupdict()
    if ambiguous:
        return {"timestamp": None, "state": "ambiguous_date_locale",
                "reason": "slash_date_requires_declared_locale", "date_only": None,
                "tz_offset": None}
    try:
        year, month, day = int(groups["y"]), int(groups["m"]), int(groups["d"])
        hour = int(groups.get("H") or 0)
        minute = int(groups.get("M") or 0)
        second = int(groups.get("S") or 0)
        frac = groups.get("frac")
        micro = int((frac or "0").ljust(6, "0")[:6])
        tz = _tzinfo(groups.get("tz"))
        dt = datetime(year, month, day, hour, minute, second, micro, tzinfo=tz)
        if tz is not None:
            dt = dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return {"timestamp": None, "state": "parse_failed", "reason": "invalid_calendar_date",
                "date_only": None, "tz_offset": None}
    date_only = groups.get("sep") is None
    return {"timestamp": pd.Timestamp(dt.replace(tzinfo=None)),
            "state": "ok", "reason": "date_only" if date_only else "timestamp",
            "date_only": bool(date_only), "tz_offset": groups.get("tz")}


def parse_date_series(series: pd.Series, *, locale: str | None = None,
                      with_metadata: bool = False):
    parsed, metadata, issues = [], [], []
    for index, value in series.items():
        result = parse_date_text(value, locale=locale)
        parsed.append(result["timestamp"])
        metadata.append(result)
        if result["state"] != "ok":
            issues.append({"index": index, "state": result["state"], "reason": result["reason"],
                           "original_value": "" if value is None else str(value)})
    parsed_series = pd.Series(parsed, index=series.index, dtype="datetime64[ns]")
    if not with_metadata:
        return parsed_series
    date_only = pd.Series([m["date_only"] for m in metadata], index=series.index, dtype="boolean")
    tz_offsets = pd.Series([m["tz_offset"] for m in metadata], index=series.index, dtype="string")
    return parsed_series, date_only, tz_offsets, issues


def display_safe_text(value: str, max_length: int = 4000) -> str:
    """Make an untrusted source cell safe for spreadsheet text exports.

    The analytical source value is not changed; this is only for a separate
    spreadsheet-safe report.  Formula-leading cells are prefixed with an apostrophe.
    """
    text = "" if value is None else str(value)
    if text[:1] in ("=", "+", "-", "@"):
        text = "'" + text
    return text[:max_length]
