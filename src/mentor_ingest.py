"""Deterministic mentor-call ingestion and follow-up workflow.

Accepted decisions are curated, versioned records; automatic extraction produces
low-confidence candidates only.  Source text is snapshotted by content hash, so a
corrected note creates a revision without erasing the previous interpretation.
No message is ever sent and no private recording is assumed to exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import BASE_DIR
from .parsing import decode_text


DEFAULTS = {
    "sources": "configs/mentor_sources.json",
    "records": "configs/mentor_requirements.json",
    "output": "docs/mentor",
}
CLASSIFICATION_RULES = (
    ("question_candidate", ("?", "уточнить", "не установлен", "неизвест", "не удалось", "требует подтверждения")),
    ("action_candidate", ("проверить", "сделать", "рассчитать", "запросить", "сравнить", "подготовить",
                          "исправить", "пересчитать", "собрать", "найти", "выяснить")),
    ("proposal_candidate", ("рекоменд", "предложил", "следует", "можно", "нужно", "не использовать",
                            "нет смысла", "попробуй", "лучше", "разреш", "допустимо")),
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_token(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def _decode(path: Path) -> str:
    return decode_text(path)[0]


def _source_text(source: Mapping) -> str | None:
    path = Path(source["path"])
    if not path.is_file():
        return None
    if path.suffix.lower() == ".pdf":
        return None
    try:
        return _decode(path)
    except Exception:
        return None


def _find_excerpt(text: str, excerpt: str) -> tuple[str | None, str | None, int | None]:
    """Return ``(locator, line, line_number)`` for a verbatim excerpt."""
    if not text or not excerpt:
        return None, None, None
    for number, line in enumerate(text.splitlines(), start=1):
        if excerpt in line:
            return f"L{number}", line, number
    return None, None, None


def _speaker_and_time(line: str):
    speaker = None
    timestamp = None
    match = re.search(r"(Говорящий\s+[A-ZА-Я])", line)
    if match:
        speaker = match.group(1)
    stamp = re.search(r"\b(\d{1,2}:\d{2})\b", line)
    if stamp:
        timestamp = stamp.group(1)
    return speaker, timestamp


def _classify(line: str) -> str:
    lowered = line.lower()
    for kind, keywords in CLASSIFICATION_RULES:
        if any(keyword in lowered for keyword in keywords):
            return kind
    return "statement_candidate"


def extract_candidates(source: Mapping, text: str, source_hash: str) -> list[dict]:
    if not text:
        return []
    candidates = []
    section = ""
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if line.startswith(">"):
            continue
        body = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if not body or len(body) < 8:
            continue
        kind = _classify(body)
        speaker, timestamp = _speaker_and_time(line)
        candidate_id = "CAND-" + _stable_token(source_hash, f"L{number}", body)
        candidates.append({
            "candidate_id": candidate_id,
            "source_id": source["source_id"],
            "source_hash": source_hash,
            "source_locator": f"L{number}",
            "section": section,
            "candidate_type": kind,
            "excerpt": body,
            "speaker": speaker or "",
            "timestamp": timestamp or "",
            "confidence": {"transcript": "medium_transcription_uncertainty",
                           "summary": "low_secondary_summary",
                           "strict_review": "low_team_interpretation",
                           "specification": "high_formal_text"}.get(source.get("doc_type", ""), "low_auto_extraction"),
            "accepted": False,
            "note": "Automatic candidate only; not a confirmed decision.",
        })
    return candidates


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _source_records(source_config: Mapping, output: Path) -> tuple[list[dict], dict[str, dict]]:
    inventory = []
    texts = {}
    revision_root = output / "revisions"
    for source in source_config["sources"]:
        path = (BASE_DIR / source["path"]).resolve()
        item = dict(source)
        item["path"] = str(path)
        item["exists"] = path.is_file()
        item["sha256"] = _sha256_file(path) if path.is_file() else None
        item["size_bytes"] = path.stat().st_size if path.is_file() else None
        item["revision_paths"] = []
        text = _source_text(source) if path.is_file() else None
        if text is not None:
            texts[source["source_id"]] = text
            revision = revision_root / source["source_id"] / f"{item['sha256'][:16]}{path.suffix.lower()}"
            revision.parent.mkdir(parents=True, exist_ok=True)
            if not revision.is_file():
                revision.write_bytes(path.read_bytes())
            item["revision_paths"].append(str(revision))
        else:
            item["revision_paths"].append(None)
        item["text_available"] = text is not None
        item["stated_meeting_date"] = source.get("stated_meeting_date")
        item["participants"] = source.get("participants")
        item["unknown_metadata"] = [key for key in ("stated_meeting_date", "participants")
                                    if source.get(key) is None]
        inventory.append(item)
    return inventory, texts


def _locate_record(record: Mapping, texts: Mapping[str, str]) -> dict:
    source_id = record.get("source_id")
    text = texts.get(source_id)
    locator, line, number = _find_excerpt(text or "", record.get("excerpt", ""))
    result = dict(record)
    result["source_locator"] = locator or "unresolved"
    result["source_excerpt_found"] = bool(locator)
    result["matched_line"] = line or ""
    result["matched_line_number"] = number
    result["excerpt_status"] = "found" if locator else ("source_missing" if text is None
                                                        else "source_excerpt_not_found")
    result["source_status"] = "current" if locator else "conflicting_source_revision"
    return result


def _record_frame(records: Iterable[Mapping], columns: list[str]) -> list[dict]:
    return [{column: row.get(column) for column in columns} for row in records]


def _diff_previous(previous_path: Path, current: pd.DataFrame, key: str, text_fields=("statement", "status")):
    if not previous_path.is_file():
        return {"baseline_created": True, "added": len(current), "changed": [], "removed": []}
    previous = pd.read_csv(previous_path, dtype="string", keep_default_na=False)
    return _diff_frames(previous, current, key, text_fields)


def _diff_frames(previous: pd.DataFrame, current: pd.DataFrame, key: str,
                 text_fields=("statement", "status")):
    if previous is None or key not in previous:
        return {"baseline_created": True, "added": len(current), "changed": [], "removed": []}
    old = {row[key]: row for _, row in previous.iterrows()}
    new = {row[key]: row for _, row in current.iterrows()}
    changed = []
    for identifier in sorted(set(old) & set(new)):
        differences = [field for field in text_fields if str(old[identifier].get(field, "")) != str(new[identifier].get(field, ""))]
        if differences:
            changed.append({"id": identifier, "fields": differences,
                            "old": {f: old[identifier].get(f) for f in differences},
                            "new": {f: new[identifier].get(f) for f in differences}})
    return {"baseline_created": False,
            "added": sorted(set(new) - set(old)),
            "changed": changed,
            "removed": sorted(set(old) - set(new))}


def _follow_up(questions: pd.DataFrame, actions: pd.DataFrame, inventory: list[dict]) -> str:
    order = {"correctness": 0, "evaluation": 1, "improvement": 2}
    q = questions.copy()
    if q.empty:
        q = pd.DataFrame(columns=["question_id", "priority", "source_id", "source_locator",
                                  "evidence_excerpt", "decision_unlocked", "needed_evidence"])
    else:
        q["priority_order"] = q.priority.map(order).fillna(9)
        q = q.sort_values(["priority_order", "question_id"], kind="stable")
    lines = ["# Follow-up draft (not sent)", "",
             "Generated locally from the supplied notes. No message was sent, no call was scheduled, "
             "and no private recording/Telegram history is assumed. Unknown dates and participants stay unknown.", "",
             "## Next-call agenda (correctness, then evaluation, then improvement)", ""]
    for index, row in enumerate(q.itertuples(index=False), start=1):
        lines.append(f"{index}. **{row.question_id}** ({row.priority}) — {row.question}")
    lines += ["", "## Draft questions tied to evidence", ""]
    for row in q.itertuples(index=False):
        lines += [f"### {row.question_id}", "", row.question,
                  f"- Evidence: `{row.source_id}` {row.source_locator}: \"{row.evidence_excerpt}\"",
                  f"- Unlocks: {row.decision_unlocked}",
                  f"- Needed evidence: {row.needed_evidence}", ""]
    lines += ["## Actions and dependencies", ""]
    for row in actions.itertuples(index=False):
        lines.append(f"- **{row.action_id}** [{row.status}]: {row.statement} "
                     f"(owner: {row.owner or 'not supplied'}; due: {row.due_date or 'not supplied'}; "
                     f"dependency: {row.dependency or 'none'})")
    lines += ["", "## Source coverage gaps", ""]
    for source in inventory:
        if source["unknown_metadata"] or not source["text_available"]:
            lines.append(f"- {source['source_id']}: unknown metadata {source['unknown_metadata']}; "
                         f"text_available={source['text_available']}.")
    lines += ["", "Do not send these drafts without explicit authorization.", ""]
    return "\n".join(lines)


def _implementation_impact(requirements: pd.DataFrame, questions: pd.DataFrame) -> str:
    lines = ["# Implementation impact report", "",
             "This report is generated from the requirement matrix; it does not claim that blocked science is resolved.", "",
             "## Implemented / verified components", ""]
    for row in requirements[requirements.status.isin(["implemented", "verified"])].itertuples(index=False):
        lines.append(f"- **{row.requirement_id}** [{row.status}] {row.statement} "
                     f"→ `{row.implementation_reference}`; tests: `{row.test_reference}`")
    lines += ["", "## Blocked or unresolved", ""]
    for row in questions[questions.status.isin(["open", "blocked"])].itertuples(index=False):
        lines.append(f"- **{row.question_id}**: {row.question} Evidence needed: {row.needed_evidence}")
    lines += ["", "## Rerun requirements when inputs change", "",
              "- Re-ingest mentor notes after a new note/revision: `python -m src.mentor_ingest`.",
              "- Rebuild preprocessing after any source/config/schema/rule change: `python -m src.data_prep --output-root <new>`. ",
              "- Re-run model/evaluation after feature or label changes; do not reuse an earlier cutoff's features.",
              "- Temperature scenario is explicit and isolated: `python scripts/calculate_segment_temperature.py --sensitivity`.",
              "- Submission validation must rerun against the derived grid or the authoritative template if one appears.", ""]
    return "\n".join(lines)


def _change_report(diff_req, diff_sources, candidates: pd.DataFrame) -> str:
    lines = ["# Mentor evidence change report", "",
             "## Requirement matrix changes", ""]
    if diff_req.get("baseline_created"):
        lines.append(f"- Baseline created with {diff_req['added']} requirements.")
    else:
        lines.append(f"- Added: {', '.join(diff_req['added']) or 'none'}")
        lines.append(f"- Removed: {', '.join(diff_req['removed']) or 'none'}")
        if diff_req["changed"]:
            for item in diff_req["changed"]:
                lines.append(f"- Changed **{item['id']}**: {', '.join(item['fields'])}")
        else:
            lines.append("- Changed: none")
    lines += ["", "## Source changes", ""]
    lines += [f"- {line}" for line in diff_sources] or ["- No previous source registry; baseline created."]
    lines += ["", "## Candidate extraction", "",
              f"- {len(candidates)} automatic candidates; accepted records remain curated and separate.",
              "- Automatic candidates are not confirmed decisions and never change the matrix by themselves.", ""]
    return "\n".join(lines)


def ingest(sources_config=DEFAULTS["sources"], records_config=DEFAULTS["records"],
           output=DEFAULTS["output"], *, previous_output=None) -> dict:
    sources_path = (BASE_DIR / sources_config).resolve() if not Path(sources_config).is_absolute() else Path(sources_config)
    records_path = (BASE_DIR / records_config).resolve() if not Path(records_config).is_absolute() else Path(records_config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_config = _load_json(sources_path)
    records_config_data = _load_json(records_path)
    previous_output = Path(previous_output) if previous_output is not None else output
    previous_requirements = (pd.read_csv(previous_output / "requirements.csv", dtype="string",
                                         keep_default_na=False)
                             if (previous_output / "requirements.csv").is_file() else None)
    previous_decisions = (pd.read_csv(previous_output / "decisions.csv", dtype="string",
                                      keep_default_na=False)
                          if (previous_output / "decisions.csv").is_file() else None)
    previous_inventory_data = (_load_json(previous_output / "source_inventory.json")
                               if (previous_output / "source_inventory.json").is_file()
                               else {"sources": []})
    inventory, texts = _source_records(source_config, output)
    # Keep revision history when a caller writes a new output directory.
    if previous_output != output and (previous_output / "revisions").is_dir():
        shutil.copytree(previous_output / "revisions", output / "revisions", dirs_exist_ok=True)
    inventory_by_id = {item["source_id"]: item for item in inventory}

    candidates = []
    for source in source_config["sources"]:
        text = texts.get(source["source_id"], "")
        candidates.extend(extract_candidates(source, text, inventory_by_id[source["source_id"]]["sha256"] or ""))
    candidates.sort(key=lambda row: (row["source_id"], row["source_locator"], row["candidate_id"]))
    _write_csv(output / "candidate_records.csv", candidates,
               ["candidate_id", "source_id", "source_hash", "source_locator", "section",
                "candidate_type", "excerpt", "speaker", "timestamp", "confidence", "accepted", "note"])

    located = {key: [_locate_record(row, texts) for row in records_config_data.get(key, [])]
               for key in ("decisions", "requirements", "open_questions", "actions")}
    id_fields = {"decisions": "decision_id", "requirements": "requirement_id",
                 "open_questions": "question_id", "actions": "action_id"}
    for group_name, group in located.items():
        for row in group:
            source = inventory_by_id.get(row["source_id"], {})
            row["source_hash"] = source.get("sha256")
            row["source_doc_type"] = source.get("doc_type")
            row["unknown_source_metadata"] = ",".join(source.get("unknown_metadata", []))
            row[id_fields[group_name]] = row.get("id")
    decision_rows = _record_frame(located["decisions"], [
        "decision_id", "statement", "status", "source_id", "source_locator", "evidence_type",
        "authority", "topic", "affected_component", "implementation_reference", "test_reference",
        "unresolved_question_ids", "supersedes_id", "source_hash", "source_doc_type",
        "unknown_source_metadata", "confidence", "excerpt", "source_excerpt_found", "excerpt_status", "source_status"])
    for row in decision_rows:
        value = row.get("unresolved_question_ids")
        row["unresolved_question_ids"] = ";".join(value) if isinstance(value, list) else value
        row["confidence"] = ("medium_transcription_uncertainty"
                             if str(row.get("source_doc_type", "")).startswith("meeting transcript")
                             else "low_secondary_summary_or_team_interpretation")
    requirement_rows = _record_frame(located["requirements"], [
        "requirement_id", "statement", "status", "source_id", "source_locator", "evidence_type",
        "affected_component", "implementation_reference", "test_reference", "unresolved_question_ids",
        "supersedes_id", "decision_id", "source_hash", "source_doc_type", "source_excerpt_found", "excerpt_status"])
    for row in requirement_rows:
        row["requirement_id"] = row.get("requirement_id") or row.get("id")
        value = row.get("unresolved_question_ids")
        row["unresolved_question_ids"] = ";".join(value) if isinstance(value, list) else value
    question_rows = _record_frame(located["open_questions"], [
        "question_id", "question", "status", "source_id", "source_locator", "evidence_type",
        "affected_component", "needed_evidence", "decision_unlocked", "priority", "source_hash",
        "source_doc_type", "source_excerpt_found", "excerpt_status", "source_status", "excerpt"])
    for row in question_rows:
        row["evidence_excerpt"] = row.get("excerpt", "")
    action_rows = _record_frame(located["actions"], [
        "action_id", "statement", "status", "source_id", "source_locator", "evidence_type",
        "owner", "due_date", "dependency", "implementation_reference", "source_hash",
        "source_doc_type", "source_excerpt_found", "excerpt_status", "source_status"])
    # Ensure canonical IDs exist; curated IDs must survive re-ingestion.
    def ensure(frame_rows, id_key, prefix):
        for index, row in enumerate(frame_rows, start=1):
            if not row.get(id_key):
                row[id_key] = f"{prefix}-{index:04d}"
        return frame_rows
    decision_rows = ensure(decision_rows, "decision_id", "DEC")
    requirement_rows = ensure(requirement_rows, "requirement_id", "REQ")
    question_rows = ensure(question_rows, "question_id", "Q")
    action_rows = ensure(action_rows, "action_id", "ACT")

    _write_csv(output / "decisions.csv", decision_rows, list(decision_rows[0].keys()) if decision_rows else [])
    _write_csv(output / "requirements.csv", requirement_rows,
               ["requirement_id", "statement", "status", "source_id", "source_locator",
                "evidence_type", "affected_component", "implementation_reference",
                "test_reference", "unresolved_question_ids", "supersedes_id", "decision_id",
                "source_hash", "source_doc_type", "source_excerpt_found", "excerpt_status", "source_status"])
    _write_csv(output / "open_questions.csv", question_rows,
               ["question_id", "question", "status", "source_id", "source_locator",
                "evidence_type", "affected_component", "needed_evidence", "decision_unlocked",
                "priority", "source_hash", "source_doc_type", "source_excerpt_found",
                "excerpt_status", "source_status", "evidence_excerpt"])
    _write_csv(output / "actions.csv", action_rows,
               ["action_id", "statement", "status", "source_id", "source_locator",
                "evidence_type", "owner", "due_date", "dependency", "implementation_reference",
                "source_hash", "source_doc_type", "source_excerpt_found", "excerpt_status", "source_status"])

    # Normalize requirement frame before diff/report generation.
    requirements_frame = pd.DataFrame(requirement_rows, columns=[
        "requirement_id", "statement", "status", "source_id", "source_locator",
        "evidence_type", "affected_component", "implementation_reference",
        "test_reference", "unresolved_question_ids", "supersedes_id", "decision_id",
        "source_hash", "source_doc_type", "source_excerpt_found", "excerpt_status",
        "source_status"])
    decisions_frame = pd.DataFrame(decision_rows, columns=[
        "decision_id", "statement", "status", "source_id", "source_locator", "evidence_type",
        "authority", "topic", "affected_component", "implementation_reference",
        "test_reference", "unresolved_question_ids", "supersedes_id", "source_hash",
        "source_doc_type", "unknown_source_metadata", "confidence", "excerpt",
        "source_excerpt_found", "excerpt_status", "source_status"])
    questions_frame = pd.DataFrame(question_rows, columns=[
        "question_id", "question", "status", "source_id", "source_locator", "evidence_type",
        "affected_component", "needed_evidence", "decision_unlocked", "priority",
        "source_hash", "source_doc_type", "source_excerpt_found", "excerpt_status",
        "source_status", "evidence_excerpt"])
    actions_frame = pd.DataFrame(action_rows, columns=[
        "action_id", "statement", "status", "source_id", "source_locator", "evidence_type",
        "owner", "due_date", "dependency", "implementation_reference", "source_hash",
        "source_doc_type", "source_excerpt_found", "excerpt_status", "source_status"])
    _write_json(output / "source_inventory.json", {
        "version": "mentor-source-inventory-v1",
        "sources": inventory,
        "unknown_metadata_policy": "Meeting dates and participant lists stay null when the source does not state them.",
    })
    diff_req = (_diff_frames(previous_requirements, requirements_frame,
                             "requirement_id", ("statement", "status", "source_locator"))
                if previous_requirements is not None
                else {"baseline_created": True, "added": len(requirement_rows),
                      "changed": [], "removed": []})
    diff_sources = []
    old = {item["source_id"]: item for item in previous_inventory_data.get("sources", [])}
    new = {item["source_id"]: item for item in inventory}
    for source_id in sorted(set(old) | set(new)):
        old_hash = old.get(source_id, {}).get("sha256")
        new_hash = new.get(source_id, {}).get("sha256")
        if source_id not in old:
            diff_sources.append(f"Added source {source_id}.")
        elif source_id not in new:
            diff_sources.append(f"Removed source {source_id}; previous interpretation retained.")
        elif old_hash != new_hash:
            diff_sources.append(f"Source {source_id} changed hash {old_hash} -> {new_hash}; "
                                "a revision was retained and accepted excerpts are flagged if not found.")
    # Revision history: previous accepted rows remain even when a source changes.
    history_requirements = pd.concat([previous_requirements, requirements_frame], ignore_index=True) if previous_requirements is not None else requirements_frame.copy()
    history_decisions = pd.concat([previous_decisions, decisions_frame], ignore_index=True) if previous_decisions is not None else decisions_frame.copy()
    if len(history_requirements):
        history_requirements = history_requirements.drop_duplicates(
            subset=["requirement_id", "source_hash", "statement", "status"], keep="first")
        history_requirements.to_csv(output / "requirements_history.csv", index=False,
                                    encoding="utf-8", lineterminator="\n")
    if len(history_decisions):
        history_decisions = history_decisions.drop_duplicates(
            subset=["decision_id", "source_hash", "statement", "status"], keep="first")
        history_decisions.to_csv(output / "decisions_history.csv", index=False,
                                 encoding="utf-8", lineterminator="\n")
    (output / "change_report.md").write_text(_change_report(diff_req, diff_sources, pd.DataFrame(candidates)),
                                             encoding="utf-8")
    (output / "follow_up.md").write_text(_follow_up(questions_frame, actions_frame, inventory), encoding="utf-8")
    (output / "implementation_impact.md").write_text(_implementation_impact(requirements_frame, questions_frame),
                                                      encoding="utf-8")
    summary = {
        "source_count": len(inventory),
        "sources_with_text": sum(1 for item in inventory if item["text_available"]),
        "candidate_records": len(candidates),
        "decisions": len(decision_rows),
        "requirements": len(requirement_rows),
        "open_questions": len(question_rows),
        "actions": len(action_rows),
        "source_excerpts_not_found": sum(1 for group in located.values() for row in group
                                         if not row.get("source_excerpt_found")),
        "unknown_dates_or_participants": sum(1 for item in inventory if item["unknown_metadata"]),
        "sent_messages": 0,
        "diff": diff_req,
    }
    _write_json(output / "run_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", default=DEFAULTS["sources"])
    parser.add_argument("--records-config", default=DEFAULTS["records"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--previous-output", default=None,
                        help="Directory containing the prior semantic outputs for diffing")
    args = parser.parse_args()
    summary = ingest(args.sources_config, args.records_config, args.output,
                     previous_output=args.previous_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
