# Remediation changelog

Nine confirmed defects fixed with narrowly scoped changes. DEFECT_REGISTER.csv carries source evidence, exact regression names and severity. regressions_before.txt contains seven failures and one error for PA-01..08 against the original parser. publication_before.txt separately demonstrates PA-09 after the first eight corrections. tests_after.txt records all 88 tests passing.

- PA-01: Decimal-based exact positive Int64 IDs; no float intermediate.
- PA-02/03: Strict logical CSV records, unique nonempty headers and exact field counts before pandas reads. Quoted delimiters/newlines supported.
- PA-04: Missing exposure dates mark invalid. Changes 38 flags and adds 38 issue records; current eligibility remains unchanged.
- PA-05: Hard mask nonpositive Kelvin before temperature persistence rules; raw readings retained.
- PA-06: Full timestamp precision in CSV. Date string presentation changes are benign and tested through CSV consumer.
- PA-07: Exact integer round-trip comparisons; float tolerances retained only for real-valued quantities.
- PA-08: Remove archive-wide diagnostic influence from cold features and use pre-cutoff VTD chronology for between-inspection features. Preserve nullable boolean public dtype. The existing ML cold wrapper already cleared this flag; the public preprocessing API now enforces the same isolation itself.
- PA-09: Roll back every touched destination on caught publication failure; mark incomplete publication and reject it in ML readers. Recovery limitations are explicit in cleaning_rules.md and UNRESOLVED_AMBIGUITIES.md.

Additional adversarial/property tests cover UTF-8/BOM/CP1251, Cyrillic quoted multiline fields, decimal comma, trailing empty header, invalid/ambiguous dates, adjacent floating-point segment boundaries, row reordering, duplicate-invariant minima, and interrupted-generation readers. Existing future-injection and per-pipe/calendar-gap persistence tests retained without weakening assertions.

No original sources or root published artifacts changed. Corrected builds are in corrected/ and repeat evidence in repeated/; both excluded from Git to avoid committing redundant bulk data. Their full schemas/hashes, detailed reconciliation and reproducible scripts are prepared for review and a future commit. User-owned docs/stage_prompts/Analysis.md was not edited or staged. Model methodology and temperature experiments unchanged.
