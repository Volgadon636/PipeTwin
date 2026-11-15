# Follow-up draft (not sent)

Generated locally from the supplied notes. No message was sent, no call was scheduled, and no private recording/Telegram history is assumed. Unknown dates and participants stay unknown.

## Next-call agenda (correctness, then evaluation, then improvement)

1. **Q-0001** (correctness) — Is an authoritative submission grid/template available, and what is the endpoint rule for the final incomplete segment?
2. **Q-0002** (correctness) — What are the exact units and physical basis of pressure, viscosity, q/q_oil/q_gas, d_mm, t_mix, CO2 and inhibitor measurements?
3. **Q-0003** (correctness) — Are CO2, КВЧ/mechanical impurities and inhibitor fields concentrations, fractions, doses or scores, and at what timestamp/reference condition?
4. **Q-0004** (correctness) — Can you provide the historical RepCorr ID/direction crosswalk with effective dates? The current join rejects 412 unknown IDs and 83 direction conflicts and has zero usable target records.
5. **Q-0005** (correctness) — Why are UZT thicknesses above passport nominal, and are they measured on a different wall reference or method?
6. **Q-0007** (correctness) — Is there repair/replacement history, and how does it reset segment history?
7. **Q-0008** (correctness) — Are same-day 2023-12-01 diagnostics allowed? The current conservative cutoff excludes the entire day.
8. **Q-0006** (evaluation) — What critical/rejection thickness and ranking policy applies at the forecast date?
9. **Q-0009** (evaluation) — What is the train/test role of 1250009583, and which cold-start calibration evidence is permitted?
10. **Q-0011** (evaluation) — Are campaign detection and apparent thickening complete, or are there missing/extra observations and repairs?
11. **Q-0013** (evaluation) — Should the starting state use the minimum of the 2018 and 2022 inspections or the latest available inspection? The team selected minimum; prove it on identical splits.
12. **Q-0010** (improvement) — What measured ground temperature/thermal properties and flow basis should be used, or is a temperature scenario desired at all?
13. **Q-0012** (improvement) — How should defect coordinates/identities be matched across VTD campaigns, if at all?

## Draft questions tied to evidence

### Q-0001

Is an authoritative submission grid/template available, and what is the endpoint rule for the final incomplete segment?
- Evidence: `SRC-M1-SUMMARY` L102: "Как обрабатывать последнюю неполную секцию трубы и измерения точно на конечной границе"
- Unlocks: DEC-0005 / REQ-0001
- Needed evidence: Organizer-provided sample_result/template or written endpoint rule

### Q-0002

What are the exact units and physical basis of pressure, viscosity, q/q_oil/q_gas, d_mm, t_mix, CO2 and inhibitor measurements?
- Evidence: `SRC-M1-SUMMARY` L104: "Какое точное имя поля остаточной толщины в выгрузке и как трактовать неоднозначные процентные значения?"
- Unlocks: DEC-0013; Reynolds/thermal eligibility
- Needed evidence: Field dictionary or instrument/export documentation with units and reference conditions

### Q-0003

Are CO2, КВЧ/mechanical impurities and inhibitor fields concentrations, fractions, doses or scores, and at what timestamp/reference condition?
- Evidence: `SRC-M1-TRANSCRIPT` L49: "В Ripcore упоминались **CO₂, минерализация и ПАЖ**"
- Unlocks: chemistry feature use
- Needed evidence: Measurement method, units, sampling point and availability time

### Q-0004

Can you provide the historical RepCorr ID/direction crosswalk with effective dates? The current join rejects 412 unknown IDs and 83 direction conflicts and has zero usable target records.
- Evidence: `SRC-CLEANING-RULES` L20: "Мониторинг с конфликтом ID/направления"
- Unlocks: monitoring integration
- Needed evidence: Authoritative ID/direction mapping with validity intervals; approved spatial/temporal transfer policy

### Q-0005

Why are UZT thicknesses above passport nominal, and are they measured on a different wall reference or method?
- Evidence: `SRC-M1-TRANSCRIPT` L390: "В УЗД тоже есть какие-то изменения, где написано просто, что там паспортная толщина стенок трубы от 5 до 8, а там 11 написано."
- Unlocks: UZT feature/label policy
- Needed evidence: UZT method, calibration block, measurement location and wall reference

### Q-0007

Is there repair/replacement history, and how does it reset segment history?
- Evidence: `SRC-M1-SUMMARY` L63: "Ремонты могут менять историю деградации участка."
- Unlocks: history reset/segmentation
- Needed evidence: Dated repair/replacement records with intervals

### Q-0008

Are same-day 2023-12-01 diagnostics allowed? The current conservative cutoff excludes the entire day.
- Evidence: `SRC-CLEANING-RULES` L7: "Время ВТД внутри дня, поэтому день T исключён целиком"
- Unlocks: cutoff change only after review
- Needed evidence: Observation time and organizer rule for same-day measurements

### Q-0006

What critical/rejection thickness and ranking policy applies at the forecast date?
- Evidence: `SRC-M1-SUMMARY` L103: "Какой критерий критичности использовать для отбора участков на УЗТ"
- Unlocks: priority reason/threshold
- Needed evidence: Approved critical thickness formula or explicit statement that none is provided

### Q-0009

What is the train/test role of 1250009583, and which cold-start calibration evidence is permitted?
- Evidence: `SRC-M1-SUMMARY` L29: "В разговоре неоднозначно обсуждалась принадлежность объекта **1250009583** к train/test"
- Unlocks: cold-start training policy
- Needed evidence: Organizer split file or explicit statement

### Q-0011

Are campaign detection and apparent thickening complete, or are there missing/extra observations and repairs?
- Evidence: `SRC-M2-SUMMARY` L41: "Ментор отметил, что стенка сама по себе не срастается"
- Unlocks: thickening interpretation
- Needed evidence: Campaign coverage, tool calibration and repair records

### Q-0013

Should the starting state use the minimum of the 2018 and 2022 inspections or the latest available inspection? The team selected minimum; prove it on identical splits.
- Evidence: `SRC-M2-STRICT-REVIEW` L21: "Compare it against the latest-available-inspection baseline on appropriate historical tests."
- Unlocks: DEC-0012
- Needed evidence: Paired forward-time comparison with pre-cutoff features

### Q-0010

What measured ground temperature/thermal properties and flow basis should be used, or is a temperature scenario desired at all?
- Evidence: `SRC-M2-STRICT-REVIEW` L54: "Numerical values and units for roughness, heat capacity and the heat-transfer coefficient, and whether 5 °C is the intended ground-temperature assumption."
- Unlocks: DEC-0013
- Needed evidence: Measured/declared parameters and applicability range

### Q-0012

How should defect coordinates/identities be matched across VTD campaigns, if at all?
- Evidence: `SRC-M2-TRANSCRIPT` L5: "это же другой участок, то есть 18-й год это угол 67 градусов, 22-й год 181 градус"
- Unlocks: defect-level features
- Needed evidence: Defect start/centre convention, tool orientation and matching tolerance

## Actions and dependencies

- **ACT-0001** [implemented]: Recompute source and target counts from verified lengths and segmentation rules; do not force 2,000 rows. (owner: not supplied; due: not supplied; dependency: nan)
- **ACT-0002** [implemented]: Re-run source parsing comparison on concrete raw rows and record real parser defects separately from ambiguous measurements. (owner: not supplied; due: not supplied; dependency: nan)
- **ACT-0003** [blocked]: Compare minimum-of-prior-inspections and latest-prior-inspection baselines on the same forward-time splits. (owner: not supplied; due: not supplied; dependency: Q-0013; eligible independent folds)
- **ACT-0004** [open]: Request the heat-loss formula parameters/units and the ground-temperature evidence. (owner: not supplied; due: not supplied; dependency: Q-0010)
- **ACT-0005** [open]: Request the RepCorr ID/direction crosswalk and approved transfer policy. (owner: not supplied; due: not supplied; dependency: Q-0004)
- **ACT-0006** [implemented]: Publish a separate UZT priority table and keep result.csv to three columns. (owner: not supplied; due: not supplied; dependency: nan)
- **ACT-0007** [blocked]: Do not fit or select on the final validation labels; compare the baseline, baseline plus temperature scenario, and additional agreed features on identical splits. (owner: not supplied; due: not supplied; dependency: Q-0009; Q-0010)
- **ACT-0008** [implemented]: Generate follow-up drafts locally; do not send messages or schedule calls without explicit authorization. (owner: not supplied; due: not supplied; dependency: nan)

## Source coverage gaps

- SRC-M1-TRANSCRIPT: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-M1-SUMMARY: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-M2-TRANSCRIPT: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-M2-SUMMARY: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-M2-STRICT-REVIEW: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-CLEANING-RULES: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-TECH-SPEC: unknown metadata ['stated_meeting_date', 'participants']; text_available=True.
- SRC-CASE-PDF: unknown metadata ['stated_meeting_date', 'participants']; text_available=False.
- SRC-CRITERIA-PDF: unknown metadata ['stated_meeting_date', 'participants']; text_available=False.
- SRC-PAOT-PDF: unknown metadata ['stated_meeting_date', 'participants']; text_available=False.

Do not send these drafts without explicit authorization.
