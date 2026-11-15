# Second mentor meeting: agreed approach

Sources: the transcript `C:\Users\Honor\Downloads\Менторы-2.txt`, the mentor's heat-loss equation shared in the screenshot, and the team's subsequent decision on handling inspection data.

## Inspection data: selected rule

For each 10 m segment, use the smallest residual wall thickness from the 2018 and 2022 inspections of that same segment:

`initial_thickness = min(minimum_2018, minimum_2022)`

If only one prior measurement exists, use that measurement and mark the missing inspection. If neither exists, do not invent a measurement: use and disclose the separately chosen fallback.

For every candidate residual wall thickness, including observations associated with welds or full-circumference fittings, compare the thickness in mm with the passport nominal wall thickness of the same section. Exclude a candidate from the segment-minimum calculation if its residual thickness is greater than that nominal value. If it is at or below nominal and passes the other existing validity checks, include it when calculating the minimum: it changes the result only if it is smaller than the current eligible minimum. This comparison concerns wall thickness, not defect length or width. Keep excluded records in the audit with the reason; do not overwrite the source data.

This is the team's working filter, not proof that every above-nominal observation is erroneous. If the passport nominal is missing or unreliable, flag the comparison as unresolved instead of silently excluding or accepting the record. Apply the filter to individual observations before calculating each inspection's segment minimum; if no eligible value remains, treat that inspection as missing.

**We do not use circumferential angle** to match, filter or model defects. We do not need to establish why the two inspections differ before proceeding with this approach.

**Time boundary:** apply the 2018/2022 minimum when predicting a date after both inspections, such as the first case on 1 December 2023. In a historical test predicting the 2022 inspection, the 2022 value is the answer and cannot be included in the input. For cold start, prior inspections of the target pipe are unavailable and must not be used.

The mentor presented the minimum of inspections as one possible approach; the team has now selected it as the working rule. Compare it against the latest-available-inspection baseline on appropriate historical tests.

## Heat-loss equation supplied by the mentor

The screenshot gives:

`T(x) = T_ground + (T_in - T_ground) * exp[-π * U * (d - 2ε) * x / (ṁ * c_p)]`

Where:

- `T(x)`: fluid temperature at distance `x`, °C.
- `T_ground`: ground temperature, °C; the meeting mentioned approximately 5 °C as an assumption.
- `T_in`: fluid temperature at the section inlet, °C.
- `x`: distance from the section inlet, m; section length `L` defines its range, normally `0 ≤ x ≤ L`.
- `d`: inner diameter, m, as labelled in the mentor's message.
- `ε`: roughness, m.
- `U`: heat-transfer coefficient, W/(m²·K).
- `ṁ * c_p`: mixture mass flow multiplied by its specific heat capacity, W/K.

The mentor said roughness, heat capacity and heat transfer can be treated as constants. Actual values, units and applicability must be specified before numerical implementation. Verify the mapping of `d`, flow and temperature fields to the actual dataset. Use inner diameter only if available or calculable from verified inputs; do not silently treat an outer diameter as inner diameter. Handle missing or nonpositive mass flow explicitly. Use the equation as supplied; confirm the `d - 2ε` term and which diameter the mentor intends before altering it.

This equation estimates temperature along a section. It does not by itself specify how temperature determines corrosion rate; that relationship and its parameters are separate modeling choices. The ambient 5 °C is an assumption from the discussion, not a verified measurement for every section or date.

## ML and validation

- Build features at the simple-section and 10 m segment level. Do not add angle features or defect matching.
- Compare the baseline, baseline plus the supplied temperature profile, and additional agreed features using identical splits and train-only transformations.
- Retain the 177 observed increases in underlying records. Apply the selected minimum rule to the forecast starting state where both prior inspections are available; do not overwrite raw data.
- Report squared Pearson correlation, MAE and RMSE separately for each case. The available spatial splits concern related simple sections; they do not establish performance on an independent pipeline.
- Keep the final validation observations out of feature selection and model fitting.

## Implementation details to clarify

1. Numerical values and units for roughness, heat capacity and the heat-transfer coefficient, and whether 5 °C is the intended ground-temperature assumption.
2. Confirm that `d` is the inner diameter in the supplied formula and the intended interpretation of `d - 2ε`.
3. Which available flow field should be used for mixture mass flow, and how to handle missing or zero flow.
4. How to initialize a segment when neither prior inspection supplies a usable thickness.

The team can continue preparing data and validation while these values are clarified.
