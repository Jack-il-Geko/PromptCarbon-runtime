# Evaluation results

## Open the Excel files

| File | Contents |
| --- | --- |
| [component_results.xlsx](component_results.xlsx) | Three sheets: Task Classifier (330 prompts), Constraint Extractor (200 prompts), and Output Token Estimator (260 prompts). Each record contains the prompt, expected or observed result, and prediction. |
| [energy_comparison.xlsx](energy_comparison.xlsx) | 100 prompts, 100 valid comparisons. Same three sheets, all 307 original/repeat measurement blocks, protocol and exact requests. |
| [ecologits_comparison.xlsx](ecologits_comparison.xlsx) | Primary full 260-prompt validation, followed by the preserved original 100-prompt comparison and separate error-selected demonstration. Saved Qwen responses are included. |

Download an Excel file to open it locally. GitHub does not display workbook sheets directly.

## CSV files

The same component and energy records are also provided as CSV for automated analysis. These files are optional evaluation material; the application does not load them.

| Folder | Sample | Result |
| --- | --- | --- |
| [task_classifier](task_classifier/results.csv) | Complete revised Test Set v2: 330 prompts | 311 correct, 94.24% accuracy |
| [constraint_extractor](constraint_extractor/README.md) | 200 synthetic requests: 70 short and 130 long (over 512 BERT content tokens) | 187 exact extractions (93.50%); 188 automatic results accepted, 176 correct. |
| [output_tokens](output_tokens/results.csv) | Complete supported validation sample: 260 responses | MAE 101.81 tokens in this export |
| [energy](energy/) | 100 randomly selected requests from the existing validation pool | 300 initial PromptCarbon workflows, 18 repeat workflows and a ten-analysis block; 200 complete Qwen responses; 100 valid pairs |

## Full environmental validation (RQ2)

The main comparison uses all 260 original complete validation examples: 188 without structured preferences and 72 with word preferences. It reuses the frozen original predictions (MAE 101.82 tokens), not the browser-normalized CSV predictions. No new inference, training, generated responses or error-based filtering was used.

- [ecologits/results.csv](ecologits/results.csv): every pair of token counts, energy, operational emissions, embodied emissions and total emissions, with individual errors.
- [ecologits/summary.csv](ecologits/summary.csv): means, MAE, NMAE, median absolute percentage error and signed aggregate differences.
- [ecologits/protocol.json](ecologits/protocol.json): assumptions, definitions and original-source hashes.

The workbook opens with `Validation 260` (summary and individual comparisons) and `Prompts 260` (exact original prompts, saved responses and source IDs). All four earlier worksheets retain their original cells and formatting. IDs are local to each sample; use Source ID to link records.

Energy MAE is 8.470519 mWh; operational-emission MAE is 2.176923 mgCO2; total-emission MAE is 2.865203 mgCO2e (approximate). NMAE is 29.88%, median absolute percentage error 24.12%, and the signed aggregate difference -14.59% for each quantity. NMAE and the signed aggregate difference are different measures; the latter permits cancellation. The old 100-prompt workbook percentage is MAPE, not NMAE.

Both estimates use EcoLogits 0.11.3, Qwen2.5-3B-Instruct, 3.09B dense parameters, 16-bit H100 reference, batch 64, PUE 1.20, a three-year hardware lifetime and 257 gCO2/kWh. Only output length changes. Operational CO2 is treated as a CO2e approximation in the total. This measures prediction-error propagation, not physical environmental accuracy or independent generalisation.

## Interpretation

Component samples include successful and unsuccessful predictions and were not selected by prediction error. The task test was revised and already used during development. Constraint extraction uses a revised synthetic sample, including previously evaluated short prompts and long variants of twenty reports. It is not independent real-user validation. RF results use development validation, not a new independent final test. The export preserves browser-normalized prompt records; the historical RF report retains its original-record MAE of 101.82 tokens.

Constraint results evaluate decoder word constraints, including correct empty results. They do not score the server's decision to request manual input.

In addition to the full validation, the EcoLogits workbook keeps two historical 100-prompt samples. The original sample was selected without prediction-error filtering. The demonstration sample was chosen after inspecting errors and cannot establish general prediction accuracy. Both reuse development records. The reference calculation uses observed visible Qwen output length, while PromptCarbon uses predicted length under the same environmental assumptions. This is not a comparison with measured physical emissions.

## Energy records

- [summary.csv](energy/summary.csv): overall, automatic-only and confirmed-manual averages. Prompts have equal weight; Qwen repetitions are averaged first.
- [per_prompt.csv](energy/per_prompt.csv): all 100 valid prompt-level comparisons.
- [runs.csv](energy/runs.csv): all 307 blocks: the original 300, six three-analysis repeats and the P-en13 ten-analysis block. Attempt, operation count, active-observation and used-in-summary fields identify each record's role. Gross and idle-corrected energy, baseline power and coverage are retained.
- [prompts.csv](energy/prompts.csv): original requests, source IDs, confirmed preferences and exact final Qwen requests. IDs P-en1 to P-en100 match the workbook.
- [protocol.json](energy/protocol.json): selection, hashes, generation settings, baselines, repeat authorisation and exclusions.

The 100 requests were randomly sampled without replacement from the existing 260-request validation pool using seed 20261116. Selection did not use prediction errors, response lengths or energy outcomes. Forty-eight input requests exceed 512 tokens. Related source texts occur, so the sample does not represent all possible user requests.

PromptCarbon uses the public pipeline without the Reasoning Level Estimator. The original experiment performed three workflows per prompt (300 total). Eighty-four prompts complete automatically; sixteen require a confirmed word preference. Each affected workflow includes its automatic attempt and manual recalculation (48 extra recalculations). Human reading and clicking time are excluded.

Qwen2.5-3B-Instruct executes the final request prepared by the application, including the manual word instruction when present. It is not asked to estimate environmental impact. All 200 responses end naturally; completion does not establish answer correctness. Models are loaded and warmed up before measurement, systems run separately on the same device, and fixed forecasts avoid network requests.

`powermetrics` requests CPU + GPU + ANE power samples every 10 ms. Gross energy integrates reported power over recorded sample durations. Thirty-second resting baselines are measured before and after each session. The mean baseline power times execution duration is subtracted from gross energy. Repeats use their own session baseline; other observations keep their original baselines. Both versions remain in the CSVs; the workbook uses idle-corrected energy. These are component estimates, not whole-computer or application-only electricity.

The six initial invalid windows were repeated once with three workflows each and unchanged coverage criteria; five passed. P-en13 was then measured in one authorised block of ten analyses to increase measurement duration. It passed the same criteria. Its block time and energy are divided by ten; other accepted PromptCarbon blocks are divided by three. Each prompt retains equal weight, producing 100 valid pairs (84 automatic and 16 with manual confirmation). All prior attempts remain recorded. No prompt was replaced, no Qwen response was regenerated and no energy value was used to select a repeat. Short blocks retain sampling-boundary uncertainty.

On the 100 valid pairs, mean idle-corrected energy is 0.000317366 Wh for PromptCarbon and 0.030443810 Wh for Qwen generation, corresponding to 98.96% lower energy. Analysis and generation are different operations. This comparison does not validate carbon predictions; running both adds the analysis cost.

Operational carbon uses the same fixed 200 gCO2/kWh reference intensity for both systems and excludes embodied emissions. These are calculated emissions, not direct carbon measurements.

## Sources

The private research repository retains original datasets, source annotations and experimental records, including the earlier three- and nine-prompt energy experiments. Source identifiers are retained where available. The expanded energy experiment generated 200 new Qwen responses; it did not retrain any model. Published workbooks preserve the reviewed cell values, formulas and formatting. The component workbook omits embedded local-folder metadata. Original workbook hashes, publication hashes and sample scopes are recorded in [metadata.json](metadata.json).

Third-party dataset text retains its original terms. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
