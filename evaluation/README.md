# Evaluation results

## Open the Excel files

| File | Contents |
| --- | --- |
| [component_results.xlsx](component_results.xlsx) | Three sheets: Task Classifier (330 prompts), Constraint Extractor (200 prompts), and Output Token Estimator (260 prompts). Each record contains the prompt, expected or observed result, and prediction. |
| [energy_comparison.xlsx](energy_comparison.xlsx) | 100 prompts, 99 valid energy comparisons. Same three sheets, all 306 original/repeat blocks, protocol and exact requests. |
| [ecologits_comparison.xlsx](ecologits_comparison.xlsx) | Original 100-prompt comparison and a separate 100-prompt demonstration selected after observing errors. Both samples and their Qwen responses are retained and labelled. |

Download an Excel file to open it locally. GitHub does not display workbook sheets directly.

## CSV files

The same component and energy records are also provided as CSV for automated analysis. These files are optional evaluation material; the application does not load them.

| Folder | Sample | Result |
| --- | --- | --- |
| [task_classifier](task_classifier/results.csv) | Complete revised Test Set v2: 330 prompts | 311 correct, 94.24% accuracy |
| [constraint_extractor](constraint_extractor/README.md) | 200 synthetic requests: 70 short and 130 long (over 512 BERT content tokens) | 187 exact extractions (93.50%); 188 automatic results accepted, 176 correct. |
| [output_tokens](output_tokens/results.csv) | Complete supported validation sample: 260 responses | MAE 101.81 tokens in this export |
| [energy](energy/) | 100 randomly selected requests from the existing validation pool | 300 initial PromptCarbon workflows plus 18 repeat workflows; 200 complete Qwen responses; 99 valid pairs |

## Interpretation

Component samples include successful and unsuccessful predictions and were not selected by prediction error. The task test was revised and already used during development. Constraint extraction uses a revised synthetic sample, including previously evaluated short prompts and long variants of twenty reports. It is not independent real-user validation. RF results use development validation, not a new independent final test. The export preserves browser-normalized prompt records; the historical RF report retains its original-record MAE of 101.82 tokens.

Constraint results evaluate decoder word constraints, including correct empty results. They do not score the server's decision to request manual input.

The EcoLogits workbook keeps two distinct samples. The original sample was selected without prediction-error filtering. The demonstration sample was chosen after inspecting errors and cannot establish general prediction accuracy. Both reuse development records. The reference calculation uses observed visible Qwen output length, while PromptCarbon uses predicted length under the same environmental assumptions. This is not a comparison with measured physical emissions.

## Energy records

- [summary.csv](energy/summary.csv): overall, automatic-only and confirmed-manual averages. Prompts have equal weight; Qwen repetitions are averaged first.
- [per_prompt.csv](energy/per_prompt.csv): all 100 comparisons, with P-en13 still flagged as invalid.
- [runs.csv](energy/runs.csv): all 306 blocks, including the original 300 and six authorised repeats. Attempt, active-observation and used-in-summary fields identify which records contribute. Gross and idle-corrected energy, baseline power and coverage are retained.
- [prompts.csv](energy/prompts.csv): original requests, source IDs, confirmed preferences and exact final Qwen requests. IDs P-en1 to P-en100 match the workbook.
- [protocol.json](energy/protocol.json): selection, hashes, generation settings, baselines, repeat authorisation and exclusions.

The 100 requests were randomly sampled without replacement from the existing 260-request validation pool using seed 20261116. Selection did not use prediction errors, response lengths or energy outcomes. Forty-eight input requests exceed 512 tokens. Related source texts occur, so the sample does not represent all possible user requests.

PromptCarbon uses the public pipeline without the Reasoning Level Estimator. The original experiment performed three workflows per prompt (300 total). Eighty-four prompts complete automatically; sixteen require a confirmed word preference. Each affected workflow includes its automatic attempt and manual recalculation (48 extra recalculations). Human reading and clicking time are excluded.

Qwen2.5-3B-Instruct executes the final request prepared by the application, including the manual word instruction when present. It is not asked to estimate environmental impact. All 200 responses end naturally; completion does not establish answer correctness. Models are loaded and warmed up before measurement, systems run separately on the same device, and fixed forecasts avoid network requests.

`powermetrics` requests CPU + GPU + ANE power samples every 10 ms. Gross energy integrates reported power over recorded sample durations. Thirty-second resting baselines are measured before and after each session. The mean baseline power times execution duration is subtracted from gross energy. Repeats use their own session baseline; other observations keep their original baselines. Both versions remain in the CSVs; the workbook uses idle-corrected energy. These are component estimates, not whole-computer or application-only electricity.

The six original PromptCarbon windows that failed the predefined coverage check were repeated once using the same requests, three workflows each and unchanged validity criteria. Five passed. P-en13 failed again and remains excluded, leaving 99 valid pairs (83 automatic, 16 with manual confirmation). All originals and repeats remain recorded. No prompt was replaced, no Qwen response was regenerated and no energy value was used to select a repeat. Short blocks retain sampling-boundary uncertainty.

On the 99 valid pairs, mean idle-corrected energy is 0.000319691 Wh for PromptCarbon and 0.030486462 Wh for Qwen generation, corresponding to 98.95% lower energy. Analysis and generation are different operations. This comparison does not validate carbon predictions; running both adds the analysis cost.

Operational carbon uses the same fixed 200 gCO2/kWh reference intensity for both systems and excludes embodied emissions. These are calculated emissions, not direct carbon measurements.

## Sources

The private research repository retains original datasets, source annotations and experimental records, including the earlier three- and nine-prompt energy experiments. Source identifiers are retained where available. The expanded energy experiment generated 200 new Qwen responses; it did not retrain any model. Published workbooks preserve the reviewed cell values, formulas and formatting. The component workbook omits embedded local-folder metadata. Original workbook hashes, publication hashes and sample scopes are recorded in [metadata.json](metadata.json).

Third-party dataset text retains its original terms. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
