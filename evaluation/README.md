# Evaluation results

## Open the Excel files

| File | Contents |
| --- | --- |
| [component_results.xlsx](component_results.xlsx) | Three sheets: Task Classifier (330 prompts), Constraint Extractor (200 prompts), and Output Token Estimator (260 prompts). Each record contains the prompt, expected or observed result, and prediction. |
| [energy_comparison.xlsx](energy_comparison.xlsx) | 100 prompts, 94 valid energy comparisons. Same three sheets: summary, all 300 measurement blocks, protocol and requests. Idle-corrected energy; six invalid windows are retained and flagged. |
| [ecologits_comparison.xlsx](ecologits_comparison.xlsx) | Original 100-prompt comparison and a separate 100-prompt demonstration selected after observing errors. Both samples and their Qwen responses are retained and labelled. |

Download an Excel file to open it locally. GitHub does not display workbook sheets directly.

## CSV files

The same component and energy records are also provided as CSV for automated analysis. These files are optional evaluation material; the application does not load them.

| Folder | Sample | Result |
| --- | --- | --- |
| [task_classifier](task_classifier/results.csv) | Complete revised Test Set v2: 330 prompts | 311 correct, 94.24% accuracy |
| [constraint_extractor](constraint_extractor/README.md) | 200 synthetic requests: 70 short and 130 long (over 512 BERT content tokens) | 187 exact extractions (93.50%); 188 automatic results accepted, 176 correct. |
| [output_tokens](output_tokens/results.csv) | Complete supported validation sample: 260 responses | MAE 101.81 tokens in this export |
| [energy](energy/) | 100 randomly selected requests from the existing validation pool | 300 PromptCarbon workflows, 200 complete Qwen responses; 94 valid energy pairs |

## Interpretation

Component samples include successful and unsuccessful predictions and were not selected by prediction error. The task test was revised and already used during development. Constraint extraction uses a revised synthetic sample, including previously evaluated short prompts and long variants of twenty reports. It is not independent real-user validation. RF results use development validation, not a new independent final test. The export preserves browser-normalized prompt records; the historical RF report retains its original-record MAE of 101.82 tokens.

Constraint results evaluate decoder word constraints, including correct empty results. They do not score the server's decision to request manual input.

The EcoLogits workbook keeps two distinct samples. The original sample was selected without prediction-error filtering. The demonstration sample was chosen after inspecting errors and cannot establish general prediction accuracy. Both reuse development records. The reference calculation uses observed visible Qwen output length, while PromptCarbon uses predicted length under the same environmental assumptions. This is not a comparison with measured physical emissions.

## Energy records

- [summary.csv](energy/summary.csv): overall, automatic-only and confirmed-manual averages. Prompts have equal weight; Qwen repetitions are averaged first.
- [per_prompt.csv](energy/per_prompt.csv): all 100 request-level comparisons, including six invalid measurement pairs.
- [runs.csv](energy/runs.csv): all 300 measurement blocks, with gross and idle-corrected energy, baseline power and timing coverage.
- [prompts.csv](energy/prompts.csv): original requests, source IDs, manual preferences where needed and exact final Qwen requests. IDs P-en1 to P-en100 match the workbook.
- [protocol.json](energy/protocol.json): selection, source hashes, generation settings, baselines, exclusions and scope.

The 100 requests were randomly sampled without replacement from the existing 260-request validation pool using seed 20261116. Selection did not use prediction errors, response lengths or energy outcomes. Forty-eight input requests exceed 512 tokens. The sample contains related source texts and does not represent all possible user requests.

PromptCarbon uses the current public pipeline without the Reasoning Level Estimator. For each prompt it performs one block of three workflows. Eighty-four prompts complete automatically; sixteen require a confirmed word preference. Each affected workflow includes both its automatic attempt and the subsequent manual recalculation, giving 48 additional recalculations. Human reading and clicking time are excluded. These requests are retained rather than replaced.

Qwen2.5-3B-Instruct executes the final request prepared by the application, including the manual word instruction when present. It is not asked to estimate environmental impact. Two responses are generated per prompt; all 200 end naturally. Completion does not establish answer correctness. Models are loaded and warmed up before measurement, the systems run separately on the same device, and fixed forecasts avoid network requests.

`powermetrics` requests CPU + GPU + ANE (Apple Neural Engine) power samples every 10 ms. Gross energy integrates the reported power over the recorded sample durations. Before and after each system, a 30-second resting baseline is recorded with that system's models loaded. The mean baseline power multiplied by execution duration is subtracted from gross energy. Both versions remain in the CSVs; the workbook and main result use idle-corrected energy. This is estimated component energy, not whole-computer or application-only electricity.

Six PromptCarbon windows fail the predefined timing-coverage check. They remain recorded and are excluded from all paired averages, leaving 94 valid pairs (78 automatic and 16 with manual confirmation). Short three-workflow blocks retain sampling-boundary uncertainty. No failed window was automatically repeated and no prompt was replaced.

On the 94 valid pairs, mean idle-corrected energy is 0.000331423 Wh for PromptCarbon and 0.030920322 Wh for Qwen generation, corresponding to 98.93% lower energy. The means refer to different operations: preventive analysis and response generation. This comparison does not validate carbon predictions. Running both operations adds the analysis cost.

Operational carbon uses the same fixed reference intensity of 200 gCO2/kWh for both systems and excludes embodied emissions. These are calculated emissions, not direct carbon measurements.

## Sources

The private research repository retains original datasets, source annotations and experimental records, including the earlier three- and nine-prompt energy experiments. Source identifiers are retained where available. The expanded energy experiment generated 200 new Qwen responses; it did not retrain any model. Published workbooks preserve the reviewed cell values, formulas and formatting. The component workbook omits embedded local-folder metadata. Original workbook hashes, publication hashes and sample scopes are recorded in [metadata.json](metadata.json).

Third-party dataset text retains its original terms. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
