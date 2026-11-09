# Evaluation results

## Open the Excel files

| File | Contents |
| --- | --- |
| [component_results.xlsx](component_results.xlsx) | Three sheets: Task Classifier (330 prompts), Constraint Extractor (200 prompts), and Output Token Estimator (260 prompts). Each record contains the prompt, expected or observed result, and prediction. |
| [energy_comparison.xlsx](energy_comparison.xlsx) | Nine prompts, three per category. Includes summary results, all 27 measurement blocks, the protocol and the exact prompts. |
| [ecologits_comparison.xlsx](ecologits_comparison.xlsx) | Original 100-prompt comparison and a separate 100-prompt demonstration selected after observing errors. Both samples and their Qwen responses are retained and labelled. |

Download an Excel file to open it locally. GitHub does not display workbook sheets directly.

## CSV files

The same component and energy records are also provided as CSV for automated analysis. These files are optional evaluation material; the application does not load them.

| Folder | Sample | Result |
| --- | --- | --- |
| [task_classifier](task_classifier/results.csv) | Complete revised Test Set v2: 330 prompts | 311 correct, 94.24% accuracy |
| [constraint_extractor](constraint_extractor/results.csv) | Complete historical synthetic sample: 200 prompts | 190 correct word-only extractions, 95.00% exact match |
| [output_tokens](output_tokens/results.csv) | Complete supported validation sample: 260 responses | MAE 101.81 tokens in this export |
| [energy](energy/) | Nine prompts: three simple, three programming and three complex | 9 PromptCarbon blocks of 10 analyses and 18 Qwen responses |

## Interpretation

Component samples include successful and unsuccessful predictions and were not selected by prediction error. The task test was revised and already used during development. Constraint extraction uses a previously examined synthetic sample. RF results use development validation, not a new independent final test. The export preserves browser-normalized prompt records; the historical RF report retains its original-record MAE of 101.82 tokens.

Constraint results evaluate decoder word constraints, including correct empty results. They do not score the server's decision to request manual input. Reasoning is not an input to either RF or the CFE.

The EcoLogits workbook keeps two distinct samples. The original sample was selected without prediction-error filtering. The demonstration sample was chosen after inspecting errors and cannot establish general prediction accuracy. Both reuse development records. The reference calculation uses observed visible Qwen output length, while PromptCarbon uses predicted length under the same environmental assumptions. This is not a comparison with measured physical emissions.

## Energy records

- [summary.csv](energy/summary.csv): category means, giving equal weight to each of the three prompts.
- [per_prompt.csv](energy/per_prompt.csv): the nine prompt-level comparisons.
- [runs.csv](energy/runs.csv): all 27 measurement blocks, including time, power, energy and sample coverage.
- [prompts.csv](energy/prompts.csv): the nine exact prompts. IDs P1 to P9 match the workbook.
- [protocol.json](energy/protocol.json): generation settings, sampling and measurement scope.

Qwen executes each original prompt directly. It is not asked to estimate environmental impact. PromptCarbon performs automatic analysis, output prediction and CFE calculations with fixed forecasts. Models were already loaded and warmed up, and the systems ran separately on the same device. Each PromptCarbon block contains ten analyses; block totals are divided by ten. Qwen generates two responses per prompt, which are averaged before calculating category means. All eighteen responses ended naturally; completion does not establish answer correctness.

`powermetrics` samples estimated CPU + GPU + ANE (Apple Neural Engine) power at requested intervals of 20 ms. Energy integrates power over recorded sample durations. It is sampled component energy, not whole-computer or application-only electricity. No idle-power subtraction is applied. Short blocks retain sampling-boundary uncertainty.

Operational carbon uses the same fixed reference intensity of 200 gCO2/kWh for both systems and excludes embodied emissions. These are calculated emissions, not direct carbon measurements. Analysis and response generation are different operations; running both adds the analysis cost.

## Sources

The private research repository retains original datasets, source annotations and experimental records, including the earlier three-prompt energy experiment. Source identifiers are retained where available. This publication adds no training or Qwen generations. Published workbooks preserve the reviewed cell values, formulas and formatting. The component workbook omits embedded local-folder metadata. Original workbook hashes, publication hashes and sample scopes are recorded in [metadata.json](metadata.json).

Third-party dataset text retains its original terms. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
