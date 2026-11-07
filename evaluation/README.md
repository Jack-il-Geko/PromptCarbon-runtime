# Evaluation data

Open a folder below, then open its CSV file. Each row contains one prompt and its expected or observed result alongside the model prediction. CSV files use UTF-8 and can also be opened in Excel.

| Folder | Data | Sample | Result |
| --- | --- | --- | --- |
| [task_classifier](task_classifier/results.csv) | Expected and predicted task, with correctness | Complete Test Set v2, 330 prompts | 311 correct, 94.24% accuracy |
| [constraint_extractor](constraint_extractor/results.csv) | Expected and extracted word constraints, with correctness | Complete historical test, 200 prompts | 190 correct, 95.00% exact match |
| [output_tokens](output_tokens/results.csv) | Actual Qwen tokens, estimated tokens and absolute error | Complete supported validation sample, 260 prompts | MAE 101.81 tokens |
| [energy](energy/) | Time and energy of PromptCarbon analysis versus actual Qwen response generation | Three prompts, three blocks per system and prompt | Individual measurements and means |

## Reading the component files

- `id` is the short display ID. `source_id` identifies the original project record.
- `correct` and `incorrect` are Boolean values. Both successful and unsuccessful predictions are retained.
- Task Classifier uses the current BERT-Medium implementation, including its narrow opening summary-instruction check.
- Constraint Extractor compares the current decoder's **word-only** output. JSON `null` means no word-based constraint. Sentence and paragraph constraints are outside this comparison. Lists are compared without depending on their order. These are extraction results, not the server's decision to accept a prediction or ask for manual input.
- Output Token Estimator uses the actual visible lengths of saved, complete Qwen2.5-3B-Instruct responses. Predictions use the current RF, predicted task and recorded manual word preference. Prompts include the corresponding length instruction when applicable. Minor outer whitespace and line-ending normalization can occur in the workbook/browser export. `source_id` preserves the link to the original record.

## What these results establish

The component samples are complete and were not selected by prediction error. The revised task test and historical synthetic constraint test had already been used or examined during development. The RF sample is reused validation, not an independent final test. This export adds no training or Qwen generations.

These files do **not** include the separate 100-example CFE demonstration selected for low prediction error. They do not validate environmental estimates against physical inference emissions. Source and exported-file hashes, sample definitions and metrics are in [metadata.json](metadata.json).

## Energy files

- [summary.csv](energy/summary.csv): three means for time, estimated energy and the percentage difference.
- [runs.csv](energy/runs.csv): all 18 measurement blocks, including power, sample coverage and per-operation values.
- [prompts.csv](energy/prompts.csv): the three exact prompts submitted to both systems.
- [protocol.json](energy/protocol.json): model settings and measurement scope.

Qwen executes the prompt itself. It is not asked to estimate its impact. PromptCarbon performs its complete analysis and CFE calculation with fixed forecasts. Both use the same Mac, with models already loaded and the systems running separately. Each PromptCarbon block contains 50 analyses; dividing its totals by 50 gives one analysis. Each Qwen block contains one response.

`powermetrics` samples estimated CPU + GPU + ANE power approximately every 100 ms. Energy is the sum of power times sample duration, divided by 3600. This is estimated component energy during the sampled window, not whole-Mac or application-only energy. No idle consumption is subtracted. ANE means Apple Neural Engine.

Operational carbon is calculated using the same fixed reference intensity of 200 gCO2/kWh for both systems. It is not measured carbon and does not include embodied emissions. All nine Qwen responses ended naturally. Completion is not a correctness score: the programming answer did not satisfy every requested requirement. The small case study compares the cost of analysis with the cost of generation; running both adds the analysis cost.

## Sources and terms

The project combines authored prompts and material from existing public datasets. The private research repository retains training data, generation records, protocols and source annotations. Original identifiers are preserved here through `source_id`; these files do not grant a new blanket licence to third-party text. Model and software terms are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
