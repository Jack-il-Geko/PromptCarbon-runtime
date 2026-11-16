# Constraint Extractor: current evaluation

Date: 16 November 2026. One hundred synthetic clear English requests, evaluated with the frozen BERT-Medium checkpoint and decoder. No training or decoder modifications. This sample replaces the previous published evaluations; previous files remain in Git history.

## Dataset

50 short requests and 50 long requests. Long means more than 512 BERT content tokens, excluding special tokens. The long requests contain source reports and require multiple encoder windows. Ten source reports each support five requests, so the 50 long prompts are not 50 independent source documents. The 50 short requests were retained by saved category/order without consulting their prediction errors; they had already been evaluated. Long reports and labels were fixed before inference.

60 prompts contain global word requirements: 12 maximum, 12 minimum, 12 exact, 12 approximate and 12 ranges. The other 40 have no global word constraint: 10 ordinary requests, 10 irrelevant numerical facts, 10 local-only requirements, and 10 source-number cases. There are no exact normalized prompt matches with the component's training or development-validation datasets. This exact-text audit does not rule out semantic similarity.

## Results

| Group | Correct exact extractions | Accuracy |
| --- | ---: | ---: |
| All prompts | 98/100 | 98.00% |
| Short prompts | 49/50 | 98.00% |
| Long prompts | 49/50 | 98.00% |
| Global word constraint present | 59/60 | 98.33% |
| No global word constraint | 39/40 | 97.50% |

Individual-constraint precision: 97.30%; recall: 100.00%; F1: 98.63%. Two false constraints across two prompts; no expected global word constraint was missed. Exact comparison ignores the order of constraints, but requires matching type, value and unit.

## Application decisions

Automatic results were accepted for 94/100 requests. Of those, 93 were correct (98.94% accepted accuracy). Six requests required manual input and are not counted as automatically successful. Thus the application returned a correct automatic result for 93/100 requests. For long requests, 44/50 were accepted automatically and all 44 were correct; six required manual input. Raw extraction accuracy and automatic coverage measure different things.

## Scope

The prompts and labels were authored by an assistant, not independently annotated by real users. They have clear instructions, regular wording and shared templates. The short half was already examined. Multiple long prompts share source material and common report structure. The result is a controlled evaluation on the specified sample, not an independent estimate for arbitrary real users. The higher score does not prove the model improved: its weights and decoder are unchanged.

`results.csv` contains all prompts, expected constraints, predictions and automatic/manual decisions. `metrics.json` contains aggregate counts; `protocol.json` records selection, hashes and length scope. `../component_results.xlsx` has one `Constraint Extractor` sheet for the same 100 prompts. Superseded results are available in Git history rather than additional current files.
