# Constraint Extractor: current evaluation

200 synthetic English requests, evaluated using the frozen model and decoder. No training. The decoder now recognises two additional instruction headings after source material: `Response length:` and `Response instruction:`. This replaces the previous current evaluation in the same files; prior versions remain in Git history.

## Sample

70 short requests and 130 long requests. Long means over 512 BERT content tokens, excluding special tokens. The earlier 100 requests are unchanged; 100 additional requests were frozen before inference (80 long, 20 short). No examples were filtered or rewritten after inspecting predictions. The additions contain 60 global word requirements, 12 for each maximum/minimum/exact/approximate/range type, plus 40 without a global word requirement.

The long requests use twenty source reports: ten reports with five requests and ten with eight requests. Report structure and instruction templates are shared. Some short requests were already examined. This is an assistant-authored synthetic sample, not independently annotated real-user data or 200 independent source documents.

## Results

| Group | Correct exact extractions | Accuracy |
| --- | ---: | ---: |
| Overall | 187/200 | 93.50% |
| Short | 68/70 | 97.14% |
| Long | 119/130 | 91.54% |
| Earlier 100 retained | 98/100 | 98.00% |
| Additional 100 | 89/100 | 89.00% |

Individual-constraint precision: 97.81%; recall: 93.06%; F1: 95.37%. There are 134 correctly extracted constraints, three false constraints and ten missing expected constraints. The heading fix recovers 30 exact extractions with no regressions on the same 200 requests. BERT weights, numeric candidate detection, windowing and all prompts/labels are unchanged. Ten additional targeted inference checks and two fenced-boundary checks passed; these are regression checks, not an independent accuracy sample.

## Automatic application behaviour

188 results were accepted automatically; 176 were correct (93.62% accuracy among accepted results). Twelve requests required manual input. Correct automatic results cover 176/200 requests (88.00%). Long requests: 119 accepted, of which 109 correct; eleven require manual input. Short requests: 69 accepted, of which 67 correct; one requires manual input. Abstentions are not successful automatic answers, and the fallback does not catch every extraction error.

## Files

`results.csv` contains every prompt, expected and extracted constraints, correctness and automatic/manual decisions. `metrics.json` contains counts; `protocol.json` records frozen selection, source relationships, code and model hashes. The existing `Constraint Extractor` sheet in `../component_results.xlsx` contains the same 200 requests. No additional workbook or worksheet is created. Prompt line breaks are compacted in the spreadsheet for readability; CSV and private source records preserve the input text.

The dataset remains synthetic and templated. The score describes the specified sample only. The sample informed this decoder correction and is now development validation. The results are not an independent final confirmation. Ten initial global-role misses remain outside this narrow fix. Earlier evaluations are available in Git history.
