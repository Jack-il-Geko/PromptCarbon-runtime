# Constraint Extractor: new 100-prompt evaluation

Date: 16 November 2026. Frozen model and decoder; no training or decoder changes.

## Sample

100 newly authored synthetic English requests. 60 have a global word requirement: 12 maximum, 12 minimum, 12 exact, 12 approximate and 12 ranges. 40 do not: 10 ordinary requests, 10 containing irrelevant numerical facts, 10 with only local constraints and 10 with numbers in source material. Order was shuffled with seed 20261116. Expected labels and dataset SHA256 were saved before inference. There are no exact normalized prompt duplicates within the sample or against the existing Constraint Extractor training, validation and historical test files. This is an exact-text audit, not proof of no semantic overlap.

## Results

| Measurement | Result |
| --- | ---: |
| Completely correct word-constraint extractions | 97/100 (97.00%) |
| With a global word constraint | 60/60 (100.00%) |
| Without a global word constraint | 37/40 (92.50%) |
| Individual-constraint precision | 94.74% |
| Individual-constraint recall | 100.00% |
| Individual-constraint F1 | 97.30% |
| Automatic results accepted | 98/100 |
| Correct accepted results | 96/98 (97.96%) |
| Manual input requested | 2/100 |
| Correct automatic results out of all requests | 96/100 |

The decoder produced four false constraints across three requests: one request containing irrelevant numbers and two containing local constraints. No expected global word constraint was missed. One correct decoder result was routed to manual input; a manual request is not counted as an automatically correct answer. Errors and manual decisions are included in the records, not discarded.

## Limits

These are assistant-authored synthetic requests with automatic expected labels, not independently annotated real-user data. They were written with knowledge of previous project findings. Global instructions use explicit wording and occur at the beginning or end. The 60 positive requests share five instruction templates. The prompts contain 16–56 BERT content tokens; none exceeds 512 tokens. The evaluation supports performance on short, clear English instructions, not long documents, ambiguous requests or arbitrary real users. No examples were changed after inspecting predictions. Historical 190/200 results remain separate and must not be pooled as a new 300-prompt independent test.

Files: `new_prompts.jsonl` contains the frozen requests and expected labels, `new_results.csv` contains every prediction and acceptance decision, `new_metrics.json` contains aggregate counts, and `new_protocol.json` records scope and hashes. The workbook `../component_results.xlsx` includes these data in the `Constraint Extractor New` sheet; the old 200-prompt sheet is retained.
