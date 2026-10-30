# PromptCarbon

Local research prototype that counts input tokens, classifies the task and reasoning level, handles word limits, predicts visible output tokens and estimates a reference carbon footprint for Qwen2.5-3B-Instruct. Built with Qwen outputs for the experimental estimator (Improved using Qwen).

## Installation

Tested target: macOS Apple Silicon, Python 3.12, 16 GB RAM. Other systems are untested. A clean environment installation using cached pip wheels and a local verified asset mirror took 22.7 seconds; public network download time has not been measured. Install Python 3.12 first; no Apple-specific path is required.

```bash
git clone --depth 1 https://github.com/Jack-il-Geko/PromptCarbon-runtime.git
cd PromptCarbon-runtime
python3.12 install.py
.venv/bin/python run.py
```

Open http://127.0.0.1:5500. Stop with Ctrl+C. After installation, only the last command is needed. The interface retains the existing buttons. Manual word limits are the default; BERT extraction remains optional.

To install the optional constraint checkpoint immediately:

```bash
python3.12 install.py --automatic-constraints
```

Otherwise its files are downloaded when automatic extraction is first selected. The first installation needs Internet access. Models are cached in their local model folders and verified by SHA256; subsequent starts reuse them. A missing/unpublished release asset produces an explicit error; there is no substitute model.

Runtime checkpoints and the official tokenizer are hosted as versioned assets in this repository’s runtime-v1 GitHub Release. The installer verifies their SHA256 checksums before use.

## What is downloaded

Only the three fine-tuned BERT checkpoints actually used by the application are available as runtime assets: Task Classifier and Reasoning Level in the core installation; automatic Constraint Extractor optionally. The original RF is included in the source. Qwen files contain only its pinned official tokenizer, never generative weights. No MLX, alternative strategy, training dataset, generation, benchmark or Git LFS model download is required.

The runtime source ZIP is about 1.46 MB. Core assets are 311.54 MB compressed; automatic constraints add 152.91 MB. The tested Python wheels add about 166 MB, so a first install is approximately 480 MB (core) or 630 MB (all models), excluding the Python installer and small package metadata. Expanded files and the environment need roughly 1.5 GB disk space with all models. Exact compressed and expanded asset byte counts are in runtime_manifest.json. Installation also downloads Python dependencies, notably PyTorch; actual duration depends on network, Python version and existing package caches. install.py records elapsed time in .cache/install_metrics.json. Automatic downloads point to versioned GitHub Release assets, not an unversioned model alias.

## Use and limits

Models and algorithms are the selected project versions, with unchanged weights and preprocessing. English prompts are the main supported use. Quantitative preferences are in words. Reasoning is reported but does not change the RF estimate. Output estimation and CFE support Qwen2.5-3B-Instruct BF16 only. No Qwen response is generated.

The CFE uses EcoLogits 0.11.3 on the local CPU, with a reference H100 scenario, batch 64, PUE 1.20 and three-year lifetime. It queries the Great Britain carbon-intensity API for six half-hour forecasts. It does not explicitly model prompt-processing energy and approximates operational CO2 as CO2e when combining impacts. These are scenario estimates, not measured deployment emissions. Forecast failures preserve the energy/embodied estimate, with no time recommendation.

GPT-5 input counting remains optional and requires the user's own API key in a private local .env file containing OPENAI_API_KEY. It is not needed for Qwen or the CFE. Never commit that file. Unknown-model approximate counting uses tiktoken and may download its encoding on first use.

## Technical checks

```bash
.venv/bin/python -B -m unittest carbon_footprint_estimator.test_cfe input_tokenizer.test_tokenizer output_token_estimator.tests.test_estimator
```

Forecasts and OpenAI calls are mocked in tests. No training or generative inference occurs. Optional Node checks can be skipped when Node is absent. Install the constraint checkpoint before running all tests. A change of installation path does not change the model outputs. Run from a writable directory because checkpoint downloads and caches are stored beside the application.

## Files

server.py and index.html provide local APIs and the interface; run.py starts the app; runtime_assets.py verifies/downloads pinned assets; install.py creates the environment. input_tokenizer/, analyser/, output_token_estimator/ and carbon_footprint_estimator/ contain the definitive components. requirements.txt fixes runtime dependencies. Tests are technical regressions, not a new independent statistical evaluation.

## Licenses and reference sources

The author's application licensing must be selected by the owner before offering a general open-source license. Publishing on GitHub does not automatically grant a permissive license. BERT derives from Google's Apache-2.0 model; the included Qwen tokenizer is subject to the Qwen Research License, including its non-commercial scope. Do not label the entire distribution MIT/Apache without resolving these terms. See THIRD_PARTY_NOTICES.md and licenses/.

Sources: https://huggingface.co/google/bert_uncased_L-8_H-512_A-8 ; https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ; https://ecologits.ai/latest/methodology/llm_inference/ ; https://api.carbonintensity.org.uk/ .
