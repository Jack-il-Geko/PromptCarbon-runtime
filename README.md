# PromptCarbon

PromptCarbon is a university research project. It analyses a prompt, estimates how many output tokens it may produce and estimates its carbon footprint. The results are shown in a local web page.

All active components, including EcoLogits, run in one Python 3.12 virtual environment (`.venv`).

This repository contains the version for running the application. It does not include training datasets or previous experiments.

## What you need

- A Mac with Apple Silicon, such as M1, M2, M3 or M4. The application was tested on an M4 with 16 GB RAM. Windows, Linux and Intel Macs have not been tested.
- Python **3.12**, installed before starting.
- An Internet connection for installation and carbon-intensity forecasts.

You can download the project from your browser. Installation and startup still require a few Terminal commands; opening `index.html` directly does not start the application.

## Before starting: check Python

Open Terminal and run:

```bash
python3.12 --version
```

You should see **Python 3.12.x**. If you see **command not found**, Python 3.12 is not available as a Terminal command. Install Python 3.12 first, then close and reopen Terminal and repeat the check. Python 3.9, 3.13 or another version is not the version supported by this installer.

You can find a macOS **universal2 installer** on the [Python 3.12.10 release page](https://www.python.org/downloads/release/python-31210/). Open the downloaded `.pkg` and follow its installation steps. This is an available 3.12 installer, not a claim that it is the latest security release. You do not need to remove other Python versions.

A Python interpreter inside another application's private environment is not automatically available as `python3.12`. The public installation must not depend on the author's computer or folders.

## 1. Download the project

On this GitHub page, click **Code → Download ZIP**, then open the ZIP to extract the folder.

Open Terminal in the extracted folder. To check the location, run `ls`: the list must include `install.py`. On Mac, you can right-click the folder in Finder and look for **Services → New Terminal at Folder**. The menu name may differ with your system language.

If you prefer Git, use these commands instead of downloading the ZIP:

```bash
git clone --depth 1 https://github.com/Jack-il-Geko/PromptCarbon-runtime.git
cd PromptCarbon-runtime
```

Choose one download method. You do not need to do both.

## 2. Install — only the first time

From the project folder, run:

```bash
python3.12 install.py --automatic-constraints
```

This installs the Python packages and downloads the models, including the BERT constraint extractor used by the default automatic mode. It does not start the application. Wait until installation finishes before continuing.

**No existing cache, models or Python packages are required.** The installer creates `.venv`, installs the packages listed in `requirements.txt` from the Python package index and downloads the BERT models and Qwen tokenizer from the project's GitHub Release. Downloaded model files are checked against their SHA256 hashes. The Random Forest model is included in the project ZIP. The full Qwen language model is not downloaded.

Without a cache, the first installation downloads about **630 MB**, excluding Python itself, and needs roughly **1.5 GB** of disk space. Allow extra free space for temporary downloads. A cache only saves downloads; it is not needed for installation. Duration depends on the connection and computer, so no fixed installation time is promised. The observed **55.23 seconds** used cached Python packages and is not a measurement of a fully uncached installation.

`Collecting`, `Downloading` and `Installing collected packages` are normal progress messages. Wait until you see **Installed. Start with:** followed by a command. Only then continue to the startup step.

If the installation is interrupted, check your connection and run the same install command again. Installed packages and verified model files are reused where available; an incomplete model archive may need to download again. Do not delete the project folder just to retry.

For a smaller initial installation, you can use `python3.12 install.py` instead. This downloads about **480 MB**. The automatic constraint model will then be downloaded when you first run an estimate in automatic mode. Choose Manual before estimating if you do not want to download this optional model.

## 3. Start PromptCarbon

From the same project folder, run:

```bash
.venv/bin/python run.py
```

The browser opens automatically. PromptCarbon uses port **5500** when available; if it is occupied, it chooses a free local port. If the browser does not open, copy the address printed after **PromptCarbon is running at** into your browser.

Keep the Terminal window open while using the application. To stop it, press **Ctrl+C** in that window.

## Next time

Open Terminal in the project folder and run only:

```bash
.venv/bin/python run.py
```

You do not need to download or install everything again. The models are saved locally.

## Using the application

1. Enter your prompt in English and include the context needed to answer it. If you specify a word limit, make clear that it applies to the whole response. For best results in automatic mode, place it near the beginning or end.
2. Under **How to set the response limit**, leave **Automatic** selected to detect word limits with BERT, or choose **Manual** to enter a word limit (or select **No word limit**).
3. Click the green **Estimate environmental impact** button to see the carbon-footprint estimate and six coloured time periods. The grey **Estimate output tokens** button shows only the prompt analysis and output-token estimate.

The **Task** card shows the selected category, “Selected from 33 categories” and the unchanged softmax value as **Model score**. This score is not a calibrated probability that the answer is correct.

Times use your browser’s local time zone. The first slot contains the current time (for example, 15:00–15:30 at 15:15). Electricity forecasts still refer to Great Britain, not your location. **How is this estimated?** at the bottom explains the calculation. Closing the browser tab does not stop the server: use **Ctrl+C** in Terminal.

Output-token and carbon-footprint estimates refer to **Qwen2.5-3B-Instruct BF16**. PromptCarbon does not execute that language model or generate its answer. The estimates may differ from actual output length and energy consumption. Reasoning level is shown for information but does not change the Random Forest or carbon-footprint calculations. Automatic word-limit extraction can be wrong, especially with long or ambiguous prompts; use Manual if needed.

The carbon calculation uses EcoLogits 0.11.3 and a reference H100 scenario, with batch size 64, PUE 1.2 and a three-year hardware lifetime. It uses Great Britain carbon-intensity forecasts for six 30-minute periods. These are estimates, not measured emissions from your Mac. Input-processing energy is not explicitly modelled. If forecasts are unavailable, the application cannot recommend a time period.

No OpenAI API key is needed for Qwen token counting or the carbon estimator. The optional GPT-5 input-token counting feature needs your own key in a private `.env` file. Never upload that file to GitHub.

## If something does not work

- **`python3.12: command not found`**: complete the Python check above. Downloading PromptCarbon does not install Python itself.
- **`can't open file ... install.py`**: Terminal is not in the extracted project folder. Run `ls` and check that `install.py` is listed.
- **Package or model download error**: keep an Internet connection and retry the install command. Copy the error if it keeps failing. Installation needs access to the Python package index and GitHub Releases.
- **No browser window**: use the exact local address printed in Terminal. An occupied port 5500 is handled automatically.
- **No time recommendation**: the carbon-intensity forecast service may be unreachable. Installation and forecast availability are separate issues.

## Optional technical tests

**These tests do not start PromptCarbon and are not needed for normal use.** After installing with automatic constraints, you can run them with:

```bash
.venv/bin/python -B -m unittest carbon_footprint_estimator.test_cfe input_tokenizer.test_tokenizer output_token_estimator.tests.test_estimator
```

External API calls are simulated during these tests. No model is trained and no Qwen answer is generated.

## Licenses and sources

The `licenses/` folder contains the original BERT and Qwen license texts. `THIRD_PARTY_NOTICES.md` explains which external components are used. BERT uses Apache 2.0; the Qwen tokenizer uses the Qwen Research License, with non-commercial terms. No separate license has yet been selected for the original application code.

Improved using Qwen: Qwen-generated answers were used to train the experimental output-token estimator.

- [BERT-Medium](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8)
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [EcoLogits methodology](https://ecologits.ai/latest/methodology/llm_inference/)
- [Great Britain Carbon Intensity API](https://api.carbonintensity.org.uk/)
