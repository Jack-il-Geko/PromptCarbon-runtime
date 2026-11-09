# PromptCarbon

PromptCarbon analyses an English prompt, estimates its output tokens and compares its estimated carbon footprint across six half-hour periods. It runs locally in your browser and does not generate an LLM response.

## Requirements

- **Mac with Apple Silicon** (M1 or later). Tested on MacBook Air M4 with 16 GB RAM; other platforms have not been tested.
- **Python 3.12**.
- Internet for installation and electricity forecasts. GPT token counting also needs Internet and your own OpenAI API key.
- About **1.5 GB of disk space**, plus room for temporary downloads.

## 1. Check Python

Open Terminal and run:

```bash
python3.12 --version
```

It must show `Python 3.12.x`. If you see `command not found`, install Python 3.12 first. A macOS `.pkg` installer is available on the [Python 3.12.10 page](https://www.python.org/downloads/release/python-31210/) (an available installer, not the latest security release). Close and reopen Terminal, then check again.

## 2. Download PromptCarbon

Click **Code → Download ZIP** on this page and extract the ZIP.

Open Terminal, type `cd` followed by a space, drag the extracted folder into Terminal and press Enter. Run `ls`: you should see `install.py`.

## 3. Install — only the first time

```bash
python3.12 install.py --automatic-constraints
```

Wait for **Installed. Start with:**. The installer creates `.venv` and downloads all required packages, BERT models and the Qwen tokenizer. **No existing cache is needed.** It does not download the full Qwen language model.

The download is about **630 MB**, excluding Python. One installation without pip cache or preinstalled model assets took **67.54 seconds** on the author's M4; your time depends on the connection and computer. If interrupted, run the same command again.

## 4. Start and use

```bash
.venv/bin/python run.py
```

The browser opens automatically. If it does not, open the address printed after **PromptCarbon is running at**. Port 5500 is used when free; otherwise another local port is selected.

1. Enter your prompt **in English**.
2. Select **Qwen2.5-3B-Instruct** or **GPT-6-Sol** as the target model.
3. Choose **Automatic** to detect a word limit with BERT, or **Manual** to enter one or select **No word limit**.
4. Click **Update estimate** to see the analysis and environmental estimate.

The refresh button above the forecast table updates the carbon comparison using the saved token counts. **Run in ChatGPT** opens ChatGPT with your prompt; it does not run the local Qwen model or guarantee the selected target model in ChatGPT.

Keep Terminal open. To stop PromptCarbon, press **Ctrl+C** there. Closing the browser tab does not stop it.

**Next time:** open Terminal in the same folder and run only `.venv/bin/python run.py`. Do not install again.

## Important limits

- **Qwen2.5-3B-Instruct BF16** is the target used to train and evaluate the output-token predictor. The CFE uses a reference H100 scenario, not the measured consumption of your Mac.
- **GPT-6-Sol is experimental.** Its visible output length is transferred from the Qwen-trained RF and has not been validated on GPT responses. The CFE uses the installed EcoLogits catalog's estimated GPT architecture and latency parameters, displaying the midpoint of its architecture range. These are not confirmed proprietary hardware details; additional hidden reasoning tokens are outside the estimate.
- Times use your browser's local timezone. Electricity forecasts refer to **Great Britain**, regardless of your location. The first slot contains the current time.
- Automatic constraints can be wrong; use Manual if needed. Reasoning level is displayed but does not change the calculations. Model score is the task classifier's uncalibrated softmax score.
- The CFE uses EcoLogits 0.11.3, batch size 64, PUE 1.2 and a three-year hardware lifetime. Input-processing energy is not explicitly modelled; operational CO₂ is approximated as CO₂e when combined with hardware emissions. Savings are estimates, not guarantees.
- Qwen analysis needs no API key. GPT-6-Sol uses OpenAI's input-token counting endpoint without generating a response. It requires your own API key and access to the requested model. Create a local `.env` file in this folder containing `OPENAI_API_KEY=your_key_here`. Never commit or share this file. API errors are reported rather than replaced with a claimed exact local count.

## Evaluation data

Open [evaluation/](evaluation/README.md) and download one of the three workbooks:

- [component_results.xlsx](evaluation/component_results.xlsx): expected and predicted results for task classification, word constraints and output tokens.
- [energy_comparison.xlsx](evaluation/energy_comparison.xlsx): the nine-prompt comparison between PromptCarbon analysis and direct Qwen execution.
- [ecologits_comparison.xlsx](evaluation/ecologits_comparison.xlsx): original and separately labelled demonstration comparisons with EcoLogits.

The workbooks are optional material for the thesis; the application does not load them. Component and energy data are also available as CSV. The evaluation guide explains reused validation, selected demonstrations and the measurement limits. Training datasets and full Qwen weights are not included.

## Problems

- **`python3.12: command not found`**: complete step 1.
- **`install.py` not found**: open Terminal in the extracted project folder.
- **Download failed**: check Internet and retry the install command.
- **Time comparison unavailable**: forecasts could not be retrieved; check Internet and retry the estimate. HTTPS certificates are verified using `certifi`.


## Licenses and sources

See `THIRD_PARTY_NOTICES.md` and `licenses/` for the original BERT and Qwen terms. BERT uses Apache 2.0; the Qwen tokenizer uses the Qwen Research License with non-commercial terms. No separate license has been selected for the original application code.

Improved using Qwen: Qwen-generated answers were used to train the output-token estimator.

[BERT-Medium](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) · [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) · [EcoLogits methodology](https://ecologits.ai/latest/methodology/llm_inference/) · [Carbon Intensity API](https://api.carbonintensity.org.uk/)
