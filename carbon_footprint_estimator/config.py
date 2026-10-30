"""Reference assumptions; these do not describe an actual Qwen deployment."""
ECOLOGITS_VERSION = "0.11.3"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
PARAMETERS_BILLION = 3.09  # Official model card, rounded total dense parameters.
MODEL_SOURCE = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"
METHODOLOGY = "https://ecologits.ai/latest/methodology/llm_inference/"
CODE_SOURCE = "https://github.com/genai-impact/ecologits/blob/v0.11.3/ecologits/impacts/llm.py"
PUE = 1.20  # Explicit scenario choice; documented provider value, not Qwen-specific.
API_URL = "https://api.carbonintensity.org.uk/intensity"
TIMEOUT_SECONDS = 10
LIMITATIONS = [
    "Reference NVIDIA H100 scenario; not measured Qwen deployment energy.",
    "Input tokens are reported but EcoLogits 0.11.3 does not model prompt-processing energy explicitly.",
    "Latency uses EcoLogits regression; no measured throughput or time-to-first-token is supplied.",
    "UK scenario uses Great Britain national forecasts (excluding Northern Ireland).",
    "API gCO2/kWh is approximated as gCO2e/kWh when combined with embodied GWP; other operational greenhouse gases are not added.",
    "Forecast savings are not guaranteed; output length and hardware assumptions are uncertain.",
]

# Verified against the pinned library source. Values are checked by tests.
REFERENCE = {
    "gpu_memory_gb": 80, "server_installed_gpus": 8, "batch_size": 64,
    "server_power_without_gpus_kw": 1.2, "hardware_lifetime_seconds": 94608000,
    "server_embodied_without_gpus_kgco2e": 5700, "gpu_embodied_kgco2e": 273,
    "weight_bits": 16,
}
