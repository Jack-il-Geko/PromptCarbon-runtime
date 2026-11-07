"""Use official EcoLogits calculation functions without provider instrumentation."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def calculate_local(input_tokens: int, output_tokens: int, target_model=None) -> dict:
    """Run the documented carbon/energy functions; no unused water or ADPe factors."""
    from importlib.metadata import version
    from ecologits.impacts import llm as e
    from carbon_footprint_estimator.config import ECOLOGITS_VERSION, PARAMETERS_BILLION, PUE, MODEL
    if version("ecologits") != ECOLOGITS_VERSION:
        raise RuntimeError("Expected EcoLogits " + ECOLOGITS_VERSION)
    for value in (input_tokens, output_tokens):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Token counts must be non-negative integers.")
    if target_model == "gpt-6-sol":
        return calculate_gpt(input_tokens, output_tokens, e)
    if target_model not in {None, MODEL}:
        raise ValueError("Unsupported carbon-estimation model.")
    p = PARAMETERS_BILLION
    memory = e.model_required_memory(p, 16)
    gpus = e.gpu_required_count(memory, e.GPU_MEMORY)
    duration = e.generation_latency(p, output_tokens, e.BATCH_SIZE, e.LATENCY_ALPHA, e.LATENCY_BETA, e.LATENCY_GAMMA, math.inf)
    gpu = e.gpu_energy(p, output_tokens, e.BATCH_SIZE, e.GPU_ENERGY_ALPHA, e.GPU_ENERGY_BETA, e.GPU_ENERGY_GAMMA)
    server = e.server_energy(duration, e.SERVER_POWER, e.SERVER_GPUS, gpus, e.BATCH_SIZE)
    energy = e.request_energy(PUE, e.request_it_energy(server, gpus, gpu))
    embodied = e.request_embodied_gwp(e.server_gpu_embodied_gwp(e.SERVER_EMBODIED_IMPACT_GWP, e.SERVER_GPUS, e.GPU_EMBODIED_IMPACT_GWP, gpus), e.HARDWARE_LIFESPAN, duration, e.BATCH_SIZE) * 1000
    if not all(math.isfinite(v) and v >= 0 for v in (energy, embodied, duration)):
        raise RuntimeError("Invalid EcoLogits result.")
    return dict(inputTokens=input_tokens, outputTokens=output_tokens, energyWh=energy*1000, energyKWh=energy, embodiedGCO2e=embodied, durationSeconds=duration,
        scenario=dict(model=MODEL, parametersBillion=p, precisionBits=16, gpu="NVIDIA H100 80GB", requiredGPUs=gpus, installedGPUs=e.SERVER_GPUS, batchSize=e.BATCH_SIZE, serverPowerKW=e.SERVER_POWER, pue=PUE, lifetimeSeconds=e.HARDWARE_LIFESPAN, serverEmbodiedKgCO2e=e.SERVER_EMBODIED_IMPACT_GWP, gpuEmbodiedKgCO2e=e.GPU_EMBODIED_IMPACT_GWP, ecologitsVersion=ECOLOGITS_VERSION),
        allocation=dict(gpuEnergyKWh=gpus*gpu, serverEnergyKWh=server))


def calculate_gpt(input_tokens, output_tokens, e):
    """Use catalog bounds and latency for a visible-output GPT reference estimate."""
    from importlib.resources import files
    from .config import ECOLOGITS_VERSION, PUE

    catalog = json.loads(files("ecologits").joinpath("data/models.json").read_text())
    model = next(item for item in catalog["models"]
                 if item["provider"] == "openai" and item["name"] == "gpt-6-sol")
    parameters = model["architecture"]["parameters"]
    total = parameters["total"]
    active = parameters["active"]
    deployment = model["deployment"]
    gpus = e.gpu_required_count(e.model_required_memory(total, 16), e.GPU_MEMORY)
    duration = e.generation_latency(active["min"], output_tokens, e.BATCH_SIZE,
        e.LATENCY_ALPHA, e.LATENCY_BETA, e.LATENCY_GAMMA, math.inf,
        tps=deployment["tps"], ttft=deployment["ttft"])
    server = e.server_energy(duration, e.SERVER_POWER, e.SERVER_GPUS, gpus, e.BATCH_SIZE)
    energies = []
    for bound in (active["min"], active["max"]):
        gpu = e.gpu_energy(bound, output_tokens, e.BATCH_SIZE,
                           e.GPU_ENERGY_ALPHA, e.GPU_ENERGY_BETA, e.GPU_ENERGY_GAMMA)
        energies.append(e.request_energy(PUE, e.request_it_energy(server, gpus, gpu)))
    energy = sum(energies) / 2
    embodied = e.request_embodied_gwp(
        e.server_gpu_embodied_gwp(e.SERVER_EMBODIED_IMPACT_GWP, e.SERVER_GPUS,
                                 e.GPU_EMBODIED_IMPACT_GWP, gpus),
        e.HARDWARE_LIFESPAN, duration, e.BATCH_SIZE) * 1000
    if not all(math.isfinite(value) and value >= 0 for value in (*energies, duration, embodied)):
        raise RuntimeError("Invalid EcoLogits GPT result.")
    return dict(inputTokens=input_tokens, outputTokens=output_tokens,
        energyWh=energy*1000, energyKWh=energy, embodiedGCO2e=embodied,
        durationSeconds=duration,
        energyWhRange=dict(min=energies[0]*1000, max=energies[1]*1000),
        scenario=dict(model="gpt-6-sol", parametersBillion=total,
            activeParametersBillion=active, precisionBits=16, gpu="NVIDIA H100 80GB",
            requiredGPUs=gpus, installedGPUs=e.SERVER_GPUS, batchSize=e.BATCH_SIZE,
            serverPowerKW=e.SERVER_POWER, pue=PUE, lifetimeSeconds=e.HARDWARE_LIFESPAN,
            ecologitsVersion=ECOLOGITS_VERSION, tokensPerSecond=deployment["tps"],
            timeToFirstTokenSeconds=deployment["ttft"], sources=model["sources"],
            warnings=model["warnings"], estimateBasis="visible output; range midpoint"))


def estimate_energy(input_tokens: int, output_tokens: int, target_model=None) -> dict:
    """Calculate impacts in the application's shared Python 3.12 environment."""
    if target_model is None:
        return calculate_local(input_tokens, output_tokens)
    return calculate_local(input_tokens, output_tokens, target_model=target_model)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print(json.dumps(calculate_local(*json.load(sys.stdin)), allow_nan=False))
