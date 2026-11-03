"""Use official EcoLogits calculation functions without provider instrumentation."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def calculate_local(input_tokens: int, output_tokens: int) -> dict:
    """Run the documented carbon/energy functions; no unused water or ADPe factors."""
    from importlib.metadata import version
    from ecologits.impacts import llm as e
    from carbon_footprint_estimator.config import ECOLOGITS_VERSION, PARAMETERS_BILLION, PUE, MODEL
    if version("ecologits") != ECOLOGITS_VERSION:
        raise RuntimeError("Expected EcoLogits " + ECOLOGITS_VERSION)
    for value in (input_tokens, output_tokens):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Token counts must be non-negative integers.")
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


def estimate_energy(input_tokens: int, output_tokens: int) -> dict:
    """Calculate impacts in the application's shared Python 3.12 environment."""
    return calculate_local(input_tokens, output_tokens)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print(json.dumps(calculate_local(*json.load(sys.stdin)), allow_nan=False))
