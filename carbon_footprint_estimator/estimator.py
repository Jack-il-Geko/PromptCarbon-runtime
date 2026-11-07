"""Combine per-request EcoLogits energy with six electricity forecasts."""
import math
from .carbon_intensity import fetch_forecast, ForecastError
from .ecologits_adapter import estimate_energy
from .config import LIMITATIONS


def nonnegative(value):
    """Reject invalid numerical inputs before comparing impacts."""
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite non-negative number.")
    return value


def operational(energy_kwh, intensity):
    """Return operational grams using the declared CO2-to-CO2e approximation."""
    return nonnegative(energy_kwh)*nonnegative(intensity)


def sci(operational_g, embodied_g, requests=1):
    """Return total allocated grams per functional unit (one request by default)."""
    if nonnegative(requests) == 0:
        raise ValueError("The functional unit must be positive.")
    return (nonnegative(operational_g)+nonnegative(embodied_g))/requests


def compare_periods(energy_kwh, embodied_g, periods):
    """Select the earliest minimum, using unrounded values and a safe zero baseline."""
    if len(periods) != 6:
        raise ValueError("Six periods required.")
    rows = []
    for period in sorted(periods, key=lambda p:p["start"]):
        op = operational(energy_kwh, period["forecast"])
        total = sci(op, embodied_g)
        rows.append(dict(period, operationalGCO2eApprox=op, embodiedGCO2e=embodied_g, totalGCO2eApprox=total, sciGCO2ePerRequestApprox=total))
    best = min(range(6), key=lambda i: rows[i]["totalGCO2eApprox"])
    current, chosen = rows[0]["totalGCO2eApprox"], rows[best]["totalGCO2eApprox"]
    return dict(periods=rows, recommendation=dict(start=rows[best]["start"], end=rows[best]["end"], runNow=best==0, currentGCO2eApprox=current, bestGCO2eApprox=chosen, reductionGCO2eApprox=current-chosen, reductionPercent=100*(current-chosen)/current if current else 0))


def estimate(input_tokens, output_tokens, *, now=None, energy_fn=estimate_energy, forecast_fn=fetch_forecast, target_model=None):
    """Preserve energy and embodied estimates when the external forecast fails."""
    result = energy_fn(input_tokens, output_tokens) if target_model is None else energy_fn(input_tokens, output_tokens, target_model=target_model)
    result.update(limitations=list(LIMITATIONS), periods=[], recommendation=None, forecastError=None)
    if result.get("scenario", {}).get("model") == "gpt-6-sol":
        result["limitations"][0] = "EcoLogits GPT-6 Sol architecture assumptions; reference H100 infrastructure."
        result["limitations"][2] = "Latency uses the GPT catalog throughput and time-to-first-token assumptions."
        result["limitations"] += [
            "Output length is transferred from a Qwen-trained predictor and is unvalidated for GPT.",
            "Only visible output is estimated; additional hidden reasoning tokens are not included.",
            "Displayed values use the midpoint of the EcoLogits architecture range; this is not a statistical confidence interval.",
        ]
    try:
        result.update(compare_periods(result["energyKWh"], result["embodiedGCO2e"], forecast_fn(now)))
    except ForecastError as exc:
        result["forecastError"] = str(exc)
    return result
