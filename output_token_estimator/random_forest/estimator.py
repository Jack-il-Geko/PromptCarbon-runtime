"""Estimate visible output tokens for Qwen using the saved Random Forest.

Use input tokens, predicted task, length preferences and simple prompt features.
Reasoning is reported by the server but is not a feature of either forest.
Unit-based estimates describe expected visible tokens, not guaranteed compliance.
"""

from pathlib import Path
import hashlib
import json
import math
import joblib
import pandas as pd
from output_token_estimator.random_forest.request_features import (
    extract_request_features,
)
from output_token_estimator.random_forest.volume_features import (
    extract as extract_volume_features,
)
from output_token_estimator.random_forest.output_format_features import (
    requested_output_format,
)
from analyser.constraint_extractor.manual.length_preference import (
    normalize_length_preference,
)

DIRECTORY = Path(__file__).resolve().parent
QWEN = "Qwen/Qwen2.5-3B-Instruct"
# Keep the saved model feature columns; only words can be a new preference.
FEATURE_UNITS = ("words", "sentences", "paragraphs")


def render_request(prompt, preference):
    """Add the same length instruction used for dataset generation to a copy of the prompt."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty prompt.")

    p = normalize_length_preference(preference)

    if p is None:
        return prompt

    kind, unit, value = p["type"], p["unit"], p["value"]

    if kind == "range":
        instruction = f"Use between {value[0]} and {value[1]} {unit}, inclusive."
    else:
        modifier = {
            "maximum": "at most",
            "minimum": "at least",
            "exact": "exactly",
            "about": "about",
        }[kind]
        instruction = f"Use {modifier} {value} {unit}."

    return (
        prompt
        + "\n\nOutput length preference (applies to the entire visible response): "
        + instruction
    )


def feature_record(input_tokens, task, reasoning, preference):
    """Turn input counts, task, reasoning and length preference into model features."""

    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
    ):
        raise ValueError("Invalid input token count.")

    if reasoning not in ("low", "medium", "high"):
        raise ValueError("Invalid reasoning level.")

    p = normalize_length_preference(preference)
    r = dict(
        input_tokens=float(input_tokens),
        predicted_task=task,
        predicted_family=task.split(":", 1)[0],
        reasoning_level=reasoning,
        has_length_preference=float(p is not None),
        preference_type=p["type"] if p else "none",
        preference_unit=p["unit"] if p else "none",
    )

    for unit in FEATURE_UNITS:
        for bound in ("lower", "upper", "target"):
            r[f"{unit}_{bound}"] = 0.0

    if p:
        kind, value, unit = p["type"], p["value"], p["unit"]

        if kind == "range":
            r[f"{unit}_lower"], r[f"{unit}_upper"] = map(float, value)
        else:
            bounds = {
                "maximum": ["upper"],
                "minimum": ["lower"],
                "exact": ["lower", "upper", "target"],
                "about": ["target"],
            }[kind]

            for bound in bounds:
                r[f"{unit}_{bound}"] = float(value)

    return r


def preference_scale(preference):
    """Return the requested quantity used by the constraint forest."""

    p = normalize_length_preference(preference)

    if p is None:
        return 1.0

    if p["type"] == "range":
        return (p["value"][0] + p["value"][1]) / 2.0

    return float(p["value"])


class OutputTokenEstimator:
    """Load the saved forest and estimate the length of a visible response."""

    def __init__(self):
        """Check required file hashes and load the saved forest and task categories."""
        self.manifest = json.loads((DIRECTORY / "info_model.json").read_text())
        project = DIRECTORY.parent.parent

        for relative, expected in self.manifest["required_files_sha256"].items():
            if (
                hashlib.sha256((project / relative).read_bytes()).hexdigest()
                != expected
            ):
                raise RuntimeError(f"Output estimator dependency changed: {relative}")

        models = joblib.load(DIRECTORY / "model/model.joblib")
        self.model = models["general"]
        self.constraint_model = models["constrained"]
        self.tasks = {
            t["task"]
            for t in json.loads(
                (project / "analyser/task_classifier/taxonomy.json").read_text()
            )["tasks"]
        }

    def predict(
        self,
        input_tokens,
        task,
        reasoning_level,
        length_preference=None,
        target_model=QWEN,
        *,
        prompt=None,
    ):
        """Estimate visible tokens; maximum limits also check expected content length."""

        if target_model != QWEN:
            raise ValueError(
                "Output estimation is available only for Qwen2.5-3B-Instruct BF16."
            )

        if task not in self.tasks:
            raise ValueError("Unknown task category.")

        # Combine analyser results with simple features from the original prompt.
        row = feature_record(input_tokens, task, reasoning_level, length_preference)
        row.update(extract_request_features(prompt))
        row.update(extract_volume_features(prompt))
        row["requested_output_format"] = requested_output_format(prompt)
        # Both forests ignore reasoning. Maximum constraints also check content length.
        frame = pd.DataFrame([row])

        if length_preference is not None and row["has_length_preference"]:
            # Learn tokens per requested unit from constrained training responses.
            # The rate depends on format, task and request features, not a fixed ratio.
            rate = float(self.constraint_model.predict(frame)[0])
            predicted = rate * preference_scale(length_preference)

            if row["preference_type"] == "maximum":
                # A loose ceiling must not become a length target.
                # Clear structured bounds, keeping the same prompt features and input count.
                content_row = dict(row)
                content_row.update(
                    feature_record(input_tokens, task, reasoning_level, None)
                )
                content = float(self.model.predict(pd.DataFrame([content_row]))[0])
                if not math.isfinite(content):
                    raise RuntimeError("Non-finite content estimate.")
                predicted = min(predicted, content)
        else:
            predicted = float(self.model.predict(frame)[0])

        if not math.isfinite(predicted):
            raise RuntimeError("Non-finite output estimate.")

        return {"outputTokens": max(0, int(math.floor(predicted + 0.5)))}
