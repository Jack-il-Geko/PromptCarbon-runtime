"""Check a length preference entered by the user.

Manual mode accepts word counts and checks the type and quantity without reading the prompt
or loading BERT. No explicit limit returns None.
"""

from typing import Optional

UNITS = {"words"}
TYPES = {"maximum", "exact", "minimum", "range", "about"}
MAX_SAFE_INTEGER = 2**53 - 1  # Keep integers within the browser's exact number range.


def normalize_length_preference(preference) -> Optional[dict]:
    """Check the preference fields and return a clean object or None."""

    if preference is None:
        return None

    if not isinstance(preference, dict):
        raise ValueError("The length preference must be an object or null.")

    kind = preference.get("type")

    if kind == "none" and set(preference) == {"type"}:
        return None

    if set(preference) != {"type", "value", "unit"}:
        raise ValueError("Specify only type, value and unit for a length preference.")

    if not isinstance(kind, str) or kind not in TYPES:
        raise ValueError("Choose maximum, exact, minimum, range or about.")

    unit = preference["unit"]

    if not isinstance(unit, str) or unit not in UNITS:
        raise ValueError(
            "Only words are supported for quantitative length constraints."
        )

    value = preference["value"]

    if kind == "range":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("An interval needs a minimum and a maximum.")

        numbers = value
    else:
        numbers = [value]

    if any(
        type(number) is not int or not 1 <= number <= MAX_SAFE_INTEGER
        for number in numbers
    ):
        raise ValueError(
            "Length quantities must be positive whole numbers within browser precision."
        )

    if kind == "range" and value[0] > value[1]:
        raise ValueError("The minimum cannot exceed the maximum.")

    # Keep an approximate target as given; do not convert it to tokens.
    return {
        "type": kind,
        "value": list(value) if kind == "range" else value,
        "unit": unit,
    }
