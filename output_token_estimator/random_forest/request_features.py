"""Find simple English cues about output format and requested content.

These are features for the saved forest, not a full language parser or a
length-constraint extractor. Quoted material and unusual wording can mislead them.
"""

import re
import json

from output_token_estimator.random_forest.output_format_features import (
    requested_output_format,
    instruction_text,
    output_clauses,
    minimal_output_requested,
)

FENCE = re.compile(r"```[\s\S]*?(?:```|\Z)")
REQUEST = r"\b(?:return|output|respond|reply|produce|provide|format|give|include|supply|generate|compose|prepare|construct|draft|design|develop|present|outline|deliver)\b"
PATTERNS = {
    "json_requested": re.compile(
        REQUEST
        + r"[^\n.!?]{0,90}\bjson\b|\boutput\s+(?:must|should)\s+be\s+(?:valid\s+)?json\b",
        re.I,
    ),
    "exclusive_output": re.compile(
        r"\b(?:return|output|respond|reply|provide|give)\s+only\b|\b(?:json|answer|label|result)\s+only\b|\b(?:without|no)\s+(?:any\s+)?(?:additional\s+)?(?:explanations?|commentary|prose|markdown)\b",
        re.I,
    ),
    "code_artifact": re.compile(
        r"\b(?:implement|write|build|create|return|provide)\b[^\n.!?]{0,90}\b(?:function|program|module|source|script|application|implementation)\b",
        re.I,
    ),
    "tests_requested": re.compile(
        REQUEST
        + r"[^\n.!?]{0,180}\b(?:unit\s*tests?|unittest|test\s+cases|tests|assertions?)\b",
        re.I,
    ),
    "concise_requested": re.compile(
        REQUEST
        + r"[^\n.!?]{0,90}\b(?:brief|short|concise|compact)\b|\bbriefly\s+(?:explain|describe|state|summari[sz]e)\b",
        re.I,
    ),
    "explanation_requested": re.compile(
        r"\b(?:explain|describe|justify)\b|"
        + REQUEST
        + r"[^\n.!?]{0,90}\b(?:explanation|reasoning|limitations|rationale)\b",
        re.I,
    ),
}
SETS = {
    "format4": [
        "json_requested",
        "exclusive_output",
        "code_artifact",
        "tests_requested",
    ],
    "format6": list(PATTERNS),
}
NEGATED = re.compile(
    r"\b(?:do\s+not|don.t|must\s+not|never|not\s+to)\s+(?:\w+\s+){0,2}$", re.I
)


def _format_cues(prompt):
    """Detect format cues outside code fences and skip nearby negative instructions."""
    text = instruction_text(prompt)

    cues = {
        key: float(
            any(
                not NEGATED.search(text[max(0, m.start() - 35) : m.start()])
                for m in pattern.finditer(text)
            )
        )
        for key, pattern in PATTERNS.items()
    }

    return cues


WORDS = {
    word: i
    for i, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
NUMBER = r"(?:[1-9]\d{0,3}|" + "|".join(list(WORDS)[1:]) + r")"
DIRECTIVE = re.compile(
    r"(?:^|[.!?\n])\s*(?:(?:please|then|finally|also)\s+)?(?:(?:can|could|would)\s+you\s+)?(?:return|output|respond|reply|produce|provide|format|give|include|supply|generate|compose|prepare|construct|draft|design|develop|present|outline|deliver|write|build|create|implement|answer|list|summari[sz]e|explain|describe|finish|use|keep|make)\b[^\n.!?]{0,350}",
    re.I,
)
FORMAT = SETS["format4"] + ["table_requested", "list_requested", "prose_requested"]

FEATURE_NAMES = FORMAT + [
    "minimal_output_requested",
    "requested_items",
    "requested_sections",
    "requested_functions",
    "requested_keys",
    "requested_components",
]


def num(text):
    """Convert a supported number word or digit string into an integer."""

    return WORDS[text.lower()] if text.lower() in WORDS else int(text)


def extract_request_features(prompt):
    """Find cues about format, minimal answers and requested content."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty prompt for output estimation.")

    text = instruction_text(prompt)
    scope = text
    # Ignore negative instructions when counting requested content.
    scope = re.sub(
        r"\b(?:without|do not|must not|never)\b[^\n.!?]*", " ", scope, flags=re.I
    )
    previous_cues = _format_cues(prompt)
    result = {k: previous_cues[k] for k in SETS["format4"]}
    result["minimal_output_requested"] = float(minimal_output_requested(prompt))

    for key, pattern in [
        ("table_requested", r"\b(?:table|tabular|csv)\b"),
        ("list_requested", r"\b(?:bullet|numbered|list)\b"),
        (
            "prose_requested",
            r"\b(?:essay|report|paragraphs?|narrative|prose|guide|review|teaching note|discussion|e-?mail|letter|summary|update|briefing|translation|assessment|plan)\b",
        ),
    ]:
        result[key] = float(bool(re.search(pattern, scope, re.I)))

    for key, noun in [
        (
            "requested_items",
            r"(?:items?|examples?|recommendations?|records?|bullet points?|assertions?|tests?|measurements?|criteria|questions?)",
        ),
        ("requested_sections", r"sections?"),
        ("requested_functions", r"functions?"),
    ]:
        values = [
            num(m.group(1))
            for m in re.finditer(
                r"\b(" + NUMBER + r")\s+(?:[A-Za-z]+\s+){0,2}" + noun + r"\b",
                scope,
                re.I,
            )
        ]
        result[key] = float(max(values, default=0))

    keys = []

    for m in re.finditer(r"\b(?:keys|fields|json object (?:containing|with(?!\s+(?:these\s+)?(?:keys|fields))))\s*:?\s+([^\n.!?]{1,500})", scope, re.I):
        tokens = re.split(r"\s*,\s*|\s+and\s+", m.group(1).strip())
        keys.extend(
            k.strip(" `\"'")
            for k in tokens
            if re.fullmatch(r"[ `\"\']*[A-Za-z_][\w-]*[ `\"\']*", k)
        )

    numeric_keys = [
        num(m.group(1))
        for m in re.finditer(
            r"\b(" + NUMBER + r")\s+(?:[A-Za-z]+\s+){0,2}(?:keys|fields)\b", scope, re.I
        )
    ]
    schema_counts = []

    # Count fields only in small objects near a JSON or schema instruction.
    for m in re.finditer(
        r"\b(?:json|keys|fields)\b[^{}]{0,180}(\{[^{}]{0,600}\})",
        text if previous_cues["json_requested"] else "",
        re.I,
    ):
        try:
            obj = json.loads(m.group(1))

            if isinstance(obj, dict):
                schema_counts.append(len(obj))
        except (ValueError, TypeError):
            pass

    result["requested_keys"] = float(
        max([len(set(keys))] + numeric_keys + schema_counts)
    )
    result["requested_components"] = float(
        sum(
            bool(re.search(pattern, scope, re.I))
            for pattern in [
                r"\b(?:source|module|program)\b",
                r"\b(?:tests?|assertions?)\b",
                r"\b(?:worked )?examples?\b",
                r"\blimitations?\b",
                r"\b(?:explanation|rationale|design decisions)\b",
                r"\b(?:complexity|time and space|time or space)\b",
                r"\b(?:table|decision table)\b",
            ]
        )
    )

    return {key: result[key] for key in FEATURE_NAMES}
