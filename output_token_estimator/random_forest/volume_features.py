"""Find five simple cues about source records and requested output detail.

Zero means unknown or not detected. If several data collections are ambiguous,
do not assume the largest one is the requested source.
"""

import csv
import json
import re

NAMES = [
    "source_record_count",
    "per_record_output",
    "requested_result_count",
    "concise_requested",
    "elaborate_requested",
]
INSTRUCTION = r"(?:requested output|output contract|output requirements|response requirements|final instructions|your task|task)"
MATERIAL = r"(?:material|source(?: document)?(?: \(verbatim\))?|input(?: data)?|data|records|complete independent input records|supplied reference implementation[^\n:]*)"
HEADING = re.compile(r"^\s*(" + INSTRUCTION + "|" + MATERIAL + r")\s*:\s*$", re.I)
INLINE_HEADING = re.compile(
    r"^\s*(" + INSTRUCTION + "|" + MATERIAL + r")\s*:\s*(.*)$", re.I
)
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|‘[^’]*’')
FENCE = re.compile(r"```[^\n]*\n[\s\S]*?(?:```|\Z)")
NOUN = (
    r"(?:records?|rows?|cases?|entr(?:y|ies)|messages?|requests?|items?|submissions?)"
)
BULK = re.compile(
    r"\b(?:each|every|all)\s+(?:(?:input|supplied|provided|individual|independent|actual|current|past|selected|top-level)\s+){0,2}"
    + NOUN
    + r"\b|\bper[- ](?:record|row|case|item)\b",
    re.I,
)
SINGLE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|single|one)\s+"
    + NOUN
    + r"\b|\b(?:for|on|about)\s+(?:the\s+)?(?:record|row|case|entry|item)\s+(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9][\w-]*\b",
    re.I,
)


def sections(prompt):
    """Split source material from output instructions using marked headings."""
    lines = prompt.splitlines()
    standalone = any(
        HEADING.match(line) and re.fullmatch(INSTRUCTION, HEADING.match(line)[1], re.I)
        for line in lines
    )
    source = []
    instructions = []
    mode = "source" if standalone else "instruction"
    fenced = False

    for line in lines:
        if line.lstrip().startswith("```"):
            (source if mode == "source" else instructions).append(line)
            fenced = not fenced
            continue

        match = (
            None
            if fenced
            else (HEADING.match(line) if standalone else INLINE_HEADING.match(line))
        )

        if match:
            mode = (
                "instruction" if re.fullmatch(INSTRUCTION, match[1], re.I) else "source"
            )

            if mode == "source":
                source.append(line)
            elif not standalone and match.lastindex == 2 and match[2]:
                instructions.append(match[2])

            continue

        (source if mode == "source" else instructions).append(line)

    # Look for source data in the full text when headings are missing.
    # Hide quoted text when looking for output instructions.
    body = "\n".join(source) if source else prompt
    scope = QUOTED.sub(" ", FENCE.sub(" ", "\n".join(instructions)))

    return body, scope


def negated(text, start):
    """Check whether a nearby instruction says not to perform the operation."""
    clause = re.split(r"[.!?;\n]|\b(?:but|instead)\b", text[:start], flags=re.I)[-1]

    return bool(
        re.search(
            r"\b(?:do not|does not|don't|must not|never|avoid|without|no)\b(?:\s+[\w'-]+){0,8}\s*$",
            clause,
            re.I,
        )
    )


def label_before(text, start):
    """Read a short heading near a possible data collection."""
    before = text[:start]
    line = before.rsplit("\n", 1)[-1].strip()

    if not line:
        line = next(
            (
                x.strip()
                for x in reversed(before.splitlines())
                if x.strip() and not x.strip().startswith("```")
            ),
            "",
        )

    return line.lower()[-140:]


def json_collections(body):
    """Find record collections in JSON and count outer records, not nested lists."""
    decoder = json.JSONDecoder()
    candidates = []
    covered = -1
    flat = []

    def visit(obj, label):
        """Find outer lists of records and labelled nested collections."""

        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            candidates.append((len(obj), label))

            return

        if isinstance(obj, dict):
            nested = [(k, v) for k, v in obj.items() if isinstance(v, (dict, list))]

            if nested:
                for key, value in nested:
                    visit(value, label + " " + key.lower())
            else:
                flat.append(label)

    for match in re.finditer(r"[\[{]", body):
        if match.start() < covered:
            continue

        try:
            obj, end = decoder.raw_decode(body[match.start() :])
        except ValueError:
            continue

        covered = match.start() + end
        visit(obj, label_before(body, match.start()))

    # Count adjacent flat JSON objects as one JSONL collection.
    if flat:
        lines = body.splitlines()
        run = []

        def flush():
            """Save the count of a completed run of flat JSON records."""

            if run:
                candidates.append((len(run), run[0]))

        for line in lines + [""]:
            try:
                obj = json.loads(line)
            except ValueError:
                obj = None

            if (
                isinstance(obj, dict)
                and obj
                and all(not isinstance(v, (dict, list)) for v in obj.values())
            ):
                run.append(label_before(body, body.find(line)))
            else:
                flush()
                run = []

    return candidates


def table_collections(body):
    """Find record counts in CSV, TSV and Markdown-style tables."""
    candidates = []

    for sep in [",", ";", "\t", "|"]:
        block = []
        label = ""

        def flush():
            """Check the current table block and save its record count."""

            if len(block) < 2:
                return

            head = block[0]

            if not 2 <= len(head) <= 30 or not all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ /-]{0,35}", x.strip()) for x in head
            ):
                return

            values = [
                r
                for r in block[1:]
                if not all(re.fullmatch(r":?-{2,}:?", x.strip()) for x in r)
            ]

            if values and all(len(r) == len(head) for r in values):
                candidates.append((len(values), label))

        offset = 0

        for line in body.splitlines() + [""]:
            raw = line.strip()
            clean = raw.strip("|") if sep == "|" else raw

            if sep not in clean or not clean or raw.startswith("```"):
                flush()
                block = []
                offset += len(line) + 1
                continue

            try:
                row = next(csv.reader([clean], delimiter=sep))
            except csv.Error:
                row = []

            if block and len(row) != len(block[0]):
                flush()
                block = []

            if not block:
                label = label_before(body, max(0, offset - 1))

            block.append(row)
            offset += len(line) + 1

        flush()

    return candidates


def select_count(candidates, scope):
    """Choose a clearly identified source collection or return zero if unclear."""
    actual = [
        (n, label)
        for n, label in candidates
        if not re.search(r"\b(?:example|schema|template|sample format)\b", label)
    ]

    if not actual:
        return 0

    if len(actual) == 1:
        return actual[0][0]

    # Use simple collection names as hints when several sources are present.
    selectors = set(re.findall(r"\b(?:actual|current|past|selected)\b", scope, re.I))

    if not selectors and any(re.search(r"\bactual\b", label) for _, label in actual):
        selectors = {"actual"}

    chosen = [
        n
        for n, label in actual
        if any(re.search(r"\b" + word.lower() + r"\b", label) for word in selectors)
    ]

    return chosen[0] if len(chosen) == 1 else 0


def extract(prompt):
    """Return record counts, per-record output cues and short or detailed style cues."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Nonempty prompt required")

    body, scope = sections(prompt)
    count = select_count(json_collections(body) + table_collections(body), scope)
    bulk = any(not negated(scope, m.start()) for m in BULK.finditer(scope))
    single = any(not negated(scope, m.start()) for m in SINGLE.finditer(scope))

    def has(pattern):
        """Check for a cue that is not part of a negative instruction."""

        return float(
            any(
                not negated(scope, m.start()) for m in re.finditer(pattern, scope, re.I)
            )
        )

    return dict(
        source_record_count=float(count),
        per_record_output=float(bulk and not single),
        requested_result_count=float(1 if single else count if bulk else 0),
        concise_requested=has(
            r"\b(?:concise|compact|brief|briefly|short answer|one[- ]line)\b"
        ),
        elaborate_requested=has(
            r"\b(?:detailed|in[- ]depth|comprehensive|step[- ]by[- ]step|worked (?:examples?|solutions?|explanations?)|technical review|teaching (?:note|explanation))\b"
        ),
    )
