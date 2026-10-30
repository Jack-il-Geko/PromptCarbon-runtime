"""Find whole-response length constraints with BERT-Medium.

Automatic mode is an alternative to manual entry. It returns only type,
value and unit. BERT finds numeric spans without regex candidate searches.
Simple context rules check Role after BERT; the model weights stay unchanged.
"""

import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast

HERE = Path(__file__).resolve().parent
VALUE_LABELS = ["O", "B-VALUE", "I-VALUE"]
ROLES = ["global", "ignore"]
KINDS = ["exact", "max", "min", "approx", "lt", "gt"]
UNITS = ["words", "tokens", "sentences", "paragraphs", "items"]
HEADS = {"value": VALUE_LABELS, "role": ROLES, "kind": KINDS, "unit": UNITS}
SMALL = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
        range(20),
    )
)
SMALL.update(
    dict(
        zip(
            "twenty thirty forty fifty sixty seventy eighty ninety".split(),
            range(20, 100, 10),
        )
    )
)


class BertConstraintModel(BertPreTrainedModel):
    """Use one BERT encoder with four small output heads."""

    def __init__(self, config):
        """Create BERT and the value, role, kind and unit heads."""
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_size, len(labels))
                for name, labels in HEADS.items()
            }
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Run BERT and return four sets of scores for each token."""
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        hidden = self.dropout(hidden)

        return {name: layer(hidden) for name, layer in self.heads.items()}


def parse_value(text):
    """Convert a numeric span already found by BERT into an integer."""
    text = text.strip().lower()

    if text.startswith("-") or text.endswith("-") or "--" in text:
        raise ValueError("Malformed or signed number")

    if text in ["a hundred", "an hundred"]:
        return 100

    if text in ["a thousand", "an thousand"]:
        return 1000

    chunks = text.split(",")

    if all(x.isdigit() for x in chunks) and (
        len(chunks) == 1
        or 1 <= len(chunks[0]) <= 3
        and all(len(x) == 3 for x in chunks[1:])
    ):
        return int("".join(chunks))

    aliases = {
        "a": 1,
        "an": 1,
        "single": 1,
        "a single": 1,
        "one single": 1,
        "couple": 2,
        "a couple": 2,
        "dozen": 12,
        "a dozen": 12,
        "half a dozen": 6,
    }

    if text in aliases:
        return aliases[text]

    parts = text.replace("-", " ").split()

    def group(words):
        """Read a group of English number words below one thousand."""

        if not words:
            raise ValueError("Empty number")

        value = 0

        if "hundred" in words:
            i = words.index("hundred")

            if i != 1 or words[0] not in SMALL or not 1 <= SMALL[words[0]] <= 9:
                raise ValueError("Invalid hundreds")

            value = SMALL[words[0]] * 100
            words = words[2:]

            if words and words[0] == "and":
                words = words[1:]

                if not words:
                    raise ValueError("Incomplete hundreds")

            if not words:
                return value

        if len(words) == 1 and words[0] in SMALL:
            return value + SMALL[words[0]]

        if (
            len(words) == 2
            and words[0] in SMALL
            and SMALL[words[0]] >= 20
            and words[1] in SMALL
            and 1 <= SMALL[words[1]] <= 9
        ):
            return value + SMALL[words[0]] + SMALL[words[1]]

        raise ValueError("Unsupported numeric span: " + text)

    if "thousand" in parts:
        if parts.count("thousand") != 1:
            raise ValueError("Invalid thousands")

        i = parts.index("thousand")
        lower = parts[i + 1 :]

        if lower and lower[0] == "and":
            lower = lower[1:]

            if not lower:
                raise ValueError("Incomplete thousands")

        return group(parts[:i]) * 1000 + (group(lower) if lower else 0)

    return group(parts)


def normalized_value(text, unit):
    """Clean the edges of a numeric span already found by BERT.

    Allow a matching unit suffix or a trailing 'and' after digits.
    Do not search the prompt for other numbers.
    """

    try:
        return parse_value(text)
    except ValueError:
        pass

    parts = text.strip().lower().split()
    aliases = {unit, unit[:-1]} if unit.endswith("s") else {unit}

    if parts and parts[-1] in aliases:
        return parse_value(" ".join(parts[:-1]))

    if len(parts) == 2 and parts[1] == "and" and parts[0].replace(",", "").isdigit():
        return parse_value(parts[0])

    raise ValueError("Invalid numeric boundaries: " + text)


def public_result(items):
    """Check constraints, remove duplicates and build the public JSON."""
    unique = []

    for item in items:
        if (
            set(item) != {"type", "value", "unit"}
            or item["type"] not in KINDS[:4]
            or item["unit"] not in UNITS
            or type(item["value"]) is not int
            or item["value"] <= 0
        ):
            raise ValueError("Invalid public constraint")

        if item not in unique:
            unique.append(item)

    return {
        "lengthConstraint": (
            None if not unique else unique[0] if len(unique) == 1 else unique
        )
    }


def make_windows(tokenizer, prompt, max_length=512, stride=128):
    """Split long prompts into windows while keeping original text positions.

    Each long window includes up to 64 start tokens and 96 end tokens,
    plus part of the body. Distant instructions may still be misunderstood.
    """
    full = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = full["input_ids"], full["offset_mapping"]

    if len(ids) <= max_length - 2:
        x = tokenizer(prompt, return_offsets_mapping=True)

        return [
            dict(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"],
                token_type_ids=x["token_type_ids"],
                offsets=x["offset_mapping"],
            )
        ]

    prefix = list(range(64)) + list(range(len(ids) - 96, len(ids)))
    capacity = max_length - len(prefix) - 4
    step = capacity - stride

    if step <= 0:
        raise ValueError("Invalid stride")

    windows = []

    for start in range(0, len(ids), step):
        body = list(range(start, min(start + capacity, len(ids))))
        shared = [i for i in prefix if i not in body]
        shared_start = [i for i in shared if i < 64]
        shared_end = [i for i in shared if i >= len(ids) - 96]
        out_ids = [tokenizer.cls_token_id]
        out_offsets = [(0, 0)]
        priorities = [-1]
        segments = [(shared_start, False), (body, True), (shared_end, False)]

        for segment, primary in segments:
            for j, i in enumerate(segment):
                out_ids.append(ids[i])
                out_offsets.append(offsets[i])
                priorities.append(
                    1000 + min(j, len(segment) - 1 - j)
                    if primary
                    else min(j, len(segment) - 1 - j)
                )

            out_ids.append(tokenizer.sep_token_id)
            out_offsets.append((0, 0))
            priorities.append(-1)

        assert len(out_ids) <= max_length
        windows.append(
            dict(
                input_ids=out_ids,
                attention_mask=[1] * len(out_ids),
                token_type_ids=[0] * len(out_ids),
                offsets=out_offsets,
                priorities=priorities,
            )
        )

        if start + capacity >= len(ids):
            break

    return windows


def merge_windows(windows, scores):
    """Keep one prediction per text position, preferring the main body of a window."""
    chosen = {}

    for window, output in zip(windows, scores):
        real = [i for i, pair in enumerate(window["offsets"]) if pair[0] != pair[1]]

        for j, index in enumerate(real):
            offset = tuple(window["offsets"][index])
            margin = (
                window.get("priorities", [])[index]
                if "priorities" in window
                else min(j, len(real) - 1 - j)
            )

            if offset not in chosen or margin > chosen[offset][0]:
                chosen[offset] = (margin, {name: output[name][index] for name in HEADS})

    return [(offset, item[1]) for offset, item in sorted(chosen.items())]


# These rules check BERT spans; they never search for numeric candidates.
SECTION = re.compile(
    r"(?im)^[ \t]*(?:#{1,3}[ \t]+)?(?:"
    r"(?P<source>report|source(?: text| document)?|document|article|passage|"
    r"text to (?:summari[sz]e|translate)|input data|records|material|notes|message|case)|"
    r"(?P<instructions>final instructions|output requirements|response requirements|"
    r"output format|instructions|task))\s*:[ \t]*"
)
COMMAND = re.compile(
    r"^\s*(?:(?:please|finally|also|then|instead)\s*,?\s+)?(?:write|return|respond|answer|reply|provide|produce|give|"
    r"summari[sz]e|explain|compose|generate|use|limit|keep|deliver)\b",
    re.I,
)
QUANTITY_CUE = re.compile(
    r"\b(?:exactly|precisely|at most|no more than|up to|at least|no fewer than|"
    r"about|around|roughly|approximately|under|fewer than|over|more than|less than|"
    r"maximum of|minimum of)\s*$",
    re.I,
)
LOCAL_SCOPE = re.compile(
    r"\b(?:each|every)\s+(?:\w+\s+){0,2}(?:item|record|row|message|paragraph|"
    r"section|sentence|bullet|example|caption|function|entry|report)s?\b|"
    r"\bper[- ](?:item|record|row|message|paragraph|section|sentence|bullet|"
    r"example|caption|function|entry|report)\b",
    re.I,
)
LOCAL_PART = re.compile(
    r"\b(?:introduction|conclusion|title|heading|caption|abstract|docstring|"
    r"section|paragraph|sentence|item|bullet|example|function)s?\b",
    re.I,
)
FIELD_SCOPE = re.compile(
    r"\b(?:field|property|key)\s+(?:[`\"'][A-Za-z_]\w*[`\"']\s+)?"
    r"(?:must|should|shall|may|can|needs? to|has to|is limited to|contains?|uses?)\b|"
    r"\b(?:limit|restrict|keep)\b[^.!?;\n]{0,60}\b(?:field|property|key)\b"
    r"[^.!?;\n]{0,30}\b(?:to|under|within|at)\b",
    re.I,
)
FIELD_TAIL = re.compile(
    r"\b(?:for|in|within)\s+(?:the\s+)?(?:[`\"']?[A-Za-z_]\w*[`\"']?\s+){0,2}"
    r"(?:field|property|key)\b",
    re.I,
)
REVISION = re.compile(
    r"\b(?:ignore|disregard|cancel|drop|remove|forget|waive|lift|replace)\b"
    r"[^.!?\n]{0,100}\b(?:limit|constraint|budget|instruction|requirement|length|word count)s?\b|"
    r"\b(?:limit|constraint|budget|instruction|requirement)s?\b[^.!?\n]{0,80}"
    r"\b(?:no longer|not binding|cancelled|replaced|superseded)\b|"
    r"\b(?:instead|rather than)\b[^.!?\n]{0,80}\b(?:write|use|return|respond|produce)\b"
    r"[^.!?\n]{0,80}\b(?:words?|tokens?|sentences?|paragraphs?|items?)\b|"
    r"\b(?:unrestricted|without (?:a |any )?(?:length|word) limit|"
    r"no longer (?:binding|required|applicable)|no (?:length|word|token) limit)\b",
    re.I,
)


def source_regions(prompt):
    """Find explicitly marked source sections and fenced code, using text offsets."""
    fences = [
        (m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?(?:```|\Z)", prompt)
    ]
    regions = []
    start = None

    for marker in SECTION.finditer(prompt):
        if any(a <= marker.start() < b for a, b in fences):
            continue

        if marker.group("source") is not None:
            if start is None:
                start = marker.end()
        elif start is not None:
            regions.append((start, marker.start()))
            start = None

    if start is not None:
        regions.append((start, len(prompt)))

    return regions + fences


def checked_role(prompt, start, end, role, unit, regions):
    """Check the scope of a numeric span already found by BERT.

    Keep uncertain cases unchanged. Confirm ignored spans only in direct output
    commands with an explicit quantity cue and a matching adjacent unit.
    """
    if any(a <= start < b for a, b in regions):
        return "ignore", "source_or_code"

    left = max((m.end() for m in re.finditer(r"[.!?;\n]", prompt[:start])), default=0)
    boundary = re.search(r"[.!?;\n]", prompt[end:])
    right = end + boundary.start() if boundary else len(prompt)
    before, after = prompt[left:start], prompt[end:right]
    whole_output = re.search(
        r"\b(?:entire|whole)\s+(?:response|answer|reply|output)\b", before + after, re.I
    )

    # A field value is only part of a JSON response, even if BERT says global.
    whole_before = list(
        re.finditer(
            r"\b(?:entire|whole)\s+(?:response|answer|reply|output)\b", before, re.I
        )
    )
    fields_before = list(FIELD_SCOPE.finditer(before))
    local_field = fields_before and (
        not whole_before or fields_before[-1].start() > whole_before[-1].start()
    )
    if local_field or (not whole_before and FIELD_TAIL.search(after)):
        return "ignore", "local_field_scope"

    if LOCAL_SCOPE.search(before) and not whole_output:
        return "ignore", "per_item_scope"

    if re.match(
        r"\s*-?\s*(?:kg|kilograms?|grams?|kilomet(?:er|re)s?|"
        r"minutes?|hours?|days?|weeks?|months?|years?|percent|dollars?|euros?)\b",
        after,
        re.I,
    ):
        return "ignore", "non_length_unit"

    if role == "global":
        return role, "model_role"

    unit_text = re.match(
        r"\s*-?\s*(?:" + re.escape(unit) + "|" + re.escape(unit[:-1]) + r")\b",
        after,
        re.I,
    )

    if not (COMMAND.match(before) and QUANTITY_CUE.search(before) and unit_text):
        return role, "model_role"

    # A local component or a quoted instruction is not a clear whole-answer limit.
    tail = after[unit_text.end() :]
    part = re.match(r"\s+(?:for|in|per)\s+(?:(?:the|each|every|a|an)\s+)?", tail, re.I)
    local_tail = part and re.match(
        LOCAL_PART.pattern + r"|\b(?:record|row|message|report)s?\b",
        tail[part.end() :],
        re.I,
    )
    if not whole_output and (
        LOCAL_PART.search(before)
        or local_tail
        or re.match(r"\s+(?:for|per)\s+(?:each|every)\b", tail, re.I)
    ):
        return role, "ambiguous_scope"

    if any(mark in before for mark in ('"', "'", "“", "‘", "`")):
        return role, "quoted_instruction"

    later = list(prompt[end:])
    for a, b in regions:
        for index in range(max(a, end), b):
            later[index - end] = " "

    if REVISION.search("".join(later)):
        return role, "later_revision"

    return "global", "explicit_output_instruction"


def decode(prompt, predictions):
    """Join numeric tokens and use their average scores to choose the attributes."""
    spans = []
    current = None

    for (start, end), scores in predictions:
        tag = int(scores["value"].argmax())

        if tag == 0:
            if current:
                spans.append(current)
                current = None

            continue

        if tag == 2 and current and not prompt[current["end"] : start].strip():
            current["end"] = end
            current["scores"].append(scores)
        else:
            if current:
                spans.append(current)

            current = dict(start=start, end=end, scores=[scores])

    if current:
        spans.append(current)

    items = []
    details = []
    regions = source_regions(prompt)

    for span in spans:
        attributes = {
            name: labels[
                int(torch.stack([s[name] for s in span["scores"]]).mean(dim=0).argmax())
            ]
            for name, labels in HEADS.items()
            if name != "value"
        }
        text = prompt[span["start"] : span["end"]]
        detail = dict(start=span["start"], end=span["end"], text=text, **attributes)
        role, reason = checked_role(
            prompt,
            span["start"],
            span["end"],
            attributes["role"],
            attributes["unit"],
            regions,
        )
        detail.update(model_role=attributes["role"], role=role, role_check=reason)

        if role != "global":
            detail["status"] = "ignored_" + role
        else:
            try:
                value = normalized_value(text, attributes["unit"])
                kind = attributes["kind"]

                if kind == "lt":
                    kind, value = "max", value - 1
                elif kind == "gt":
                    kind, value = "min", value + 1

                if value <= 0:
                    raise ValueError("Nonpositive integer bound")

                item = dict(type=kind, value=value, unit=attributes["unit"])
                items.append(item)
                detail.update(status="emitted", constraint=item)
            except ValueError as error:
                detail.update(status="unparseable", error=str(error))

        details.append(detail)

    return public_result(items), details


class ConstraintExtractor:
    def __init__(self, model_directory=HERE / "model", device="cpu"):
        """Load the local model and check its structure and head labels."""
        directory = Path(model_directory)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(directory), local_files_only=True
        )
        config = BertConfig.from_pretrained(str(directory), local_files_only=True)

        if (
            config.constraint_heads != HEADS
            or config.hidden_size != 512
            or config.num_hidden_layers != 8
        ):
            raise ValueError("Unexpected model configuration or label mapping")

        self.model = (
            BertConstraintModel.from_pretrained(
                str(directory),
                config=config,
                local_files_only=True,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.max_length = config.constraint_max_length
        self.stride = config.constraint_stride
        torch.set_num_threads(4)

    def inspect(self, prompt):
        """Return the JSON and internal scope details, including unclear cases."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Keep the text positions so repeated windows can be joined later.
        windows = make_windows(self.tokenizer, prompt, self.max_length, self.stride)
        scores = []

        # Run up to eight windows together without training gradients.
        with torch.inference_mode():
            for start in range(0, len(windows), 8):
                subset = windows[start : start + 8]
                batch = self.tokenizer.pad(
                    [
                        {
                            k: w[k]
                            for k in ["input_ids", "attention_mask", "token_type_ids"]
                        }
                        for w in subset
                    ],
                    return_tensors="pt",
                )
                result = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                scores.extend(
                    {
                        name: value[i, : len(w["input_ids"])].cpu()
                        for name, value in result.items()
                    }
                    for i, w in enumerate(subset)
                )

        result, spans = decode(prompt, merge_windows(windows, scores))

        return dict(
            result=result,
            spans=spans,
            windows=len(windows),
            requires_confirmation=any(
                s["role_check"] == "ambiguous_scope" for s in spans
            ),
        )

    def extract(self, prompt):
        """Return only the public lengthConstraint JSON."""

        return self.inspect(prompt)["result"]
