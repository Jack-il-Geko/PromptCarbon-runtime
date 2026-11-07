"""Classify a prompt with the trained BERT-Medium model.

Use the saved model from seed 43, epoch 3. Return one task and its confidence.
Apply a narrow check for explicit opening summary instructions.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = Path(__file__).resolve().parent / "model"
CHECKPOINT_MANIFEST = Path(__file__).resolve().parent / "info_model.json"
TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "taxonomy.json"
MODEL_ID = "google/bert_uncased_L-8_H-512_A-8"
MODEL_REVISION = "53b17ea0d90728ac770dfd13fcb3abdc75e4f4bf"
MAX_LENGTH = 256


@dataclass(frozen=True)
class TaskClassification:
    task: str
    confidence: float


def explicit_summary(prompt: str) -> bool:
    """Recognise a direct opening summary request about a document or text."""
    instruction = prompt.lstrip().split("\n\n", 1)[0]

    # Leave competing tasks, negations and replacement instructions to BERT.
    if re.search(
        r"\b(?:do not|don't|never|instead|translate|classify|implement|explain|compare)\b",
        instruction,
        re.IGNORECASE,
    ):
        return False

    # Anchoring the request avoids matching mentions inside source text or code.
    return bool(re.match(
        r"(?:please\s+)?summari[sz]e\s+(?:the\s+|this\s+)?"
        r"(?:(?:following|provided|attached)\s+)?"
        r"(?:report|text|article|document|project update|update|meeting notes|passage)\b",
        instruction,
        re.IGNORECASE,
    ))


def encode_prompt(tokenizer, prompt: str) -> dict:
    """Keep the first 127 and last 127 tokens when the text is longer than 254 tokens."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(ids) > MAX_LENGTH - 2:
        ids = ids[:127] + ids[-127:]

    ids = tokenizer.build_inputs_with_special_tokens(ids)

    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class BertMediumTaskClassifier:
    """Run BERT-Medium and choose one of the 33 task categories."""

    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    seed = 43
    epoch = 3
    learning_rate = 3e-5

    def __init__(self):
        """Check the saved model files and category order, then load the model on CPU."""
        # Check which saved model is being loaded.
        frozen = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))

        if (
            frozen["model_id"],
            frozen["revision"],
            frozen["seed"],
            frozen["epoch"],
        ) != (MODEL_ID, MODEL_REVISION, 43, 3):
            raise RuntimeError(
                "The selected BERT-Medium checkpoint manifest does not match."
            )

        # Check that the model files have not changed.
        for name, expected in frozen["model_files"].items():
            path = MODEL_DIRECTORY / name

            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"BERT-Medium checkpoint checksum mismatch: {name}")

        # Read the 33 task categories in the expected order.
        self.labels = [
            r["task"]
            for r in json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["tasks"]
        ]

        if len(self.labels) != 33 or len(set(self.labels)) != 33:
            raise RuntimeError("The task taxonomy must contain 33 unique categories.")

        # Load the tokenizer and the trained model from local files.
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(MODEL_DIRECTORY), local_files_only=True
        )
        self.model, info = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIRECTORY),
            local_files_only=True,
            attn_implementation="eager",
            output_loading_info=True,
        )

        if any(
            info.get(k)
            for k in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        ):
            raise RuntimeError(
                "The BERT-Medium checkpoint is incomplete or incompatible."
            )

        # Check the model structure and category order.
        config = self.model.config

        if [
            config.id2label[i] for i in range(33)
        ] != self.labels or config.label2id != {
            label: i for i, label in enumerate(self.labels)
        }:
            raise RuntimeError("BERT-Medium label mapping does not match the taxonomy.")

        if config.num_hidden_layers != 8 or config.hidden_size != 512:
            raise RuntimeError("Expected the selected eight-layer BERT-Medium model.")

        # Use the CPU and turn off training behavior.
        self.model.to("cpu").eval()
        torch.set_num_threads(4)

    def classify(self, prompt: str) -> TaskClassification:
        """Run BERT, check explicit summary requests and return the original label score."""

        if not isinstance(prompt, str):
            raise TypeError("The prompt must be a string.")

        if not prompt.strip():
            raise ValueError("The prompt cannot be empty or contain only whitespace.")

        # Prepare the input in the same way as during training.
        batch = self.tokenizer.pad(
            [encode_prompt(self.tokenizer, prompt)],
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Run the model without calculating training gradients.
        with torch.inference_mode():
            logits = self.model(**batch).logits[0]
            prediction = int(torch.argmax(logits))
            if explicit_summary(prompt):
                prediction = self.labels.index("Summarization:Text Summarization")

            # Report the selected label's original score, even after the context check.
            confidence = float(torch.softmax(logits, dim=-1)[prediction])

        return TaskClassification(task=self.labels[prediction], confidence=confidence)
