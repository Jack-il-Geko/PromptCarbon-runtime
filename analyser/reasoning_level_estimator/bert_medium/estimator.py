"""Estimate low, medium or high reasoning with the local BERT-Medium model.

This model is separate from the Task Classifier. Long prompts keep their start
and end, as during training. The result is not an energy measurement.
"""

from pathlib import Path
import hashlib
import json
import time

DIRECTORY = Path(__file__).resolve().parent
LEVELS = ("low", "medium", "high")
MAX_LENGTH = 512
CONTENT_HALF = 255


class ReasoningLevelEstimator:
    """Run BERT-Medium and choose one of three reasoning levels."""

    model_id = "google/bert_uncased_L-8_H-512_A-8"

    def __init__(self, model_directory=None, device="cpu", threads=4):
        """Check the saved model files and load BERT on the chosen device."""
        import torch
        from transformers import BertTokenizerFast, AutoModelForSequenceClassification

        self.torch = torch
        self.device = torch.device(device)
        torch.set_num_threads(threads)
        self.model_directory = Path(model_directory or DIRECTORY / "model").resolve()
        provenance = json.loads((DIRECTORY / "info_model.json").read_text())

        # Check that the saved model files have not changed.
        for name, expected in provenance["checkpoint_files_sha256"].items():
            actual = hashlib.sha256(
                (self.model_directory / name).read_bytes()
            ).hexdigest()

            if actual != expected:
                raise ValueError(
                    f"Reasoning Level checkpoint integrity check failed: {name}"
                )

        # Load the tokenizer and trained model from local files.
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.model_directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_directory,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        # Check the model size and saved class order.
        config = self.model.config
        expected_labels = {
            int(index): label for index, label in provenance["id2label"].items()
        }

        if (
            config.id2label != expected_labels
            or config.num_hidden_layers != 8
            or config.hidden_size != 512
            or config.num_attention_heads != 8
            or config.max_position_embeddings != MAX_LENGTH
        ):
            raise ValueError(
                "Unexpected Reasoning Level model configuration or category mapping."
            )

        self.parameters = sum(p.numel() for p in self.model.parameters())

    def encode(self, prompt):
        """Prepare up to 512 tokens, keeping the start and end of long prompts."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty English prompt.")

        raw = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        content = (
            raw
            if len(raw) <= MAX_LENGTH - 2
            else raw[:CONTENT_HALF] + raw[-CONTENT_HALF:]
        )
        encoded = self.tokenizer.prepare_for_model(
            content, add_special_tokens=True, return_attention_mask=True
        )

        return encoded, len(raw) + 2

    def inspect(self, prompt):
        """Return the level, class scores, token counts and time for local checks."""
        started = time.perf_counter()
        encoded, tokens = self.encode(prompt)

        # Run BERT without calculating training gradients.
        with self.torch.inference_mode():
            logits = self.model(
                **{
                    key: self.torch.tensor([value], device=self.device)
                    for key, value in encoded.items()
                }
            ).logits[0]
            scores = self.torch.softmax(logits, dim=-1).cpu().tolist()

        index = max(range(3), key=lambda i: scores[i])

        return {
            "result": {"reasoningLevel": LEVELS[index]},
            "scores": dict(zip(LEVELS, scores)),
            "input_tokens": tokens,
            "used_tokens": len(encoded["input_ids"]),
            "truncated": tokens > MAX_LENGTH,
            "seconds": time.perf_counter() - started,
        }

    def estimate(self, prompt):
        """Return only the reasoning level for the public JSON response."""

        return self.inspect(prompt)["result"]
