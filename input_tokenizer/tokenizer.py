"""Count tokens for a single-turn chat without running inference."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import tiktoken
import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from requests import RequestException
from transformers import AutoTokenizer, PreTrainedTokenizerBase


QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
GPT_MODEL_ID = "gpt-6-sol"
OPENAI_TOKEN_COUNT_URL = "https://api.openai.com/v1/responses/input_tokens"
FALLBACK_ENCODING = "o200k_base"
TIKTOKEN_CACHE_DIRECTORY = Path(__file__).resolve().parents[1] / ".venv" / "tiktoken"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIRECTORY))


@dataclass(frozen=True)
class ModelConfig:
    """Associate a target model with its tokenizer configuration."""
    tokenizer_id: str
    add_generation_prompt: bool = True
    backend: Literal["huggingface", "openai"] = "huggingface"


MODELS = {
    QWEN_MODEL_ID: ModelConfig(tokenizer_id=QWEN_MODEL_ID, ),
    GPT_MODEL_ID: ModelConfig(tokenizer_id=GPT_MODEL_ID, backend="openai"),
}


@dataclass(frozen=True)
class TokenCountResult:
    """Describe a token count and whether it is exact or approximate."""

    input_tokens: int
    accuracy: Literal["EXACT", "APPROXIMATE"]
    tokenizer: str


class TokenizerError(RuntimeError):
    """The tokenizer or chat template could not produce a valid count."""


@lru_cache(maxsize=None)
def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load and cache tokenizer files without loading model weights."""
    try:
        bundled = Path(__file__).resolve().parents[1] / "assets" / "qwen_tokenizer"
        if config.tokenizer_id == QWEN_MODEL_ID and bundled.is_dir():
            tokenizer_source = str(bundled)
        else:
            try:
                tokenizer_source = snapshot_download(
                    config.tokenizer_id,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                tokenizer_source = config.tokenizer_id
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False,
        )
    except (OSError, ValueError, RequestException) as exc:
        raise TokenizerError(
            "Unable to load the tokenizer; check the network and local cache."
        ) from exc
    try:
        tokenizer.get_chat_template()
    except (TypeError, ValueError) as exc:
        raise TokenizerError("The official chat template is missing or ambiguous.") from exc
    return tokenizer


@lru_cache(maxsize=1)
def load_fallback_encoding() -> tiktoken.Encoding:
    """Load and cache the tokenizer used for approximate fallback counts."""
    try:
        return tiktoken.get_encoding(FALLBACK_ENCODING)
    except (KeyError, OSError, ValueError, RequestException) as exc:
        raise TokenizerError(
            "Unable to load o200k_base; check the network and local cache."
        ) from exc


def load_openai_api_key() -> str:
    """Read the key only from the project's private .env file."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise TokenizerError("Unable to read the project's .env file.") from exc
    for line in lines:
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            return value.strip().strip("\"'")
    return ""


def count_openai_input(prompt: str, model: str) -> int:
    """Ask OpenAI to count one user message without generating a response."""
    api_key = load_openai_api_key()
    if not api_key:
        raise TokenizerError(
            "GPT-6 Sol requires OPENAI_API_KEY in the project .env file."
        )
    try:
        response = requests.post(
            OPENAI_TOKEN_COUNT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": [{"role": "user", "content": prompt}]},
            timeout=30,
            allow_redirects=False,
        )
    except RequestException as exc:
        raise TokenizerError(
            "Unable to reach OpenAI's token-counting service; check your connection."
        ) from exc
    if response.status_code != 200:
        errors = {
            401: "OpenAI rejected the API key; check whether it is valid or expired.",
            403: "OpenAI denied access; check the key's Responses permissions and model access.",
            429: "OpenAI's request limit or account quota was reached. No local estimate was substituted.",
        }
        raise TokenizerError(errors.get(
            response.status_code,
            f"OpenAI token counting failed (HTTP {response.status_code}).",
        ))
    try:
        payload = response.json()
    except ValueError as exc:
        raise TokenizerError("OpenAI returned an invalid JSON response.") from exc
    count = payload.get("input_tokens") if isinstance(payload, dict) else None
    if type(count) is not int or count <= 0:
        raise TokenizerError("OpenAI did not return a valid input-token count.")
    return count


class InputTokenizer:
    """Use an exact registered tokenizer or an explicit approximate fallback."""

    def __init__(self, target_model: str = QWEN_MODEL_ID) -> None:
        if not isinstance(target_model, str):
            raise TypeError("The model identifier must be a string.")
        if not target_model.strip():
            raise ValueError("The model identifier cannot be empty.")
        self.target_model = target_model
        self.config = MODELS.get(target_model)

    def count_tokens(self, prompt: str) -> int:
        """Return only the integer count for compatibility with existing callers."""
        return self.count(prompt).input_tokens

    def count(self, prompt: str) -> TokenCountResult:
        """Count the prompt and return its accuracy and tokenizer metadata."""
        if not isinstance(prompt, str):
            raise TypeError("The prompt must be a string.")
        if not prompt.strip():
            raise ValueError("The prompt cannot be empty or contain only whitespace.")
        if self.config is None:
            encoding = load_fallback_encoding()
            token_ids = encoding.encode(prompt, disallowed_special=())
            return TokenCountResult(
                input_tokens=len(token_ids),
                accuracy="APPROXIMATE",
                tokenizer=f"tiktoken:{FALLBACK_ENCODING}",
            )

        if self.config.backend == "openai":
            return TokenCountResult(
                input_tokens=count_openai_input(prompt, self.config.tokenizer_id),
                accuracy="EXACT",
                tokenizer=f"openai:{self.config.tokenizer_id}",
            )

        tokenizer = load_tokenizer(self.config)
        try:
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=self.config.add_generation_prompt,
                return_dict=False,
                padding=False,
                truncation=False,
            )
        except (TypeError, ValueError) as exc:
            raise TokenizerError(
                "The chat template is incompatible with the user message."
            ) from exc
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token_id) is not int for token_id in token_ids)
        ):
            raise TokenizerError("The tokenizer did not return a valid list of token IDs.")
        return TokenCountResult(
            input_tokens=len(token_ids),
            accuracy="EXACT",
            tokenizer=self.config.tokenizer_id,
        )
