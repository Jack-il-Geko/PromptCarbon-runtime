"""Automatic tests for exact and fallback input-token counting."""
import unittest
from unittest.mock import Mock, patch

import requests

import tiktoken

from server import build_response
from input_tokenizer.tokenizer import (
    FALLBACK_ENCODING,
    GPT_MODEL_ID,
    TokenizerError,
    MODELS,
    InputTokenizer,
    QWEN_MODEL_ID,
    load_openai_api_key,
    load_fallback_encoding,
    load_tokenizer,
)


class InputTokenizerTests(unittest.TestCase):
    """Verify validation, exact counting, fallback behavior, and caching."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODELS[QWEN_MODEL_ID]
        cls.official_qwen = load_tokenizer(cls.config)
        cls.qwen = InputTokenizer(QWEN_MODEL_ID)
        cls.unknown = InputTokenizer("example/unknown-model")

    def test_qwen_template(self):
        official = Mock()
        official.apply_chat_template.return_value = [151644, 42, 151645]
        with patch("input_tokenizer.tokenizer.load_tokenizer", return_value=official) as load, patch(
            "input_tokenizer.tokenizer.load_fallback_encoding"
        ) as fallback:
            result = InputTokenizer(QWEN_MODEL_ID).count("hello")
        load.assert_called_once_with(MODELS[QWEN_MODEL_ID])
        official.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}], tokenize=True,
            add_generation_prompt=True, return_dict=False, padding=False, truncation=False,
        )
        fallback.assert_not_called()
        self.assertEqual(result.accuracy, "EXACT")
        self.assertEqual(result.tokenizer, QWEN_MODEL_ID)
        self.assertEqual(build_response(result), {"inputTokens": 3})

    def test_qwen_unavailable(self):
        with patch("input_tokenizer.tokenizer.load_tokenizer", side_effect=TokenizerError("Unavailable")), patch(
            "input_tokenizer.tokenizer.load_fallback_encoding"
        ) as fallback:
            with self.assertRaises(TokenizerError):
                InputTokenizer(QWEN_MODEL_ID).count("hello")
            fallback.assert_not_called()

    def test_qwen_counts(self) -> None:
        prompts = [
            "Hi",
            "Explain the difference between symmetric and asymmetric encryption.",
            "Spiegami come funziona la crittografia asimmetrica.",
            "Explain this example in detail. " * 1000,
            "Caffè ☕ 😀 <tag>\nαβγ\t你好",
            "  Preserve leading and trailing whitespace.  ",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                expected = len(
                    self.official_qwen.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                        padding=False,
                        truncation=False,
                    )
                )
                result = self.qwen.count(prompt)
                self.assertEqual(result.input_tokens, expected)
                self.assertEqual(result.accuracy, "EXACT")
                self.assertEqual(result.tokenizer, QWEN_MODEL_ID)

    def test_qwen_format(self) -> None:
        rendered = self.official_qwen.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIn("<|im_start|>system\n", rendered)
        self.assertIn("You are Qwen, created by Alibaba Cloud.", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_fallback_counts(self) -> None:
        prompts = [
            "Hi",
            "Explain how asymmetric encryption works.",
            "Spiegami come funziona la crittografia asimmetrica.",
            "Explain this example in detail. " * 1000,
            "Caffè ☕ <|endoftext|> 你好\nNo chat template.",
            "  Preserve leading and trailing whitespace.  ",
        ]
        encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                expected = len(encoding.encode(prompt, disallowed_special=()))
                result = self.unknown.count(prompt)
                self.assertEqual(result.input_tokens, expected)
                self.assertEqual(result.accuracy, "APPROXIMATE")
                self.assertEqual(result.tokenizer, "tiktoken:o200k_base")

    def test_fallback_overhead(self) -> None:
        prompt = "hello"
        raw_count = len(tiktoken.get_encoding(FALLBACK_ENCODING).encode(prompt))
        self.assertEqual(self.unknown.count_tokens(prompt), raw_count)

    def test_response_metadata(self) -> None:
        exact = build_response(self.qwen.count("hello"))
        approximate = build_response(self.unknown.count("hello"))

        self.assertEqual(set(exact), {"inputTokens"})
        self.assertEqual(
            set(approximate), {"inputTokens", "accuracy", "tokenizer"}
        )
        self.assertEqual(approximate["accuracy"], "APPROXIMATE")
        self.assertEqual(approximate["tokenizer"], "tiktoken:o200k_base")

    def test_env_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "environment-key"}), patch(
            "input_tokenizer.tokenizer.Path.read_text",
            return_value="OPENAI_API_KEY=file-key\n",
        ):
            self.assertEqual(load_openai_api_key(), "file-key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "environment-key"}), patch(
            "input_tokenizer.tokenizer.Path.read_text", side_effect=FileNotFoundError
        ):
            self.assertEqual(load_openai_api_key(), "")

    def test_gpt_counts(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"input_tokens": 8}
        with patch("input_tokenizer.tokenizer.load_openai_api_key", return_value="test-key"), patch(
            "input_tokenizer.tokenizer.requests.post", return_value=response
        ) as post:
            result = InputTokenizer(GPT_MODEL_ID).count("Hello!")
        self.assertEqual(result.input_tokens, 8)
        self.assertEqual(result.accuracy, "EXACT")
        self.assertEqual(build_response(result), {"inputTokens": 8})
        self.assertEqual(post.call_args.kwargs["json"], {
            "model": "gpt-5", "input": [{"role": "user", "content": "Hello!"}]
        })
        self.assertEqual(post.call_args.args[0],
                         "https://api.openai.com/v1/responses/input_tokens")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_gpt_key(self) -> None:
        with patch("input_tokenizer.tokenizer.load_openai_api_key", return_value=""), patch(
            "input_tokenizer.tokenizer.requests.post"
        ) as post:
            with self.assertRaisesRegex(TokenizerError, "OPENAI_API_KEY"):
                InputTokenizer(GPT_MODEL_ID).count("Hi")
            post.assert_not_called()

    def test_gpt_validation(self) -> None:
        for payload in ({}, [], {"input_tokens": True}, {"input_tokens": -1},
                        {"input_tokens": 0}, {"input_tokens": "8"}):
            with self.subTest(payload=payload):
                response = Mock(status_code=200)
                response.json.return_value = payload
                with patch("input_tokenizer.tokenizer.load_openai_api_key", return_value="test-key"), patch(
                    "input_tokenizer.tokenizer.requests.post", return_value=response
                ), self.assertRaises(TokenizerError):
                    InputTokenizer(GPT_MODEL_ID).count("Hi")

    def test_gpt_http(self) -> None:
        for status in (400, 401, 403, 429, 500, 302):
            with self.subTest(status=status), patch(
                "input_tokenizer.tokenizer.load_openai_api_key", return_value="test-key"
            ), patch("input_tokenizer.tokenizer.requests.post",
                     return_value=Mock(status_code=status)), self.assertRaises(TokenizerError):
                InputTokenizer(GPT_MODEL_ID).count("Hi")

    def test_gpt_errors(self) -> None:
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("invalid JSON")
        for outcome in (requests.Timeout("timeout"), response):
            with self.subTest(outcome=type(outcome).__name__), patch(
                "input_tokenizer.tokenizer.load_openai_api_key", return_value="test-key"
            ), patch("input_tokenizer.tokenizer.requests.post") as post:
                if isinstance(outcome, Exception):
                    post.side_effect = outcome
                else:
                    post.return_value = outcome
                with self.assertRaises(TokenizerError):
                    InputTokenizer(GPT_MODEL_ID).count("Hi")

    def test_repeatability(self) -> None:
        for tokenizer in (self.qwen, self.unknown):
            first = tokenizer.count("Repeatable prompt")
            second = tokenizer.count("Repeatable prompt")
            self.assertEqual(first, second)

    def test_cache(self) -> None:
        self.assertIs(load_tokenizer(self.config), load_tokenizer(self.config))
        self.assertIs(load_fallback_encoding(), load_fallback_encoding())

    def test_invalid_prompts(self) -> None:
        for prompt in (None, 123, [], {}, "", " \n\t"):
            with self.subTest(prompt=prompt), self.assertRaises((TypeError, ValueError)):
                self.qwen.count(prompt)

    def test_invalid_models(self) -> None:
        for model in (None, 123, "", "   "):
            with self.subTest(model=model), self.assertRaises((TypeError, ValueError)):
                InputTokenizer(model)


if __name__ == "__main__":
    unittest.main()
