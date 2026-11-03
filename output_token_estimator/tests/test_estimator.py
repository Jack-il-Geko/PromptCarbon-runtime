"""Check the estimator and server connection without generation or test-set scoring."""

import os

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1"
)
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P))
import unittest, json, threading, urllib.request, urllib.error
from unittest.mock import patch
from http.server import HTTPServer
import server
from output_token_estimator.random_forest.estimator import (
    OutputTokenEstimator,
    feature_record,
    render_request,
    QWEN,
)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the forest and start a temporary local HTTP server for the tests."""
        cls.estimator = OutputTokenEstimator()
        cls.http = HTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.http.server_port)

    @classmethod
    def tearDownClass(cls):
        """Stop the local server and wait for its thread to finish."""
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def post(self, payload):
        """Send one JSON request and return its HTTP status and response."""
        request = urllib.request.Request(
            self.url + "/api/estimate-output-tokens",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_pipeline(self):
        """Check the real local pipeline and the six public JSON fields."""
        status, result = self.post(
            dict(
                prompt="Explain in two short sentences why leaves look green.",
                targetModel=QWEN,
            )
        )
        self.assertEqual(status, 200, result)
        self.assertGreater(result["outputTokens"], 0)
        self.assertIn(result["reasoningLevel"], ["low", "medium", "high"])
        self.assertEqual(
            set(result),
            {
                "inputTokens",
                "outputTokens",
                "task",
                "confidence",
                "reasoningLevel",
                "lengthConstraint",
            },
        )
        self.assertIsNone(result["lengthConstraint"])
        self.assertNotIn("preparedPrompt", result)
        self.assertNotIn("warnings", result)
        print("SMOKE", json.dumps(result))

    def test_manual_preference(self):
        """Check that input tokens include the appended manual length instruction."""
        pref = dict(type="maximum", value=30, unit="words")
        prompt = "Explain why ocean water is salty."
        status, r = self.post(
            dict(prompt=prompt, targetModel=QWEN, lengthConstraint=pref)
        )
        self.assertEqual(status, 200, r)
        request = render_request(prompt, pref)
        self.assertEqual(
            r["inputTokens"], server.InputTokenizer(QWEN).count(request).input_tokens
        )
        self.assertEqual(r["lengthConstraint"], pref)
        self.assertNotIn("estimator", r)

    def test_other_models(self):
        """Reject unsupported output models before calling their tokenizers."""

        with patch.object(
            server,
            "InputTokenizer",
            side_effect=AssertionError("Must not call external/tokenizer backend"),
        ):
            for model in ["gpt-5", "other"]:
                status, r = self.post(dict(prompt="Hello", targetModel=model))
                self.assertEqual(status, 400, r)

    def test_invalid_inputs(self):
        """Reject empty prompts, invalid preferences and unknown constraint modes."""

        for payload in [
            dict(prompt=""),
            dict(
                prompt="Hello",
                lengthConstraint=dict(type="exact", value=-1, unit="words"),
            ),
            dict(prompt="Hello", constraintSource="invalid"),
        ]:
            self.assertEqual(self.post(payload)[0], 400)

    def test_constraint_adapter(self):
        """Use word limits and ignore structural units without converting them."""
        p, w = server.automatic_preference(
            [
                dict(type="max", value=100, unit="words"),
                dict(type="min", value=20, unit="words"),
            ]
        )
        self.assertEqual(p, dict(type="range", value=[20, 100], unit="words"))
        self.assertEqual(w, [])

        for unit in ("tokens", "sentences", "paragraphs", "items"):
            self.assertEqual(
                server.automatic_preference(dict(type="exact", value=3, unit=unit)),
                (None, []),
            )

        p, w = server.automatic_preference(
            [
                dict(type="exact", value=80, unit="words"),
                dict(type="max", value=2, unit="sentences"),
            ]
        )
        self.assertEqual(p, dict(type="exact", value=80, unit="words"))
        self.assertEqual(w, [])
        p, w = server.automatic_preference(
            [
                dict(type="max", value=5, unit="words"),
                dict(type="min", value=10, unit="words"),
            ]
        )
        self.assertIsNone(p)
        self.assertTrue(w)

    def test_automatic_pipeline(self):
        """Check the pipeline with the real automatic constraint extractor."""
        prompt = "Summarize the following statement in no more than 50 words: Plants need water and light to grow."
        status, r = self.post(dict(prompt=prompt, constraintSource="automatic"))
        self.assertEqual(status, 200, r)
        self.assertNotIn("preparedPrompt", r)
        self.assertNotIn("constraintSource", r)
        self.assertGreater(r["outputTokens"], 0)
        print("AUTO", json.dumps(r))

    def test_automatic_conflicts(self):
        """Reject unsupported automatic constraints before estimating without them."""
        from types import SimpleNamespace

        cases = (
            [
                dict(type="exact", value=80, unit="words"),
                dict(type="min", value=240, unit="words"),
                dict(type="max", value=6, unit="paragraphs"),
            ],
            [
                dict(type="max", value=5, unit="words"),
                dict(type="min", value=10, unit="words"),
            ],
        )

        for constraints in cases:
            extractor = SimpleNamespace(
                inspect=lambda prompt: {
                    "result": {"lengthConstraint": constraints},
                    "requires_confirmation": False,
                }
            )

            with (
                patch.object(
                    server, "get_constraint_extractor", return_value=extractor
                ),
                patch.object(
                    server,
                    "get_output_estimator",
                    side_effect=AssertionError("Do not ignore the constraints"),
                ),
            ):
                status, result = self.post(
                    dict(
                        prompt="Summarize the supplied report.",
                        constraintSource="automatic",
                    )
                )

            self.assertEqual(status, 400)
            self.assertIn("manual mode", result["error"])

    def test_unclear_scope(self):
        """Ask for manual confirmation when the scope check remains unclear."""
        from types import SimpleNamespace

        extractor = SimpleNamespace(
            inspect=lambda prompt: {
                "result": {"lengthConstraint": None},
                "requires_confirmation": True,
                "spans": [dict(unit="words", role_check="ambiguous_scope")],
            }
        )

        with (
            patch.object(server, "get_constraint_extractor", return_value=extractor),
            patch.object(
                server,
                "get_output_estimator",
                side_effect=AssertionError("Do not estimate unclear constraints"),
            ),
        ):
            status, result = self.post(
                dict(prompt="Write a title and a body.", constraintSource="automatic")
            )

        self.assertEqual(status, 400)
        self.assertIn("manual mode", result["error"])

    def test_role_pipeline(self):
        """Pass the corrected automatic constraint to RF for the report regression."""
        prompt = json.loads((P / "output_token_estimator/tests/test_fixtures.json").read_text())["report_prompt"]
        status, result = self.post(dict(prompt=prompt, constraintSource="automatic"))
        self.assertEqual(status, 200, result)
        preference = dict(type="exact", value=80, unit="words")
        self.assertEqual(result["lengthConstraint"], preference)
        self.assertEqual(result["task"], "Summarization:Text Summarization")
        self.assertEqual(
            result["outputTokens"],
            self.estimator.predict(
                result["inputTokens"],
                result["task"],
                result["reasoningLevel"],
                preference,
                prompt=prompt,
            )["outputTokens"],
        )
        self.assertGreater(result["outputTokens"], 0)

    def test_mixed_modes(self):
        """Reject manual preferences supplied in automatic mode."""
        self.assertEqual(
            self.post(
                dict(
                    prompt="Hello",
                    constraintSource="automatic",
                    lengthConstraint=dict(type="exact", value=7, unit="words"),
                )
            )[0],
            400,
        )

    def test_preference_features(self):
        """Distinguish no preference from an explicit quantity."""
        a = feature_record(50, "Coding:Debugging", "low", None)
        self.assertEqual(a["preference_type"], "none")
        self.assertEqual(a["has_length_preference"], 0)
        a = feature_record(
            50, "Coding:Debugging", "medium", dict(type="exact", value=7, unit="words")
        )
        self.assertEqual(
            [a["words_lower"], a["words_upper"], a["words_target"]], [7, 7, 7]
        )

        with self.assertRaises(ValueError):
            feature_record(True, "Coding:Debugging", "low", None)

    def test_tokenizer_inputs(self):
        """Match Qwen chat inputs against saved technical fixtures."""
        from input_tokenizer.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(server.InputTokenizer(QWEN).config)
        fixtures = json.loads(
            (P / "output_token_estimator/tests/test_fixtures.json").read_text()
        )

        for r in fixtures["tokenizer"]:
            ids = tokenizer.apply_chat_template(
                [dict(role="user", content=r["request"])],
                tokenize=True,
                add_generation_prompt=True,
            )
            self.assertEqual(ids, r["input_token_ids"])

    def test_unknown_task(self):
        """Reject task categories outside the taxonomy."""

        with self.assertRaises(ValueError):
            self.estimator.predict(42, "invalid", "low")

    def test_http_routes(self):
        """Check the page and the existing manual-preference endpoint."""

        with urllib.request.urlopen(self.url) as r:
            html = r.read().decode()
            self.assertIn("outputButton", html)
            self.assertIn("/api/estimate-output-tokens", html)
            self.assertIn("How to set the response limit", html)
            self.assertNotIn('id="lengthUnit"', html)
            self.assertNotIn("lengthUnit.value", html)

        request = urllib.request.Request(
            self.url + "/api/output-length",
            data=b'{"lengthConstraint":null}',
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request) as r:
            self.assertEqual(json.load(r), {"lengthConstraint": None})

    def test_saved_features(self):
        """Match feature values and predictions on saved training-only fixtures."""
        from output_token_estimator.random_forest.request_features import (
            extract_request_features,
        )
        from output_token_estimator.random_forest.volume_features import extract

        self.assertEqual(self.estimator.manifest["training"], 1791)
        fixtures = json.loads(
            (P / "output_token_estimator/tests/test_fixtures.json").read_text()
        )

        for r in fixtures["training_only_fixtures"]:
            self.assertEqual(extract_request_features(r["prompt"]), r["features"])
            self.assertEqual(extract(r["prompt"]), r["volume"])
            self.assertEqual(
                self.estimator.predict(
                    200, "Coding:Debugging", "medium", prompt=r["prompt"]
                )["outputTokens"],
                r["prediction"],
            )

        for r in fixtures["parser_cases"]:
            actual = extract(r["prompt"])

            for k, v in r["expected_fields"].items():
                self.assertEqual(actual[k], v)

    def test_reasoning_invariance(self):
        """Keep the estimate unchanged when only the reasoning level changes."""

        for preference in (None, dict(type="exact", value=80, unit="words")):
            values = [
                self.estimator.predict(
                    200,
                    "Coding:Debugging",
                    level,
                    preference,
                    prompt="Write a Python function that groups records by date.",
                )["outputTokens"]
                for level in ("low", "medium", "high")
            ]
            self.assertEqual(len(set(values)), 1)

    def test_constraint_scaling(self):
        """Let the explicit quantity scale the learned rate for the same request."""

        for unit in ("words",):
            values = [
                self.estimator.predict(
                    200,
                    "Summarization:Text Summarization",
                    "medium",
                    dict(type="exact", value=count, unit=unit),
                    prompt="Return only the summary of the supplied article.",
                )["outputTokens"]
                for count in (10, 20)
            ]
            self.assertGreater(values[0], 0)
            self.assertLessEqual(abs(values[1] - 2 * values[0]), 1)

    def test_word_inputs(self):
        """Reject non-word manual constraints in both the API and forest wrapper."""
        from analyser.constraint_extractor.manual.length_preference import (
            normalize_length_preference,
        )

        for unit in ("sentences", "paragraphs", "tokens", "items"):
            preference = dict(type="exact", value=3, unit=unit)
            with self.assertRaisesRegex(ValueError, "Only words"):
                normalize_length_preference(preference)
            with self.assertRaisesRegex(ValueError, "Only words"):
                self.estimator.predict(
                    100, "Coding:Debugging", "medium", preference, prompt="Write code."
                )
            self.assertEqual(
                self.post(dict(prompt="Write a summary.", lengthConstraint=preference))[
                    0
                ],
                400,
            )

        row = feature_record(
            100,
            "Coding:Debugging",
            "medium",
            dict(type="exact", value=80, unit="words"),
        )
        self.assertEqual(row["words_target"], 80)
        for unit in ("sentences", "paragraphs"):
            self.assertTrue(
                all(
                    row[unit + "_" + bound] == 0
                    for bound in ("lower", "upper", "target")
                )
            )

    def test_word_pipeline(self):
        """Filter other units from both automatic API results without changing the prompt."""
        from types import SimpleNamespace

        structural = [
            dict(type="exact", value=2, unit="sentences"),
            dict(type="max", value=3, unit="paragraphs"),
        ]
        prompt = "Summarize this report in two sentences: The library opened on Monday."

        for constraints, expected in (
            (structural, None),
            (
                [dict(type="max", value=80, unit="words")] + structural,
                dict(type="max", value=80, unit="words"),
            ),
        ):
            extractor = SimpleNamespace(
                extract=lambda prompt: {"lengthConstraint": constraints},
                inspect=lambda prompt: {
                    "result": {"lengthConstraint": constraints},
                    "requires_confirmation": True,
                    "spans": [dict(unit="sentences", role_check="ambiguous_scope")],
                },
            )
            with patch.object(
                server, "get_constraint_extractor", return_value=extractor
            ):
                status, result = self.post(
                    dict(prompt=prompt, constraintSource="automatic")
                )
                self.assertEqual(status, 200, result)
                self.assertEqual(result["lengthConstraint"], expected)
                self.assertEqual(
                    result["inputTokens"],
                    server.InputTokenizer(QWEN).count(prompt).input_tokens,
                )
                req = urllib.request.Request(
                    self.url + "/api/extract-constraints",
                    data=json.dumps(dict(prompt=prompt)).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req) as response:
                    self.assertEqual(
                        json.load(response), {"lengthConstraint": expected}
                    )

    def test_constraint_branch(self):
        """Use the general forest without a constraint and the rate forest with one."""
        from output_token_estimator.random_forest.estimator import preference_scale

        self.assertEqual(preference_scale(None), 1)
        self.assertEqual(
            preference_scale(dict(type="range", value=[20, 40], unit="words")), 30
        )
        self.assertFalse(self.estimator.manifest["uses_reasoning"])

        with (
            patch.object(self.estimator.model, "predict", return_value=[99.0]),
            patch.object(
                self.estimator.constraint_model, "predict", return_value=[2.5]
            ),
        ):
            for preference, expected in (
                (None, 99),
                (dict(type="exact", value=80, unit="words"), 200),
                (dict(type="maximum", value=80, unit="words"), 99),
                (dict(type="minimum", value=80, unit="words"), 200),
                (dict(type="about", value=80, unit="words"), 200),
                (dict(type="range", value=[20, 40], unit="words"), 75),
            ):
                actual = self.estimator.predict(
                    100, "Coding:Debugging", "medium", preference, prompt="Write code."
                )
                self.assertEqual(actual["outputTokens"], expected)

    def test_loose_maximum(self):
        """Keep a loose maximum from inflating a short content estimate."""
        from unittest.mock import patch

        with (
            patch.object(self.estimator.model, "predict", return_value=[3.0]),
            patch.object(
                self.estimator.constraint_model, "predict", return_value=[0.17]
            ),
        ):
            values = [
                self.estimator.predict(
                    100,
                    "Others",
                    "low",
                    dict(type="maximum", value=n, unit="words"),
                    prompt="Return only the word READY.",
                )["outputTokens"]
                for n in (10, 100, 1000)
            ]
        self.assertEqual(values, [2, 3, 3])

        content = self.estimator.predict(
            100, "Others", "low", prompt="Return only the word READY."
        )["outputTokens"]
        for n in (10, 100, 1000):
            bounded = self.estimator.predict(
                100,
                "Others",
                "low",
                dict(type="maximum", value=n, unit="words"),
                prompt="Return only the word READY.",
            )["outputTokens"]
            self.assertLessEqual(bounded, content)

    def test_json_field(self):
        """Keep a word budget for a JSON field separate from the whole response."""
        prompt = (
            "You classify maintenance requests for a public museum. Choose one category: "
            "ACCESS if an authorized person cannot enter; EQUIPMENT if a device is broken; "
            "SUPPLIES if consumable materials are needed; INFORMATION if only guidance is requested. "
            "Return only JSON with fields category and reason. The reason field must contain at most 12 words."
            "\n\nMessage:\nThe label printer works, but we have no replacement paper rolls. "
            "Please order more before Friday. Today we can print visitor labels at the other desk."
        )
        status, result = self.post(dict(prompt=prompt, constraintSource="automatic"))
        self.assertEqual(status, 200, result)
        self.assertIsNone(result["lengthConstraint"])
        self.assertEqual(
            result["outputTokens"],
            self.estimator.predict(
                result["inputTokens"],
                result["task"],
                result["reasoningLevel"],
                None,
                prompt=prompt,
            )["outputTokens"],
        )

    def test_frontend_state(self):
        """Clear old results and discard late responses after any input change."""
        import shutil, subprocess

        node = os.environ.get("PROMPTCARBON_TEST_NODE") or shutil.which("node")
        if not node:
            self.skipTest("Node.js is optional for frontend checks.")
        source = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
const script=html.match(/<script\b[^>]*>([\s\S]*?)<\/script>/)[1];
const nodes={};function element(id){return nodes[id]??=({value:id==='lengthType'?'none':id==='targetModel'?'Qwen/Qwen2.5-3B-Instruct':'',hidden:false,disabled:false,textContent:'',dataset:{},listeners:{},addEventListener(t,f){(this.listeners[t]??=[]).push(f)},setAttribute(k,v){this[k]=v}})}
const ctx=vm.createContext({document:{getElementById:element},performance:{now:()=>0},console});
vm.runInContext(script,ctx);
function change(id,value,event='input'){element(id).value=value;for(const fn of element(id).listeners[event]??[])fn({});}
(async()=>{
 change('userPrompt','First prompt');
 await ctx.runAction('Checking...',async()=>({inputTokens:10}));assert(!element('result').hidden);
 change('userPrompt','Second prompt');assert(element('result').hidden);assert.equal(element('result').textContent,'');
 let resolve;const pending=ctx.runAction('Checking...',()=>new Promise(r=>resolve=r));
 change('lengthType','maximum','change');change('lengthValue','80');resolve({inputTokens:99});await pending;
 assert(element('result').hidden);assert(!element('outputButton').disabled);
 let reject;const failure=ctx.runAction('Checking...',()=>new Promise((r,e)=>reject=e));
 change('targetModel','gpt-5','change');reject(new Error('old request error'));await failure;
 assert(!element('status').textContent.includes('old request error'));
 let oldResolve,newResolve;
 const older=ctx.runAction('Older...',()=>new Promise(r=>oldResolve=r));
 const newer=ctx.runAction('Newer...',()=>new Promise(r=>newResolve=r));
 newResolve({inputTokens:22});await newer;oldResolve({inputTokens:11});await older;
 assert.equal(JSON.parse(element('result').textContent).inputTokens,22);
 change('constraintMode','automatic','change');assert(element('result').hidden);
 console.log('Frontend result state verified.');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        result = subprocess.run(
            [node, "-e", source, str(P / "index.html")],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prose_only(self):
        """Do not treat a complete email or summary as a one-label answer."""
        from output_token_estimator.random_forest.request_features import (
            extract_request_features,
        )
        from output_token_estimator.random_forest.output_format_features import (
            requested_output_format,
        )

        for prompt in (
            "Write around 100 words. Return only the email.",
            "Summarize the report in no more than 160 words. Return only the summary.",
            "Return only the final report, without explanations.",
        ):
            self.assertEqual(requested_output_format(prompt), "prose")
            self.assertEqual(extract_request_features(prompt)["minimal_output_requested"], 0)

        for prompt in (
            "Return only the word READY.",
            "Return only the function name.",
            "Return only valid JSON.",
            "Return only the complete Python function.",
            "Return only the email address.",
        ):
            self.assertEqual(extract_request_features(prompt)["exclusive_output"], 1)

    def test_prose_maximum(self):
        """Apply the same maximum-content policy to prose and minimal answers."""
        with (
            patch.object(self.estimator.model, "predict", return_value=[99.0]),
            patch.object(self.estimator.constraint_model, "predict", return_value=[1.4]),
        ):
            pref = dict(type="maximum", value=160, unit="words")
            result = self.estimator.predict(
                759, "Summarization:Text Summarization", "high", pref,
                prompt="Summarize the report. Return only the summary.",
            )
            self.assertEqual(result["outputTokens"], 99)
            short = self.estimator.predict(
                759, "Others", "low", pref, prompt="Return only the word READY."
            )
            self.assertEqual(short["outputTokens"], 99)

    def test_email_length(self):
        """Check the saved forest on a complete email and a genuine short answer."""
        prompt = (
            "Write an email to my university supervisor asking to move our meeting "
            "from Wednesday afternoon to Friday morning. I have a compulsory "
            "laboratory session on Wednesday that was announced after we arranged "
            "the meeting. Explain that I can send the updated thesis chapter before "
            "our original appointment, so the change will not delay the review. "
            "Be polite and direct, without sounding excessively formal. "
            "Write around 100 words. Return only the email."
        )
        email = self.estimator.predict(111, "Others", "low", prompt=prompt)
        short = self.estimator.predict(
            111, "Others", "low", prompt="Return only the word READY."
        )
        self.assertGreater(email["outputTokens"], short["outputTokens"])
        pref = dict(type="about", value=100, unit="words")
        self.assertGreater(
            self.estimator.predict(111, "Others", "low", pref, prompt=prompt)["outputTokens"],
            short["outputTokens"],
        )

    def test_format_instructions(self):
        """Read output instructions even after a previous sentence ends."""
        from output_token_estimator.random_forest.output_format_features import (
            requested_output_format,
        )

        cases = (
            ("Return exactly one answer. Return only valid JSON.", "json"),
            (
                "Write exactly 80 words in total. Return only the summary, without a title or bullet points.",
                "prose",
            ),
            ("Return only the function name.", "name"),
            ("Return only the complete Python function.", "code"),
            ("Return only a table.", "table"),
        )

        for prompt, expected in cases:
            self.assertEqual(requested_output_format(prompt), expected)

    def test_original_prompt(self):
        """Pass the original prompt to feature extraction before adding preferences."""
        from output_token_estimator.random_forest.request_features import (
            extract_request_features,
        )

        prompt = "Return only JSON with fields status and reason. Classify this supplied message: The gate is open."

        with patch(
            "output_token_estimator.random_forest.estimator.extract_request_features",
            wraps=extract_request_features,
        ) as extractor:
            status, result = self.post(
                dict(
                    prompt=prompt,
                    lengthConstraint=dict(type="maximum", value=40, unit="words"),
                )
            )
            self.assertEqual(status, 200, result)
            extractor.assert_called_once_with(prompt)

    def test_artifact_features(self):
        """Keep complete artifacts separate from explicit minimal answers."""
        from output_token_estimator.random_forest.request_features import extract_request_features
        from output_token_estimator.random_forest.output_format_features import requested_output_format

        for prompt in ("Return only the word READY.", "Return only the function name."):
            self.assertEqual(extract_request_features(prompt)["minimal_output_requested"], 1)
        for prompt in ("Return only valid JSON.", "Return only the complete Python function.", "Return only the translation."):
            self.assertEqual(extract_request_features(prompt)["minimal_output_requested"], 0)
        prompt = "Read this request and return a JSON object with these keys: status, owner, date, risk, reason and action."
        self.assertEqual(extract_request_features(prompt)["requested_keys"], 6)
        self.assertEqual(requested_output_format("Write a program that emits JSON. Return the complete module and tests."), "code")
        self.assertEqual(extract_request_features("Write a summary.\nReport:\nWe visited 17 sites and counted 22 items.")["requested_items"], 0)

    def test_wording_stability(self):
        """Do not collapse a complete translation when only is added."""
        base = "Translate the supplied announcement into English. Return {only}the translation.\nAnnouncement:\nLa biblioteca chiude venerdì alle diciotto."
        values = [self.estimator.predict(200, "Others", "low", prompt=base.format(only=x))["outputTokens"] for x in ("", "only ")]
        self.assertEqual(values[0], values[1])
        self.assertGreater(values[0], self.estimator.predict(200, "Others", "low", prompt="Return only the word READY.")["outputTokens"])

    def test_missing_prompt(self):
        """Reject predictions without the prompt needed for format features."""

        with self.assertRaises(ValueError):
            self.estimator.predict(42, "Coding:Debugging", "low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
