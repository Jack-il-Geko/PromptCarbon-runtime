"""Local HTTP server for the PromptCarbon interface."""
import errno
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from input_tokenizer.tokenizer import (
    InputTokenizer,
    QWEN_MODEL_ID,
    TokenCountResult,
    TokenizerError,
    load_tokenizer,
)


from analyser.constraint_extractor.manual.length_preference import normalize_length_preference


ROOT = Path(__file__).resolve().parent
QWEN_TOKENIZER = InputTokenizer()
TASK_CLASSIFIER = None
TASK_CLASSIFIER_LOCK = threading.Lock()
CONSTRAINT_EXTRACTOR = None
CONSTRAINT_EXTRACTOR_LOCK = threading.Lock()
REASONING_ESTIMATOR = None
REASONING_ESTIMATOR_LOCK = threading.Lock()
OUTPUT_ESTIMATOR = None
OUTPUT_ESTIMATOR_LOCK = threading.Lock()
TASK_CLASSIFIER_STRATEGY = "bert_medium"


def build_response(result: TokenCountResult) -> dict:
    """Build the public JSON response for a token count."""
    response = {"inputTokens": result.input_tokens}
    if result.accuracy == "APPROXIMATE":
        response.update(
            {
                "accuracy": result.accuracy,
                "tokenizer": result.tokenizer,
            }
        )
    return response


def build_task_response(classification) -> dict:
    """Expose argmax + confidence; uncertainty fields apply only to historical strategies."""
    response = {
        "task": classification.task,
        "confidence": f"{classification.confidence * 100:.2f}%",
    }
    if TASK_CLASSIFIER_STRATEGY not in {"bert_medium", "bert"} and classification.is_uncertain:
        response["taskCategory"] = classification.task.split(":", 1)[0]
    return response


class Handler(BaseHTTPRequestHandler):
    """Serve the web page and token-counting endpoint."""

    def send_json(self, status: int, payload: dict) -> None:
        """Send a JSON response."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        """Serve the application's single web page."""
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return

        data = (ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        """Process the prompt using the selected PromptCarbon component."""
        if self.path not in {"/api/tokenize", "/api/classify-task", "/api/output-length", "/api/extract-constraints", "/api/estimate-reasoning-level", "/api/estimate-output-tokens", "/api/estimate-carbon"}:
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Invalid request size (maximum 2 MB).")

            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("The request must be a JSON object.")

            prompt = payload.get("prompt")
            if self.path == "/api/estimate-carbon":
                from carbon_footprint_estimator.estimator import estimate
                response = public_output_response(estimate_output_tokens(payload))
                response["carbonFootprint"] = estimate(response["inputTokens"], response["outputTokens"])
            elif self.path == "/api/estimate-output-tokens":
                response = public_output_response(estimate_output_tokens(payload))
            elif self.path == "/api/output-length":
                response = {
                    "lengthConstraint": normalize_length_preference(payload.get("lengthConstraint"))
                }
            elif self.path == "/api/extract-constraints":
                response = {"lengthConstraint": word_constraints(
                    get_constraint_extractor().extract(prompt)["lengthConstraint"]
                )}
            elif self.path == "/api/estimate-reasoning-level":
                response = get_reasoning_estimator().estimate(prompt)
            elif self.path == "/api/tokenize":
                target_model = payload.get("targetModel", QWEN_MODEL_ID)
                response = build_response(InputTokenizer(target_model).count(prompt))
            else:
                classification = get_task_classifier().classify(prompt)
                response = build_task_response(classification)
        except (TypeError, ValueError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except TokenizerError as exc:
            self.send_json(503, {"error": str(exc)})
            return
        except (OSError, RuntimeError) as exc:
            self.send_json(503, {"error": f"Component unavailable: {exc}"})
            return

        self.send_json(200, response)


def get_constraint_extractor():
    """Load the standard BERT extractor only for an explicit automatic request."""
    global CONSTRAINT_EXTRACTOR
    if CONSTRAINT_EXTRACTOR is None:
        with CONSTRAINT_EXTRACTOR_LOCK:
            if CONSTRAINT_EXTRACTOR is None:
                from runtime_assets import ensure_assets
                ensure_assets(("constraint",))
                from analyser.constraint_extractor.automatic.extractor import ConstraintExtractor
                CONSTRAINT_EXTRACTOR = ConstraintExtractor()
    return CONSTRAINT_EXTRACTOR


def get_reasoning_estimator():
    """Load the selected, separate Reasoning Level BERT once, on first request."""
    global REASONING_ESTIMATOR
    if REASONING_ESTIMATOR is None:
        with REASONING_ESTIMATOR_LOCK:
            if REASONING_ESTIMATOR is None:
                from analyser.reasoning_level_estimator.bert_medium.estimator import ReasoningLevelEstimator
                REASONING_ESTIMATOR = ReasoningLevelEstimator()
    return REASONING_ESTIMATOR


def get_task_classifier():
    """Load the definitive BERT-Medium classifier once."""
    global TASK_CLASSIFIER
    if TASK_CLASSIFIER is None:
        with TASK_CLASSIFIER_LOCK:
            if TASK_CLASSIFIER is None:
                from analyser.task_classifier.strategia_BERT_Medium.bert_medium import BertMediumTaskClassifier
                TASK_CLASSIFIER = BertMediumTaskClassifier()
    return TASK_CLASSIFIER


def public_output_response(result):
    """Espone solo le caratteristiche richieste, senza prompt e metadati tecnici."""
    fields = ('inputTokens', 'outputTokens', 'task', 'confidence', 'reasoningLevel', 'lengthConstraint')
    return {field: result[field] for field in fields}


def get_output_estimator():
    """Carica una sola RF verificata; non allena e non carica modelli generativi."""
    global OUTPUT_ESTIMATOR
    if OUTPUT_ESTIMATOR is None:
        with OUTPUT_ESTIMATOR_LOCK:
            if OUTPUT_ESTIMATOR is None:
                from output_token_estimator.random_forest.estimator import OutputTokenEstimator
                OUTPUT_ESTIMATOR = OutputTokenEstimator()
    return OUTPUT_ESTIMATOR


def word_constraints(constraints):
    """Keep word limits; other units stay in the prompt without conversion."""
    items = [] if constraints is None else constraints if isinstance(constraints, list) else [constraints]
    words = [item for item in items if item['unit'] == 'words']
    return None if not words else words[0] if len(words) == 1 else words


def automatic_preference(constraints):
    """Encode only global word constraints for the active output estimator."""
    selected = word_constraints(constraints)
    items = [] if selected is None else selected if isinstance(selected, list) else [selected]
    if not items:
        return None, []
    if len(items) == 1:
        item = items[0]
        return normalize_length_preference(dict(type={'max':'maximum','min':'minimum','exact':'exact','approx':'about'}[item['type']], value=item['value'], unit='words')), []
    if len(items) == 2 and {i['type'] for i in items} == {'min','max'}:
        values = {i['type']:i['value'] for i in items}
        if values['min'] <= values['max']:
            return normalize_length_preference(dict(type='range',value=[values['min'],values['max']],unit='words')), []
    return None, ['Automatic word constraints are conflicting or cannot be combined safely.']


def estimate_output_tokens(payload):
    """Esegue i componenti sullo stesso prompt, evitando risultati del browser obsoleti."""
    from output_token_estimator.random_forest.estimator import QWEN, render_request
    if payload.get('targetModel', QWEN) != QWEN:
        raise ValueError('Output estimation is available only for Qwen2.5-3B-Instruct BF16.')
    if TASK_CLASSIFIER_STRATEGY not in {'bert_medium', 'bert'}:
        raise ValueError('This RF requires the active BERT-Medium Task Classifier.')
    prompt = payload.get('prompt')
    render_request(prompt, None)  # Validazione prima di caricare i componenti.
    source = payload.get('constraintSource','manual')
    warnings = []
    if source == 'manual':
        preference = normalize_length_preference(payload.get('lengthConstraint'))
        constraint = preference
        request = render_request(prompt, preference)
    elif source == 'automatic':
        if payload.get('lengthConstraint') is not None:
            raise ValueError('Choose manual or automatic constraints, not both.')
        analysis = get_constraint_extractor().inspect(prompt)
        if analysis['requires_confirmation'] and any(
            span.get('unit') == 'words' and span.get('role_check') == 'ambiguous_scope'
            for span in analysis.get('spans', [])
        ):
            raise ValueError(
                'The scope of an automatic length constraint is unclear. '
                'Use manual mode to confirm the intended output length.'
            )
        constraint = word_constraints(analysis['result']['lengthConstraint'])
        preference, warnings = automatic_preference(constraint)
        # Do not silently estimate without a constraint that could not be encoded.
        if warnings:
            raise ValueError(
                'Automatic length constraints are conflicting or unsupported. '
                'Use manual mode to confirm the intended output length.'
            )
        request = prompt  # I vincoli estratti sono già nel testo: non duplicarli.
        warnings.append('Automatic constraints are suggestions and may be incorrect, especially on long prompts.')
    else:
        raise ValueError('Choose manual or automatic constraint mode.')
    input_tokens = InputTokenizer(QWEN).count(request).input_tokens
    classification = get_task_classifier().classify(prompt)
    reasoning = get_reasoning_estimator().estimate(prompt)['reasoningLevel']
    estimator = get_output_estimator()
    result = estimator.predict(input_tokens, classification.task, reasoning, preference, prompt=prompt)
    manifest = estimator.manifest
    if preference and not manifest['uses_preferences']:
        warnings.append('This RF does not use length preferences directly; their instruction is included in the request token count.')
    if not manifest['uses_reasoning']:
        warnings.append('Reasoning level is reported for downstream use but is not a feature of this RF.')
    warnings.append('Experimental estimate of visible output tokens; not a guarantee or an emissions measurement. Current RF evaluated on reused development data; no independent final confirmation of this version; automatic constraints are not independently validated end-to-end.')
    result.update(inputTokens=input_tokens, **build_task_response(classification),
                  taskFamily=classification.task.split(':',1)[0], reasoningLevel=reasoning,
                  lengthConstraint=constraint, constraintSource=source,
                  targetModel=QWEN, preparedPrompt=request,
                  estimator=dict(variant=manifest['selected'],usesLengthPreference=manifest['uses_preferences'],usesReasoningLevel=manifest['uses_reasoning'],generationPrecision='BF16',temperature=0),
                  warnings=warnings)
    return result


def create_local_server(preferred_port: int = 5500) -> HTTPServer:
    """Prefer port 5500; let the OS choose a free port if it is occupied."""
    try:
        return HTTPServer(("127.0.0.1", preferred_port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"Port {preferred_port} is in use; choosing a free local port.", flush=True)
        return HTTPServer(("127.0.0.1", 0), Handler)


def run() -> None:
    """Load the components, start the server, and open the browser."""
    print("Loading the official Qwen tokenizer...", flush=True)
    try:
        load_tokenizer(QWEN_TOKENIZER.config)
    except TokenizerError as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc

    print("Loading the Task Classifier...", flush=True)
    try:
        classifier = get_task_classifier()
        print(f"Task Classifier: {getattr(classifier, 'model_id', TASK_CLASSIFIER_STRATEGY)}", flush=True)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc

    with create_local_server() as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"PromptCarbon is running at {url}", flush=True)
        browser_timer = threading.Timer(0.5, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            browser_timer.cancel()


if __name__ == "__main__":
    run()
