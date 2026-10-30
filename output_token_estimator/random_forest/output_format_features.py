"""Find output artifacts and minimal answers in explicit English instructions."""
import re

HEADINGS = re.compile(
    r"(?im)^[ \t]*(?:#{1,3}[ \t]+)?(?P<title>"
    r"requested output|output requirements|output format|output contract|"
    r"response requirements|final instructions|deliverable to produce|analysis task|instructions|task|"
    r"source(?: document| text)?(?: \(verbatim\))?|report|notes|message|case|announcement|notice|"
    r"document|article|passage|material|input data|records|complete independent input records|"
    r"supplied reference implementation[^\n:]*|text to (?:summari[sz]e|translate))\s*:[ \t]*"
)
INSTRUCTION_HEADING = re.compile(
    r"requested output|output requirements|output format|output contract|"
    r"response requirements|final instructions|deliverable to produce|analysis task|instructions|task", re.I
)
COMMAND = r"return|output|provide|write|respond|reply|produce|give|include|extract|generate|prepare|compose|draft|summari[sz]e|translate|turn|implement|create|build|explain|describe|list|use|keep|make|deliver|supply|solve|state|fix|repair|correct"
CLAUSE = re.compile(r"\b(?:" + COMMAND + r")\b[^\n.!?]*", re.I)
MINIMAL = re.compile(
    r"\b(?:return|output|respond|reply|give|provide|state)\s+(?:only|just)\s+"
    r"(?:(?:the|a|an|one|single|final|required|main|public|that|prescribed)\s+)*"
    r"(?:function[_ ]name|name|label|word|number|value|numerical answer|numeric answer)\b|"
    r"\b(?:return|output|answer|respond|reply)\b[^.!?\n]{0,45}\b(?:exactly|just)\s+(?:one|1)\s+word\b",
    re.I,
)


def instruction_text(prompt):
    """Hide marked source material and code while keeping output instructions."""
    if not isinstance(prompt, str):
        return ""
    text = re.sub(r"```[\s\S]*?(?:```|\Z)", " ", prompt)
    pieces = []
    start = 0
    source = False
    for marker in HEADINGS.finditer(text):
        if not source:
            pieces.append(text[start:marker.start()])
        source = not bool(INSTRUCTION_HEADING.fullmatch(marker['title']))
        start = marker.end()
    if not source:
        pieces.append(text[start:])
    return '\n'.join(pieces)


def output_clauses(prompt):
    """Read commands even after 'read the message and return ...'."""
    text = instruction_text(prompt)
    clauses = []
    for match in CLAUSE.finditer(text):
        before = text[max(0, match.start()-40):match.start()]
        if re.search(r"\b(?:do not|don't|must not|never)\s*$", before, re.I):
            continue
        # Keep the positive part before a trailing negative instruction.
        clause = re.split(r"\b(?:do not|don't|must not|never)\b", match.group(), flags=re.I)[0]
        clauses.append(clause)
    return clauses


def minimal_output_requested(prompt):
    """Recognize an explicit word, label, name or numerical answer, not 'only' alone."""
    return bool(MINIMAL.search('\n'.join(output_clauses(prompt))))


def requested_output_format(prompt):
    """Choose name, JSON, code, table, prose or unspecified without using output length."""
    clauses = output_clauses(prompt)
    # A final explicit output contract takes precedence over earlier tasks.
    for clause in reversed(clauses):
        if not re.match(r"(?:return|output|respond|reply|provide|produce|give|supply|deliver)\b", clause, re.I):
            continue
        if MINIMAL.search(clause):
            return 'name'
        if re.search(r'\bjson\b', clause, re.I):
            return 'json'
        if re.search(r'\b(?:code|implementation|python function|complete function|script|program|module|(?:all|complete|python) source)\b', clause, re.I):
            return 'code'
        if re.search(r'\b(?:table|csv|tabular)\b', clause, re.I):
            return 'table'
        if re.search(r'\b(?:e-?mail(?!\s+address)|letter|summary|explanation|report|review|article|update|briefing|translation|assessment|plan|prose|essay|paragraph)\b', clause, re.I):
            return 'prose'
    scope = '\n'.join(clauses)
    if minimal_output_requested(prompt):
        return 'name'
    if re.search(r'\b(?:implement|write|build|create|generate|refactor|fix|repair|correct)\b[^.!?\n]{0,110}\b(?:functions?|scripts?|programs?|modules?|code|implementation|test harness)\b', scope, re.I):
        return 'code'
    if re.search(r'\bjson\b', scope, re.I):
        return 'json'
    if re.search(r'\b(?:summari[sz]e|translate|explain|describe|essay|paragraph|narrative|guide|review|teaching note|discussion|e-?mail|letter|summary|update|briefing|translation|assessment|plan)\b', scope, re.I):
        return 'prose'
    if re.search(r'\b(?:table|csv|tabular)\b', scope, re.I):
        return 'table'
    return 'unspecified'
