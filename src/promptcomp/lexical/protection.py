"""Conservative occurrence guards for the experimental lexical pilot.

These guards exclude known risky regions. They do not establish that a synonym
preserves intent, and deliberately sacrifice coverage when syntax is uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from ..protect import extract_negations, extract_numbers, extract_protected_terms
from ..types import BlockKind, Span


GUARD_VERSION = "lexical-guards-2"
MODALS = frozenset({"shall", "must", "may", "will", "would", "should", "can", "could", "might", "cannot"})
SCOPE_WORDS = frozenset({
    "all", "each", "any", "every", "only", "solely", "exclusively", "both",
    "either", "neither", "some", "few", "many", "most", "exactly", "unless",
    "except", "if", "otherwise", "until", "whenever", "whether",
})
UNITS = frozenset({
    "mm", "cm", "m", "km", "mg", "g", "kg", "ml", "l", "hz", "khz", "mhz",
    "ghz", "kb", "mb", "gb", "tb", "ms", "sec", "min", "mph", "kph", "psi",
    "meter", "meters", "metre", "metres", "kilometer", "kilometers", "kilometre",
    "kilometres", "millimeter", "millimeters", "centimeter", "centimeters",
    "gram", "grams", "kilogram", "kilograms", "milligram", "milligrams",
    "liter", "liters", "litre", "litres", "milliliter", "milliliters",
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "byte", "bytes",
    "kilobyte", "kilobytes", "megabyte", "megabytes", "gigabyte", "gigabytes",
    "percent", "percentage", "celsius", "fahrenheit", "kelvin", "inch", "inches",
    "foot", "feet", "yard", "yards", "mile", "miles", "pound", "pounds",
    "ounce", "ounces", "dollar", "dollars", "euro", "euros",
})
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_INLINE_RE = re.compile(
    r"https?://[^\s<>\"`]+|www\.[^\s<>\"`]+"
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
    r"|(?<!\w)(?:[A-Za-z]:[\\/]|\.{0,2}/|~/)[^\s<>\"'`]+"
    r"|\b[\w.-]+(?:/[\w./-]+)+"
    r"|<[^>\n]+>"
)
_STRUCTURED_LINE_RE = re.compile(
    r"^(?:[ \t]{2,}\S|\s*(?:-\s+)?[A-Za-z_][\w.-]*\s*[:=]|\s*<[/!A-Za-z]"
    r"|\s*(?:def|class|import|from|return|const|let|var|function)\s+)"
)
_MARKUP_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z][\w:.-]*)\b"
    r"(?:[^<>\"']|\"[^\"]*\"|'[^']*')*>"
)


@dataclass(frozen=True)
class FrozenRange:
    span: Span
    reason: str


def overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def normalize_protected_spans(text: str, spans) -> tuple[Span, ...]:
    """Accept Span or (start, end), rejecting ambiguous/invalid coordinates."""
    normalized = []
    for value in spans:
        if isinstance(value, Span):
            start, end = value.start, value.end
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            start, end = value
        else:
            raise ValueError("protected_spans entries must be Span or (start, end)")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("protected_spans require integer offsets 0 <= start < end <= len(text)")
        normalized.append(Span(start, end))
    return tuple(sorted(set(normalized), key=lambda span: (span.start, span.end)))


def frozen_ranges(text: str, blocks, explicit: tuple[Span, ...], defined_terms) -> tuple[FrozenRange, ...]:
    ranges = [FrozenRange(span, "caller_protected") for span in explicit]
    ranges.extend(FrozenRange(Span(block.start, block.end), "structured_block")
                  for block in blocks if block.kind == BlockKind.PASSTHROUGH)

    # Both Markdown fence forms, including unclosed fences. Existing segment()
    # handles backtick fences; this layer also handles tildes and inline runs.
    fence_start = None
    fence_marker = ""
    fence_size = 0
    blockquote_start = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if _STRUCTURED_LINE_RE.match(line):
            ranges.append(FrozenRange(Span(offset, offset + len(line)), "structured_line"))
        match = re.match(r"\s*(`{3,}|~{3,})", line)
        if match and fence_start is None:
            fence_start, fence_marker, fence_size = offset, match[1][0], len(match[1])
        elif match and match[1][0] == fence_marker and len(match[1]) >= fence_size:
            ranges.append(FrozenRange(Span(fence_start, offset + len(line)), "code_fence"))
            fence_start = None
        # Markdown permits unprefixed continuation lines within a blockquote.
        # Keep the complete nonblank run exact, not just its leading > marker.
        if blockquote_start is None and re.match(r" {0,3}>", line):
            blockquote_start = offset
        if blockquote_start is not None and not line.strip():
            ranges.append(FrozenRange(Span(blockquote_start, offset), "quoted_block"))
            blockquote_start = None
        offset += len(line)
    if fence_start is not None:
        ranges.append(FrozenRange(Span(fence_start, len(text)), "code_fence"))
    if blockquote_start is not None:
        ranges.append(FrozenRange(Span(blockquote_start, len(text)), "quoted_block"))

    # Scan delimiters rather than a pair of greedy regexes, so nested JSON and
    # quotes inside it cannot expose the remainder of the object to rewriting.
    stack: list[tuple[str, int]] = []
    pairs = {"{": "}", "[": "]"}
    quote_pairs = {'"': '"', "“": "”", "‘": "’", "'": "'"}
    i = 0
    while i < len(text):
        char = text[i]
        if char == "`":
            j = i + 1
            while j < len(text) and text[j] == "`":
                j += 1
            delimiter = text[i:j]
            close = text.find(delimiter, j)
            end = len(text) if close == -1 else close + len(delimiter)
            ranges.append(FrozenRange(Span(i, end), "inline_code"))
            i = end
            continue
        # An apostrophe inside a word is not an opening quoted string.
        is_apostrophe = char in {"'", "‘"} and i > 0 and text[i - 1].isalnum()
        if char in quote_pairs and not is_apostrophe:
            closing = quote_pairs[char]
            j = i + 1
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == closing:
                    # Ignore the apostrophe in a quoted contraction.
                    if closing == "'" and j + 1 < len(text) and text[j - 1].isalnum() and text[j + 1].isalnum():
                        j += 1
                        continue
                    break
                j += 1
            end = min(j + 1, len(text))
            ranges.append(FrozenRange(Span(i, end), "quoted_text"))
            i = end
            continue
        if char in pairs:
            stack.append((char, i))
        elif char in pairs.values():
            if stack and pairs[stack[-1][0]] == char:
                _, start = stack.pop()
                ranges.append(FrozenRange(Span(start, i + 1), "structured_literal"))
        i += 1
    ranges.extend(FrozenRange(Span(start, len(text)), "unclosed_literal") for _, start in stack)
    ranges.extend(FrozenRange(Span(match.start(), match.end()), "url_path_or_tag")
                  for match in _INLINE_RE.finditer(text))
    # A non-greedy paired-tag regex exposes text following nested same-name
    # tags. Balance actual tags and freeze unclosed/mismatched outer regions.
    markup_stack: list[tuple[str, int]] = []
    for match in _MARKUP_TAG_RE.finditer(text):
        name = match["name"].casefold()
        if match["closing"]:
            if markup_stack and markup_stack[-1][0] == name:
                _, start = markup_stack.pop()
                ranges.append(FrozenRange(Span(start, match.end()), "structured_markup"))
        elif not match[0].rstrip().endswith("/>"):
            markup_stack.append((name, match.start()))
    ranges.extend(FrozenRange(Span(start, len(text)), "unclosed_markup")
                  for _, start in markup_stack)
    # Comment and CDATA contents can span lines without an ordinary tag pair.
    for pattern in (r"<!--[\s\S]*?(?:-->|$)", r"<!\[CDATA\[[\s\S]*?(?:\]\]>|$)"):
        ranges.extend(FrozenRange(Span(match.start(), match.end()), "structured_markup")
                      for match in re.finditer(pattern, text))
    # Catch shell/printf-style placeholders in addition to the brace/angle forms.
    for match in re.finditer(r"\$[A-Za-z_]\w*|%\([^)]+\)[a-zA-Z]|%[sdif]", text):
        ranges.append(FrozenRange(Span(match.start(), match.end()), "placeholder"))
    for term in sorted(defined_terms):
        if term:
            for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE):
                ranges.append(FrozenRange(Span(match.start(), match.end()), "defined_term"))
    return tuple(sorted(set(ranges), key=lambda item: (item.span.start, item.span.end, item.reason)))


def additional_defined_terms(text: str) -> frozenset[str]:
    """Supplement the legacy capitalized definitions with lowercase terms."""
    terms = set()
    for pattern in (r'["“]([^"”\n]{1,80})["”]\s+(?:means|denotes|refers to)\b',
                    r"(?<!\w)'([^'\n]{1,80})'\s+(?:means|denotes|refers to)\b"):
        terms.update(match[1].strip() for match in re.finditer(pattern, text, re.IGNORECASE))
    return frozenset(terms)


def sentence_scope_reason(sentence, protect_list) -> str | None:
    text = sentence.text
    words = {word.casefold() for word in _WORD_RE.findall(text)}
    if extract_negations(text):
        return "protected_negation_scope"
    if words & MODALS:
        return "protected_modal_scope"
    if words & SCOPE_WORDS or extract_protected_terms(text, protect_list):
        return "protected_condition_or_quantifier_scope"
    # Units and quantitative sentences are frozen together: merely retaining
    # the same digits does not preserve their relationship to a changed word.
    if extract_numbers(text) or any(token.pos_ == "NUM" for token in sentence):
        return "protected_numeric_scope"
    if words & UNITS:
        return "protected_unit_scope"
    return None


def replacement_protection_reason(replacement: str, protect_list, defined_terms) -> str | None:
    words = {word.casefold() for word in _WORD_RE.findall(replacement)}
    if extract_numbers(replacement) or extract_negations(replacement):
        return "introduced_protected_content"
    if words & (MODALS | SCOPE_WORDS | UNITS) or extract_protected_terms(replacement, protect_list):
        return "introduced_protected_content"
    lower = replacement.casefold()
    if any(re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", lower)
           for term in defined_terms if term):
        return "introduced_defined_term"
    return None
