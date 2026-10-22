"""Opt-in, auditable lexical suggestions; never a semantic safety guarantee.

The existing compressor does not call this module. A selected edit is an
unreviewed suggestion which passed mechanical guards and saved measured tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import re
from time import perf_counter

from ..parse import load_nlp, parse
from ..protect import (
    extract_defined_terms, extract_negations, extract_numbers,
    extract_protected_terms, load_defined_terms, load_protect_list, veto,
)
from ..segment import segment
from ..tokens import TokenCounter
from ..types import BlockKind, Candidate, Span
from .models import CandidateProvider, LexicalCandidate
from .tokenization import local_counter
from .protection import (
    GUARD_VERSION, MODALS, SCOPE_WORDS, UNITS, additional_defined_terms, frozen_ranges,
    normalize_protected_spans, overlaps, replacement_protection_reason,
    sentence_scope_reason,
)


ENGINE_VERSION = "lexical-pilot-1"
_MORPH_KEYS = ("Number", "Tense", "VerbForm", "Person", "Degree", "Voice")
_LEXICAL_POS = frozenset({"ADJ", "ADV", "NOUN", "VERB"})
_EXPERIMENTAL_WARNING = (
    "Experimental, unreviewed lexical suggestions: token savings and structural "
    "guards do not establish semantic equivalence or grammatical correctness. "
    "Human review is required before using rewritten prompts."
)


@dataclass(frozen=True)
class LexicalEdit:
    span: Span
    original: str
    replacement: str
    provenance: str
    similarity: float | None
    pos: str | None
    sentence_before: str
    sentence_after: str
    tokens_before: int
    tokens_after: int
    token_delta: int


@dataclass(frozen=True)
class LexicalDecision:
    span: Span
    original: str
    replacement: str
    provenance: str
    similarity: float | None
    pos: str | None
    sentence_before: str
    sentence_after: str
    tokens_before: int | None
    tokens_after: int | None
    token_delta: int | None
    decision: str
    reason: str
    candidate_rank: int


@dataclass(frozen=True)
class LexicalResult:
    original: str
    suggested: str
    tokens_before: int
    tokens_after: int
    token_counter: str
    provider: str
    edits: tuple[LexicalEdit, ...]
    decisions: tuple[LexicalDecision, ...]
    warnings: tuple[str, ...]
    config_hash: str
    timings_ms: dict[str, float]
    review_status: str = "pending"


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _count(counter: TokenCounter, text: str) -> int:
    value = counter.count(text)
    if type(value) is not int or value < 0:
        raise ValueError("TokenCounter.count must return a nonnegative integer")
    return value


def _compose(text: str, edits) -> str:
    parts = []
    cursor = 0
    for edit in sorted(edits, key=lambda item: (item.span.start, item.span.end)):
        if edit.span.start < cursor or text[edit.span.start:edit.span.end] != edit.original:
            raise ValueError("Overlapping edits or stale original offsets")
        parts.extend((text[cursor:edit.span.start], edit.replacement))
        cursor = edit.span.end
    parts.append(text[cursor:])
    return "".join(parts)


def _adapt_case(source: str, replacement: str, *, sentence_initial: bool) -> str | None:
    if source.isupper():
        return replacement.upper()
    if source.islower():
        # Capitalized candidate spellings may denote a name. Do not silently
        # convert them into a common noun just to make the edit pass.
        return replacement if replacement.islower() else None
    if sentence_initial and source[0].isupper() and source[1:].islower():
        if not (replacement.islower() or (replacement[0].isupper() and replacement[1:].islower())):
            return None
        return replacement[0].upper() + replacement[1:]
    return None


def _grammar_reason(source_span, sentence, replacement: str, new_sentence: str, nlp_cache) -> str | None:
    if new_sentence not in nlp_cache:
        nlp_cache[new_sentence] = parse(new_sentence)
    edited_doc = nlp_cache[new_sentence]
    start = source_span.start_char - sentence.start_char
    edited_span = edited_doc.char_span(start, start + len(replacement))
    if edited_span is None or len(list(edited_doc.sents)) != 1:
        return "replacement_parse_unaligned"
    if any(token.pos_ in {"PROPN", "NUM"} for token in edited_span):
        return "introduced_entity_or_number"
    if any(ent.start_char < start + len(replacement) and start < ent.end_char
           for ent in edited_doc.ents):
        return "introduced_entity_or_number"
    if len(source_span) == 1:
        source = source_span[0]
        target = edited_span.root
        if source.pos_ not in _LEXICAL_POS or target.pos_ != source.pos_:
            return "pos_mismatch"
        if any(source.morph.get(key) != target.morph.get(key) for key in _MORPH_KEYS):
            return "morphology_mismatch"
        if source.dep_ != target.dep_:
            return "dependency_mismatch"
        # Avoid obvious a/an breakage. Spelling is not pronunciation, so this
        # is an extra conservative rejection rule, not an article validator.
        if source.i > sentence.start:
            previous = source.doc[source.i - 1].text.casefold()
            if previous in {"a", "an"}:
                if (source.text[0].casefold() in "aeiou") != (replacement[0].casefold() in "aeiou"):
                    return "article_context_change"
    else:
        # Phrase rules can legitimately change internal dependencies (e.g.
        # "in order to" -> "to"). Preserve the sentence root outside the edit.
        original_root = sentence.root
        if not source_span.start <= original_root.i < source_span.end:
            new_root = next(iter(edited_doc.sents)).root
            if (new_root.lemma_, new_root.pos_, new_root.morph) != (
                    original_root.lemma_, original_root.pos_, original_root.morph):
                return "sentence_root_changed"
    return None


def _invariants_hold(original, output, edits, frozen, protect_list, defined_terms) -> bool:
    if _compose(original, edits) != output:
        return False
    for item in frozen:
        if any(overlaps(edit.span, item.span) for edit in edits):
            return False
        shift = sum(len(edit.replacement) - len(edit.original)
                    for edit in edits if edit.span.end <= item.span.start)
        if output[item.span.start + shift:item.span.end + shift] != original[item.span.start:item.span.end]:
            return False
    if extract_numbers(original) != extract_numbers(output) or extract_negations(original) != extract_negations(output):
        return False
    if extract_protected_terms(original, protect_list) != extract_protected_terms(output, protect_list):
        return False
    for term in defined_terms:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if len(re.findall(pattern, original, re.IGNORECASE)) != len(re.findall(pattern, output, re.IGNORECASE)):
            return False
    return True


def suggest(
    text: str,
    provider: CandidateProvider,
    *,
    counter: TokenCounter | None = None,
    protected_spans=(),
    max_edits_per_sentence: int = 1,
    min_embedding_similarity: float | None = None,
    adaptive_similarity: bool = False,
) -> LexicalResult:
    """Return unreviewed, token-saving occurrence suggestions and their audit.

    ``protected_spans`` uses original Python character offsets, accepting Span
    objects or (start, end) pairs. This pilot supports exactly one selected edit
    per sentence. The default tokenizer requires locally cached assets and never
    downloads them. Parser and candidate assets must also be available locally.
    An explicitly supplied counter/provider controls its own I/O behavior.
    """
    started = perf_counter()
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if type(max_edits_per_sentence) is not int or max_edits_per_sentence != 1:
        raise ValueError("The lexical pilot supports max_edits_per_sentence=1 only")
    if min_embedding_similarity is not None and (
        isinstance(min_embedding_similarity, bool)
        or not isinstance(min_embedding_similarity, (int, float))
        or not math.isfinite(min_embedding_similarity)
        or not -1.0 <= min_embedding_similarity <= 1.0
    ):
        raise ValueError("min_embedding_similarity must be a finite cosine threshold in [-1, 1]")
    if not isinstance(adaptive_similarity, bool) or (adaptive_similarity and min_embedding_similarity is None):
        raise ValueError("adaptive_similarity requires a minimum embedding similarity")
    explicit = normalize_protected_spans(text, protected_spans)
    max_words = provider.max_phrase_words
    if type(max_words) is not int or not 1 <= max_words <= 16:
        raise ValueError("provider.max_phrase_words must be an integer between 1 and 16")
    counter = local_counter() if counter is None else counter
    original_count = _count(counter, text)
    current_count = original_count
    nlp = load_nlp()
    protect_list = load_protect_list()
    yaml_terms, defined_version = load_defined_terms()
    defined_terms = frozenset(set(yaml_terms) | extract_defined_terms(text) | additional_defined_terms(text))
    blocks = segment(text)
    frozen = frozen_ranges(text, blocks, explicit, defined_terms)
    setup_done = perf_counter()
    edits: list[LexicalEdit] = []
    decisions: list[LexicalDecision] = []
    candidate_identity = []
    parse_cache = {}
    warnings = [_EXPERIMENTAL_WARNING]

    for block in blocks:
        if block.kind != BlockKind.PROSE:
            continue
        doc = parse(block.text)
        for sentence in doc.sents:
            # Sentence-local decisions are finalized only after comparing every
            # eligible occurrence; provider ordering breaks equal-saving ties.
            local_decisions: list[LexicalDecision] = []
            available: list[tuple[int, LexicalEdit]] = []
            scope_reason = sentence_scope_reason(sentence, protect_list)
            first_word = next((token.i for token in sentence if not token.is_space), sentence.start)
            for start_token in range(sentence.start, sentence.end):
                for width in range(1, min(max_words, sentence.end - start_token) + 1):
                    source_span = doc[start_token:start_token + width]
                    if any(not token.is_alpha for token in source_span):
                        break
                    if any(not doc.text[left.idx + len(left):right.idx].isspace()
                           for left, right in zip(source_span, source_span[1:])):
                        break
                    source = source_span.text
                    normalized_source = " ".join(source.split()).casefold()
                    span = Span(block.start + source_span.start_char, block.start + source_span.end_char)
                    source_pos = source_span.root.pos_
                    proposals = provider.candidates(normalized_source, pos=source_pos)
                    for rank, candidate in enumerate(proposals):
                        if not isinstance(candidate, LexicalCandidate):
                            raise TypeError("Candidate providers must return LexicalCandidate values")
                        if candidate.similarity is not None and not math.isfinite(candidate.similarity):
                            raise ValueError("Candidate similarity must be finite or None")
                        candidate_identity.append({"span": asdict(span), "rank": rank, **asdict(candidate)})
                        replacement = candidate.replacement
                        reason = None
                        if " ".join(candidate.source.split()).casefold() != normalized_source:
                            reason = "candidate_source_mismatch"
                        elif not replacement or replacement != " ".join(replacement.split()) or not all(word.isalpha() for word in replacement.split()):
                            reason = "invalid_replacement_format"
                        else:
                            adapted = _adapt_case(source, replacement, sentence_initial=start_token == first_word)
                            if adapted is None:
                                reason = "unsupported_case_or_name"
                            else:
                                replacement = adapted
                        if reason is None and source.casefold() == replacement.casefold():
                            reason = "unchanged"
                        if reason is None:
                            frozen_match = next((item for item in frozen if overlaps(span, item.span)), None)
                            if frozen_match is not None:
                                reason = "protected_" + frozen_match.reason
                        if reason is None:
                            reason = scope_reason
                        if reason is None:
                            existing = Candidate(source_span.root.dep_, source_span.start_char, source_span.end_char,
                                                 source_span.root.text, source, len(source_span))
                            _, vetoes = veto(doc, [existing], protect_list, defined_terms)
                            if vetoes:
                                reason = "protected_" + vetoes[0].protect_class
                        if reason is None and any(token.pos_ == "PROPN" for token in source_span):
                            reason = "protected_proper_noun"
                        if reason is None and candidate.pos is not None and candidate.pos != source_pos:
                            reason = "candidate_pos_mismatch"
                        if reason is None:
                            reason = replacement_protection_reason(replacement, protect_list, defined_terms)
                        start = source_span.start_char - sentence.start_char
                        end = source_span.end_char - sentence.start_char
                        sentence_after = sentence.text[:start] + replacement + sentence.text[end:]
                        trial_count = None
                        delta = None
                        trial = LexicalEdit(span, source, replacement, candidate.provenance, candidate.similarity,
                                            source_pos, sentence.text, sentence_after, current_count, current_count, 0)
                        if reason is None:
                            trial_count = _count(counter, _compose(text, [*edits, trial]))
                            delta = current_count - trial_count
                            if delta <= 0:
                                reason = "no_token_saving"
                        if reason is None and min_embedding_similarity is not None and candidate.provenance.startswith("embedding:"):
                            # A conservative exploratory policy: one-token gains and
                            # noun/verb edits need stronger evidence than larger gains
                            # or adjective/adverb edits. Cosine remains only a proxy.
                            required = float(min_embedding_similarity)
                            if adaptive_similarity:
                                required = min(1.0, required + (0.05 if delta == 1 else 0.0)
                                               + (0.05 if source_pos in {"NOUN", "VERB"} else 0.0))
                            if candidate.similarity is None:
                                reason = "embedding_similarity_unavailable"
                            elif candidate.similarity < required:
                                reason = "embedding_similarity_below_threshold"
                        if reason is None:
                            reason = _grammar_reason(source_span, sentence, replacement, sentence_after, parse_cache)
                        decision = LexicalDecision(
                            span, source, replacement, candidate.provenance, candidate.similarity, source_pos,
                            sentence.text, sentence_after,
                            current_count if trial_count is not None else None, trial_count, delta,
                            "rejected", reason or "sentence_edit_limit", rank,
                        )
                        local_decisions.append(decision)
                        if reason is None:
                            available.append((len(local_decisions) - 1,
                                              replace(trial, tokens_after=trial_count, token_delta=delta)))
            if available:
                index, chosen = min(available, key=lambda item: (
                    -item[1].token_delta, local_decisions[item[0]].candidate_rank,
                    item[1].span.start, item[1].span.end, item[1].replacement, item[1].provenance,
                ))
                edits.append(chosen)
                current_count = chosen.tokens_after
                local_decisions[index] = replace(local_decisions[index], decision="selected", reason="token_saving_unreviewed")
            decisions.extend(local_decisions)

    proposal_done = perf_counter()
    output = _compose(text, edits)
    final_count = _count(counter, output)
    if (final_count != current_count or final_count > original_count
            or (edits and final_count >= original_count)
            or not _invariants_hold(text, output, edits, frozen, protect_list, defined_terms)):
        output, final_count, edits = text, original_count, []
        decisions = [replace(item, decision="rejected", reason="invariant_rollback")
                     if item.decision == "selected" else item for item in decisions]
        warnings.append("All suggestions were rolled back because a final invariant or token recount failed.")
    config = {
        "engine": ENGINE_VERSION, "guards": GUARD_VERSION,
        "provider": provider.name, "provider_metadata": getattr(provider, "metadata", {}),
        "observed_candidates_sha256": _digest(candidate_identity),
        "counter": {"name": counter.name, "type": f"{type(counter).__module__}.{type(counter).__qualname__}",
                    "metadata": getattr(counter, "metadata", {}), "tiktoken_version": _package_version("tiktoken")},
        "parser": {"spacy_version": _package_version("spacy"), "name": nlp.meta.get("name"),
                   "version": nlp.meta.get("version"), "lang": nlp.meta.get("lang"), "pipeline": nlp.pipe_names},
        "protect_list": {"version": protect_list.version, "ner_numeric": sorted(protect_list.ner_numeric),
                         "ner_entity": sorted(protect_list.ner_entity),
                         "single_terms": {k: sorted(v) for k, v in protect_list.single_terms.items()},
                         "phrase_terms": {k: sorted(v) for k, v in protect_list.phrase_terms.items()}},
        "defined_terms_version": defined_version, "defined_terms": sorted(defined_terms),
        "scope_words": sorted(SCOPE_WORDS), "modal_words": sorted(MODALS), "units": sorted(UNITS),
        "protected_spans": [asdict(span) for span in explicit],
        "max_edits_per_sentence": max_edits_per_sentence, "max_phrase_words": max_words,
        "min_embedding_similarity": min_embedding_similarity,
        "adaptive_similarity": adaptive_similarity,
        "adaptive_similarity_policy": "base + 0.05 for one-token gain + 0.05 for noun/verb; capped at 1"
        if adaptive_similarity else None,
        "ranking": "full_text_token_gain,provider_rank,original_start,original_end,replacement,provenance",
    }
    config_hash = _digest(config)
    finished = perf_counter()
    return LexicalResult(
        text, output, original_count, final_count, counter.name, provider.name,
        tuple(edits), tuple(decisions), tuple(warnings), config_hash,
        {"setup": (setup_done - started) * 1000, "proposals": (proposal_done - setup_done) * 1000,
         "finalize": (finished - proposal_done) * 1000, "total": (finished - started) * 1000},
    )
