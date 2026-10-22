"""Mechanical invariants for unreviewed lexical suggestions, not semantic evals."""
from dataclasses import asdict

import pytest

from promptcomp.lexical.engine import suggest
from promptcomp.lexical.models import LexicalCandidate
from promptcomp.tokens import TiktokenCounter
from promptcomp.types import Span


class CharacterCounter:
    name = "test:characters-not-llm-tokens"

    def count(self, text):
        return len(text)


class Provider:
    name = "test:fixed-lexicon-v1"

    def __init__(self, mapping=None):
        self.mapping = mapping or {"substantial": ["big"], "meticulous": ["careful"]}
        self.max_phrase_words = max(len(source.split()) for source in self.mapping)
        self.metadata = {"fixture": True, "version": "1"}

    def candidates(self, source, *, pos=None):
        return tuple(
            value if isinstance(value, LexicalCandidate)
            else LexicalCandidate(source, value, "test:authored-fixture")
            for value in self.mapping.get(source, [])
        )


def run(text, provider=None, **kwargs):
    return suggest(text, provider or Provider(), counter=kwargs.pop("counter", CharacterCounter()), **kwargs)


def test_original_offsets_distinguish_repeated_occurrences_and_preserve_frozen_one():
    text = "The substantial plan works. The substantial plan works."
    first = text.index("substantial")
    second = text.index("substantial", first + 1)
    result = run(text, protected_spans=[(first, first + len("substantial"))])
    assert result.suggested == "The substantial plan works. The big plan works."
    assert result.edits[0].span == Span(second, second + len("substantial"))
    assert result.decisions[0].reason == "protected_caller_protected"
    assert result.decisions[0].tokens_after is None
    assert result.edits[0].sentence_before == "The substantial plan works."
    assert result.edits[0].sentence_after == "The big plan works."


def test_two_edits_in_separate_sentences_keep_original_offsets():
    text = "The substantial plan works. The meticulous plan works."
    result = run(text)
    assert len(result.edits) == 2
    assert result.edits[1].span.start == text.index("meticulous")
    assert result.edits[1].tokens_before == result.edits[0].tokens_after
    assert sum(edit.token_delta for edit in result.edits) == result.tokens_before - result.tokens_after
    assert result.suggested == "The big plan works. The careful plan works."


def test_only_one_edit_per_sentence_and_largest_complete_text_saving_wins():
    result = run("The meticulous preparation produced a substantial improvement.")
    assert len(result.edits) == 1
    assert result.edits[0].original == "substantial"
    assert any(item.reason == "sentence_edit_limit" for item in result.decisions)


def test_repeated_occurrences_in_one_sentence_are_not_globally_replaced():
    result = run("The substantial plan produced a substantial improvement.")
    assert len(result.edits) == 1
    assert result.suggested == "The big plan produced a substantial improvement."


def test_equal_token_gain_uses_provider_rank_then_original_position():
    result = run("The substantial plan works.", Provider({"substantial": ["large", "great"]}))
    assert result.edits[0].replacement == "large"
    assert result.decisions[0].decision == "selected"
    assert result.decisions[1].reason == "sentence_edit_limit"


def test_complete_string_counts_handle_nonadditive_interactions():
    original = "The substantial plan works. The meticulous plan works."
    first_only = "The big plan works. The meticulous plan works."
    combined = "The big plan works. The careful plan works."

    class InteractionCounter:
        name = "test:nonadditive"

        def __init__(self):
            self.seen = []

        def count(self, text):
            self.seen.append(text)
            return {original: 30, first_only: 28, combined: 29}[text]

    counter = InteractionCounter()
    result = run(original, counter=counter)
    assert result.suggested == first_only
    assert result.tokens_after == 28
    assert len(result.edits) == 1
    second = next(item for item in result.decisions if item.original == "meticulous")
    assert second.reason == "no_token_saving" and second.token_delta == -1
    assert set(counter.seen) == {original, first_only, combined}


def test_shorter_word_without_token_saving_is_rejected():
    class ConstantCounter:
        name = "test:constant"

        def count(self, text):
            return 10

    result = run("The substantial plan works.", counter=ConstantCounter())
    assert result.suggested == result.original
    assert result.edits == ()
    assert result.decisions[0].reason == "no_token_saving"


def test_real_tokenizer_counts_entire_phrase_output():
    text = "We proceeded in order to finish the report."
    counter = TiktokenCounter()
    result = run(text, Provider({"in order to": ["to"]}), counter=counter)
    assert result.suggested == "We proceeded to finish the report."
    assert result.tokens_before == counter.count(text)
    assert result.tokens_after == counter.count(result.suggested)
    assert result.tokens_after < result.tokens_before


@pytest.mark.parametrize("text", [
    "The substantial plan must succeed.",
    "The substantial plan did not succeed.",
    "The substantial plan won't succeed.",
    "If approved, the substantial plan succeeds.",
    "The substantial plan succeeds unless funding ends.",
    "Every substantial plan succeeds.",
    "The substantial plan costs 15 dollars.",
    "The substantial plan covers three regions.",
    "The substantial plan describes kilograms.",
    "The substantial plan applies at least once.",
])
def test_scope_and_quantity_sentences_are_frozen(text):
    result = run(text)
    assert result.suggested == text
    assert result.edits == ()
    assert result.decisions and all(item.reason.startswith("protected_") for item in result.decisions)


@pytest.mark.parametrize("text", [
    "The `substantial` field exists.",
    "The ``substantial`` field exists.",
    'The "substantial" field exists.',
    "The 'substantial' field exists.",
    "The “substantial” field exists.",
    "The ‘substantial’ field exists.",
    "The `substantial field exists.",
    'The "substantial field exists.',
    "The {substantial} field exists.",
    "The {{ substantial }} field exists.",
    "The ${substantial} field exists.",
    "The $substantial field exists.",
    "The <substantial> field exists.",
    "The [substantial] field exists.",
    "The { substantial field exists.",
    "The /substantial/file path exists.",
    "The ./substantial/file path exists.",
    "The C:\\substantial\\file path exists.",
    "The https://example.com/substantial page exists.",
    "The substantial/file path exists.",
    'The payload is {"size": "substantial", "description": "long"}.',
    'The payload is {"outer": {"size": "substantial"}}.',
    "size: substantial\n",
    "size = substantial\n",
    "description: |\n  The substantial plan works.\n",
    "- description: The substantial plan works.\n",
    "<instruction>\nThe substantial plan works.\n</instruction>\n",
    "<instruction>\n<instruction>Example</instruction>\nThe substantial plan works.\n</instruction>",
    "<instruction>\nThe substantial plan works.",
    "<instruction>\n<example>The substantial plan works.\n</instruction>",
    "<input>\nThe substantial plan works.\n</input>",
    '<instruction test="a > b">\nThe substantial plan works.\n</instruction>',
    "<!--\nThe substantial plan works.\n-->",
    "<!--\nThe substantial plan works.",
    "<![CDATA[\nThe substantial plan works.\n]]>",
    "> The substantial plan works.",
    "> Example quotation\nThe substantial plan works.",
    "```text\nThe substantial plan works.\n```\n",
    "~~~text\nThe substantial plan works.\n~~~\n",
    "~~~text\nThe substantial plan works.\n",
])
def test_inline_exact_and_structured_regions_are_immutable(text):
    assert run(text).suggested == text


def test_prose_can_change_next_to_immutable_inline_text():
    text = 'The substantial plan contains "substantial" and `meticulous` text.'
    result = run(text)
    assert result.suggested == 'The big plan contains "substantial" and `meticulous` text.'
    assert len(result.edits) == 1


def test_structured_block_survives_offset_shifts_from_prior_edits():
    text = "The substantial plan works.\n```\nsubstantial meticulous\n```\nThe meticulous plan works."
    result = run(text)
    assert result.suggested == "The big plan works.\n```\nsubstantial meticulous\n```\nThe careful plan works."


def test_markup_and_quote_guards_do_not_freeze_following_paragraph():
    text = (
        '<instruction>\n<instruction>Example</instruction>\nThe substantial plan works.\n</instruction>\n\n'
        '> The substantial plan works.\n\n'
        'The substantial plan works.'
    )
    result = run(text)
    assert len(result.edits) == 1
    assert result.edits[0].span.start == text.rindex("substantial")
    assert result.suggested == text[:text.rindex("substantial")] + "big plan works."


def test_self_closing_markup_does_not_freeze_following_prose():
    text = "<instruction />\nThe substantial plan works."
    assert run(text).suggested == "<instruction />\nThe big plan works."


@pytest.mark.parametrize("text", [
    '"Substantial" means the approved plan. The substantial plan works.',
    '"substantial" means the approved plan. The substantial plan works.',
    "'substantial' denotes the approved plan. The substantial plan works.",
])
def test_defined_terms_remain_protected_at_later_occurrences(text):
    result = run(text)
    assert result.suggested == text
    assert any("defined_term" in item.reason for item in result.decisions)


def test_named_entity_and_proper_noun_remain_unchanged():
    result = run("Meticulous Labs develops software.")
    assert result.suggested == result.original
    assert result.decisions[0].reason in {"protected_entity", "protected_proper_noun"}


def test_sentence_initial_case_is_adapted():
    result = run("Substantial improvements followed.")
    assert result.suggested == "Big improvements followed."
    assert result.edits[0].original == "Substantial"


def test_mid_sentence_title_case_is_not_converted_into_common_word():
    result = run("The Substantial plan works.")
    assert result.suggested == result.original
    assert result.decisions[0].reason == "unsupported_case_or_name"


def test_plural_source_rejects_singular_replacement():
    result = run("The procedures govern operations.", Provider({"procedures": ["method"]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "morphology_mismatch"


def test_contextual_pos_mismatch_rejects_replacement():
    result = run("The substantial plan works.", Provider({"substantial": ["run"]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "pos_mismatch"


def test_article_context_change_is_rejected():
    result = run("An enormous plan emerged.", Provider({"enormous": ["big"]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "article_context_change"


@pytest.mark.parametrize("replacement", ["may", "not", "each", "unless", "kilograms"])
def test_candidates_cannot_introduce_protected_content(replacement):
    result = run("The substantial plan works.", Provider({"substantial": [replacement]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "introduced_protected_content"


def test_candidate_cannot_introduce_lowercase_named_entity():
    result = run("The substantial plan works.", Provider({"substantial": ["japanese"]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "introduced_entity_or_number"


@pytest.mark.parametrize("failed_text", ["The substantial plan works.", "The big plan works."])
def test_parser_failure_propagates_without_returning_unchecked_suggestions(monkeypatch, failed_text):
    from promptcomp.lexical import engine

    original_parse = engine.parse

    def broken_parse(text):
        if text == failed_text:
            raise RuntimeError("parser unavailable for this input")
        return original_parse(text)

    monkeypatch.setattr(engine, "parse", broken_parse)
    with pytest.raises(RuntimeError, match="parser unavailable"):
        run("The substantial plan works.")


@pytest.mark.parametrize("replacement", ["", "big\nsmall", " big", "42", "big!", "big  plan"])
def test_malformed_replacements_are_rejected(replacement):
    result = run("The substantial plan works.", Provider({"substantial": [replacement]}))
    assert result.suggested == result.original
    assert result.decisions[0].reason == "invalid_replacement_format"


def test_candidate_source_and_pos_are_checked():
    provider = Provider({"substantial": [LexicalCandidate("wrong", "big", "test")]})
    assert run("The substantial plan works.", provider).decisions[0].reason == "candidate_source_mismatch"
    provider = Provider({"substantial": [LexicalCandidate("substantial", "big", "test", pos="VERB")]})
    assert run("The substantial plan works.", provider).decisions[0].reason == "candidate_pos_mismatch"


@pytest.mark.parametrize("spans", [
    [(-1, 2)], [(0, 999)], [(3, 2)], [(2, 2)], [(True, 3)], [(0.0, 2)], [(0,)], ["0:3"],
])
def test_invalid_protected_spans_raise_actionable_errors(spans):
    with pytest.raises(ValueError, match="protected_spans"):
        run("The substantial plan works.", protected_spans=spans)


def test_explicit_span_forms_and_order_are_normalized():
    text = "The substantial plan works."
    a = run(text, protected_spans=[Span(4, 15), (1, 3), (4, 15)])
    b = run(text, protected_spans=[(1, 3), (4, 15)])
    assert a.config_hash == b.config_hash
    assert a.suggested == text


@pytest.mark.parametrize("maximum", [0, 2, -1, True, 1.0])
def test_unsupported_edit_caps_are_rejected(maximum):
    with pytest.raises(ValueError, match="max_edits_per_sentence=1"):
        run("The substantial plan works.", max_edits_per_sentence=maximum)


def test_no_op_and_empty_input_are_reported_as_unreviewed():
    for text in ("", "A clear report exists."):
        result = run(text)
        assert result.suggested == text and result.edits == ()
        assert result.tokens_before == result.tokens_after
        assert result.review_status == "pending"
        assert "unreviewed" in result.warnings[0]
        assert "do not establish semantic equivalence" in result.warnings[0]


def test_trace_is_deterministic_and_configuration_sensitive():
    text = "The substantial plan works."
    provider = Provider()
    first, second = run(text, provider), run(text, provider)
    a, b = asdict(first), asdict(second)
    a.pop("timings_ms")
    b.pop("timings_ms")
    assert a == b
    provider.metadata["version"] = "2"
    assert run(text, provider).config_hash != first.config_hash
    assert run(text, protected_spans=[(4, 15)]).config_hash != first.config_hash
    assert run(text, Provider({"substantial": ["large"]})).config_hash != first.config_hash


def test_final_recount_failure_rolls_back_selected_edits():
    class UnstableCounter:
        name = "test:unstable-counter"

        def __init__(self):
            self.calls = 0

        def count(self, text):
            self.calls += 1
            return 20 if self.calls == 1 else (19 if self.calls == 2 else 21)

    result = run("The substantial plan works.", counter=UnstableCounter())
    assert result.suggested == result.original
    assert result.edits == () and result.tokens_after == result.tokens_before
    assert result.decisions[0].reason == "invariant_rollback"
    assert len(result.warnings) == 2


def test_invalid_counter_output_is_not_reported_as_token_savings():
    class InvalidCounter:
        name = "test:invalid"

        def count(self, text):
            return -1

    with pytest.raises(ValueError, match="nonnegative integer"):
        run("The substantial plan works.", counter=InvalidCounter())
