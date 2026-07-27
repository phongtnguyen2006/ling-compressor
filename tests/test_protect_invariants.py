# tests/test_protect_invariants.py
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from promptcomp import compress
from promptcomp.protect import extract_numbers, extract_negations, extract_protected_terms, load_protect_list
from promptcomp.parse import parse
from promptcomp.segment import segment
from promptcomp.types import BlockKind

_SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
CORPUS = [p.read_text() for p in sorted(_SAMPLES.glob("*.txt"))]

_PL = load_protect_list()

_MEMO1 = next(p.read_text() for p in _SAMPLES.glob("*.txt") if p.name == "memo1.txt")
_POLICY_SNIP = next(p.read_text() for p in _SAMPLES.glob("*.txt") if p.name == "policy_snip.txt")

# Curated sentences that stress each protected class. Each must be checked
# empirically: some are deliberately over-protected (modal/negation scope
# swallows the whole sentence) and are kept anyway to exercise that path,
# but the majority must actually delete something at ratio 0.4, or the
# invariant assertions below would pass vacuously (see DELETING_DOCS).
CURATED = [
    # Deletable relative clause ("who arrived early that rainy morning")
    # alongside a protected amount and a protected negation.
    "The auditor, who arrived early that rainy morning, confirmed the balance of $4,200 without objection.",
    # Heavily protected: modal ("may not") + negation + percent — the whole
    # sentence sits inside modal/negation scope, so nothing survives to
    # delete. Kept intentionally to exercise the over-protection path.
    "Employees may not claim more than 15% above the standard rate in any month.",
    "No exception applies unless approved in writing by the regional director beforehand.",
    # Deletable relative clause alongside a protected date and negation.
    # ("on a rainy morning", not "that rainy morning" -- the latter is tagged
    # by spaCy as a TIME entity and, once TIME is protected, swallows the
    # whole deletable clause, making this fixture vacuous.)
    "The auditor, arriving early on a rainy morning, confirmed compliance on March 3, 2026, without exception.",
    # Deletable relative clause alongside a protected amount and negation.
    "The manager, who reviewed the file carefully, approved the invoice for $3,750 without further delay.",
    # Contains a spelled-out-number TIME entity ("three hours") plus deletable
    # adjuncts; regression fixture for the under-veto where TIME entities
    # (and thus spelled-out numbers inside them) were not protected.
    "The engineers assembled the finished unit in three hours after a long lunch, and delivered it quietly.",
]

# Docs/sentences empirically confirmed (see below) to produce at least one
# deletion at target_ratio=0.4, min_tokens=0. This guards against the whole
# invariant suite silently degrading into a vacuous no-op suite if the
# scoring/deletion logic changes such that nothing is ever deleted.
DELETING_DOCS = [
    _MEMO1,
    _POLICY_SNIP,
    CURATED[0],
    CURATED[2],
    CURATED[3],
    CURATED[4],
    CURATED[5],
]


def _numbers_multiset(s):
    return Counter(extract_numbers(s))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_numbers_multiset_preserved(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(result.original) == _numbers_multiset(result.compressed)


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_negation_count_non_decreasing(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    before = len(extract_negations(result.original))
    after = len(extract_negations(result.compressed))
    assert after >= before


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_no_vetoed_class_token_deleted(doc):
    # Every negation and number present originally must still be present.
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    assert Counter(extract_numbers(result.compressed)) >= Counter(extract_numbers(result.original))
    assert Counter(extract_negations(result.compressed)) >= Counter(extract_negations(result.original))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_passthrough_blocks_byte_identical(doc):
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    for block in segment(result.original):
        if block.kind is BlockKind.PASSTHROUGH:
            assert block.text in result.compressed


@pytest.mark.parametrize("doc", DELETING_DOCS)
def test_deletion_actually_occurs(doc):
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    assert len(result.deleted) > 0, "fixture no longer exercises deletion; invariants would be vacuous"


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_protected_terms_preserved(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    before = extract_protected_terms(result.original, _PL)
    after = extract_protected_terms(result.compressed, _PL)
    assert after >= before, f"lost protected term(s): {before - after}"


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_named_entities_preserved(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    protected_labels = _PL.ner_numeric | _PL.ner_entity
    doc_parsed = parse(result.original)
    for ent in doc_parsed.ents:
        if ent.label_ in protected_labels:
            assert ent.text in result.compressed, (
                f"protected {ent.label_} entity lost: {ent.text!r}"
            )


PASSTHROUGH_DOC = (
    "The committee approved the annual budget in the morning after a long debate.\n"
    "```\nx = in order to stay\n```\n"
    "The vendor delivered the goods quickly and without any delay.\n"
)


def test_passthrough_present_and_byte_identical():
    result = compress(PASSTHROUGH_DOC, target_ratio=0.4, min_tokens=0)
    blocks = [b for b in segment(PASSTHROUGH_DOC) if b.kind is BlockKind.PASSTHROUGH]
    assert blocks, "fixture must contain a passthrough block"
    for b in blocks:
        assert b.text in result.compressed


@settings(max_examples=40, deadline=None)
@given(
    template=st.sampled_from(CURATED),
    ratio=st.floats(min_value=0.2, max_value=0.9),
)
def test_property_numbers_and_negation_preserved(template, ratio):
    result = compress(template, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(template) == _numbers_multiset(result.compressed)
    assert len(extract_negations(result.compressed)) >= len(extract_negations(template))


from promptcomp.substitute import load_substitutions


def test_no_substitution_touches_a_protected_term():
    phrases, _ = load_substitutions()
    for key, value in phrases.items():
        assert not extract_protected_terms(key, _PL), (
            f"substitution key {key!r} is a protected term; substitution must not rewrite protected classes"
        )
        assert not extract_protected_terms(value, _PL), (
            f"substitution value {value!r} is a protected term; would corrupt protected-term auditing"
        )


# Independent oracle: numeric/temporal entities spaCy recognizes must never lose tokens,
# regardless of what the protect-list yaml currently enumerates.
_NUMERIC_NER_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE", "TIME"}


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_numeric_entities_preserved_independent_oracle(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    for ent in parse(result.original).ents:
        if ent.label_ in _NUMERIC_NER_LABELS:
            assert ent.text.strip() in result.compressed, (
                f"lost numeric entity {ent.label_} {ent.text!r} at ratio {ratio}"
            )
