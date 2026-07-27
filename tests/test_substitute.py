import yaml
from pathlib import Path

from promptcomp.substitute import substitute, load_substitutions, SubstituteResult

DATA = Path(__file__).resolve().parents[1] / "data" / "substitutions.yaml"


def test_substitutions_file_loads_and_has_phrases():
    doc = yaml.safe_load(DATA.read_text())
    assert "version" in doc
    assert isinstance(doc["phrases"], dict)
    assert doc["phrases"]["in order to"] == "to"
    assert doc["phrases"]["due to the fact that"] == "because"


def test_substitute_shortens_known_phrase():
    res = substitute("We did this in order to win.", {"in order to": "to"})
    assert res.text == "We did this to win."
    assert any(s.original == "in order to" for s in res.applied)


def test_substitute_is_case_insensitive_preserves_nothing_extra():
    res = substitute("In order to proceed, wait.", {"in order to": "to"})
    assert res.text == "to proceed, wait."


def test_substitute_longest_match_first():
    phrases = {"due to": "because of", "due to the fact that": "because"}
    res = substitute("He left due to the fact that it rained.", phrases)
    assert res.text == "He left because it rained."


def test_substitute_no_match_is_identity():
    res = substitute("Nothing to change here.", {"in order to": "to"})
    assert res.text == "Nothing to change here."
    assert res.applied == ()


def test_substitute_does_not_break_words():
    # "for" inside "form" must not be touched by a "for" -> "4" style rule.
    res = substitute("Fill the form for me.", {"for": "4"})
    assert "form" in res.text  # 'form' intact
    assert res.text == "Fill the form 4 me."


def test_load_substitutions_default_path():
    phrases, version = load_substitutions()
    assert phrases["in order to"] == "to"
    assert version.startswith("sub-")
