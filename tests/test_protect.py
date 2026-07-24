import yaml
from importlib.resources import files

from promptcomp.protect import load_protect_list, load_defined_terms, ProtectList


def test_protect_terms_resource_loads():
    doc = yaml.safe_load(files("promptcomp").joinpath("data/protect_terms.yaml").read_text())
    assert doc["version"].startswith("protect-")
    assert "MONEY" in doc["ner_numeric"]
    assert "ORG" in doc["ner_entity"]
    assert "shall" in doc["single_terms"]["modal"]
    assert "provided that" in doc["phrase_terms"]["conditional"]


def test_defined_terms_resource_loads():
    doc = yaml.safe_load(files("promptcomp").joinpath("data/defined_terms.yaml").read_text())
    assert doc["version"].startswith("defined-")
    assert isinstance(doc["terms"], list)


def test_load_protect_list_default():
    pl = load_protect_list()
    assert isinstance(pl, ProtectList)
    assert pl.version.startswith("protect-")
    assert "MONEY" in pl.ner_numeric
    assert "ORG" in pl.ner_entity
    assert "shall" in pl.single_terms["modal"]
    assert "provided that" in pl.phrase_terms["conditional"]


def test_load_defined_terms_default():
    terms, version = load_defined_terms()
    assert isinstance(terms, frozenset)
    assert version.startswith("defined-")


from promptcomp.protect import extract_numbers, extract_negations


def test_extract_numbers_covers_currency_percent_decimals():
    text = "The fee is $1,000.00, a 15% surcharge, and 42 units at $0.67 each."
    nums = extract_numbers(text)
    assert "$1,000.00" in nums
    assert "15%" in nums
    assert "42" in nums
    assert "$0.67" in nums


def test_extract_numbers_is_multiset_ordered():
    text = "5 and 5 and 10."
    assert extract_numbers(text) == ["5", "5", "10"]


def test_extract_negations_finds_all_forms():
    text = "It is not allowed; no exceptions, never waived, without consent."
    negs = extract_negations(text)
    assert "not" in negs
    assert "no" in negs
    assert "never" in negs
    assert "without" in negs


def test_extract_negations_word_boundary():
    # "nothing" contains "no" but must not count as a bare "no".
    text = "There is nothing here."
    assert "no" not in extract_negations(text)


from promptcomp.protect import extract_defined_terms


def test_defined_term_quoted_means_pattern():
    text = 'The "Reimbursable Amount" means the total eligible expense.'
    terms = extract_defined_terms(text)
    assert "Reimbursable Amount" in terms


def test_defined_term_parenthetical_pattern():
    text = 'Acme Corporation ("the Company") shall pay all fees.'
    terms = extract_defined_terms(text)
    assert any("Company" in t for t in terms)


def test_no_false_positive_on_plain_prose():
    text = "The committee approved the budget in the morning."
    assert extract_defined_terms(text) == set()


def test_extract_defined_terms_handles_curly_quotes():
    text = "The “Effective Date” means the date hereof."
    assert "Effective Date" in extract_defined_terms(text)


def test_extract_negations_catches_contractions():
    negs = extract_negations("It isn't allowed and won't be permitted.")
    assert negs.count("n't") == 2


def test_extract_numbers_no_trailing_comma_and_signs():
    assert extract_numbers("$1,000, $2,000 owed")[0] == "$1,000"
    assert ".5" in extract_numbers("a value of .5 units")
    assert "-5" in extract_numbers("a delta of -5 points")
