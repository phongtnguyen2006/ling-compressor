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
