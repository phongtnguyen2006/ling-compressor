import yaml
from importlib.resources import files


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
