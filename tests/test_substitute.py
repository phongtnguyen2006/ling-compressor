import yaml
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "substitutions.yaml"


def test_substitutions_file_loads_and_has_phrases():
    doc = yaml.safe_load(DATA.read_text())
    assert "version" in doc
    assert isinstance(doc["phrases"], dict)
    assert doc["phrases"]["in order to"] == "to"
    assert doc["phrases"]["due to the fact that"] == "because"
