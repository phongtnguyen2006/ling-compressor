"""Stage 2: protect-list veto.

Strikes any candidate subtree that contains — or is syntactically scoped by —
a protected token. This is a hard rule: over-protecting is safe, under-protecting
is a build failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProtectList:
    version: str
    ner_numeric: frozenset[str]
    ner_entity: frozenset[str]
    single_terms: dict[str, frozenset[str]]
    phrase_terms: dict[str, frozenset[str]]


def _read_resource(name: str, path: str | Path | None) -> dict:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = files("promptcomp").joinpath(f"data/{name}").read_text()
    return yaml.safe_load(text)


def load_protect_list(path: str | Path | None = None) -> ProtectList:
    doc = _read_resource("protect_terms.yaml", path)
    single = {k: frozenset(t.lower() for t in v) for k, v in doc.get("single_terms", {}).items()}
    phrase = {k: frozenset(t.lower() for t in v) for k, v in doc.get("phrase_terms", {}).items()}
    return ProtectList(
        version=str(doc["version"]),
        ner_numeric=frozenset(doc.get("ner_numeric", [])),
        ner_entity=frozenset(doc.get("ner_entity", [])),
        single_terms=single,
        phrase_terms=phrase,
    )


def load_defined_terms(path: str | Path | None = None) -> tuple[frozenset[str], str]:
    doc = _read_resource("defined_terms.yaml", path)
    return frozenset(doc.get("terms", []) or []), str(doc.get("version", ""))
