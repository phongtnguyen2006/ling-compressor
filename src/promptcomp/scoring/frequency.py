"""Deterministic word-rarity table for the heuristic scorer.

Loads `data/word_freq.txt`: a min-max-normalized log-frequency rarity per word
in [0.0, 1.0], where 0.0 is the most common word (cheapest to delete) and 1.0
is the rarest in the table (dearest). Out-of-vocabulary words are treated as
maximally rare (they are, by construction, rarer than anything in the top-N
table). This is a corpus-rarity / tf-idf-style lexical-importance signal
(cf. Filippova & Strube 2008), the deterministic replacement for the old
average-word-length rarity proxy.

The table stores integer thousandths so the runtime does only int->float
division — no `math.log` — keeping scores byte-identical across platforms.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Bumping the table (new source or regeneration) must change this string so it
# flows into HeuristicScorer.version and therefore into the audit config_hash.
TABLE_VERSION = "norvig-count1w-top30k-v1"

# OOV words are rarer than any in-table word => maximally dear.
_DEFAULT_RARITY = 1.0

_TABLE: dict[str, float] | None = None


def load_freq_table(path: str | Path | None = None) -> dict[str, float]:
    """Lazily parse and cache the rarity table (word -> rarity in [0,1]).

    `path` overrides the packaged asset (for tests); an override is parsed fresh
    and never cached, so it cannot poison the singleton.
    """
    global _TABLE
    if path is not None:
        return _parse_table(Path(path).read_text(encoding="utf-8"), str(path))
    if _TABLE is None:
        resource = files("promptcomp").joinpath("data/word_freq.txt")
        try:
            text = resource.read_text(encoding="utf-8")
        except FileNotFoundError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError(
                "promptcomp data/word_freq.txt is missing. It ships with the "
                "package; reinstall promptcomp (pip install -e .) to restore it."
            ) from exc
        _TABLE = _parse_table(text, "data/word_freq.txt")
    return _TABLE


def _parse_table(text: str, source: str) -> dict[str, float]:
    """Parse TSV `word<TAB>rarity_milli`, skipping `#` header comments.

    Comment detection requires the '#' line to have no tab, so a legitimate
    '#'-initial *word* entry would still be read rather than silently dropped.
    """
    table: dict[str, float] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        word, sep, milli = line.partition("\t")
        if not sep:
            if line.startswith("#"):
                continue  # header/provenance comment
            raise ValueError(f"{source}:{lineno}: expected <word>TAB<int>, got {line!r}")
        try:
            table[word] = int(milli) / 1000.0
        except ValueError as exc:
            raise ValueError(
                f"{source}:{lineno}: rarity for {word!r} is not an integer: {milli!r}"
            ) from exc
    return table


def get_rarity(word: str) -> float:
    """Rarity of `word` in [0.0, 1.0]; 1.0 for out-of-vocabulary words."""
    return load_freq_table().get(word.lower(), _DEFAULT_RARITY)
