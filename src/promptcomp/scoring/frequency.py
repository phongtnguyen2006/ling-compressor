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

# Bumping the table (new source or regeneration) must change this string so it
# flows into HeuristicScorer.version and therefore into the audit config_hash.
TABLE_VERSION = "norvig-count1w-top30k-v1"

# OOV words are rarer than any in-table word => maximally dear.
_DEFAULT_RARITY = 1.0

_TABLE: dict[str, float] | None = None


def load_freq_table() -> dict[str, float]:
    """Lazily parse and cache the rarity table (word -> rarity in [0,1])."""
    global _TABLE
    if _TABLE is None:
        text = files("promptcomp").joinpath("data/word_freq.txt").read_text(encoding="utf-8")
        table: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            word, _, milli = line.partition("\t")
            if not milli:
                continue
            table[word] = int(milli) / 1000.0
        _TABLE = table
    return _TABLE


def get_rarity(word: str) -> float:
    """Rarity of `word` in [0.0, 1.0]; 1.0 for out-of-vocabulary words."""
    return load_freq_table().get(word.lower(), _DEFAULT_RARITY)
