"""Regenerate src/promptcomp/data/word_freq.txt from a unigram frequency list.

Build-time only — promptcomp never runs this at import or compression time. The
shipped table is a frozen artifact; this script exists so that artifact is
reproducible and auditable rather than an opaque blob, and so the source can be
swapped in one place if its licensing is ever in question.

Usage:
    curl -sL https://www.norvig.com/ngrams/count_1w.txt -o count_1w.txt
    python scripts/build_word_freq.py count_1w.txt src/promptcomp/data/word_freq.txt

Input format: `<word><TAB><integer count>` per line.
Expected sha256 of Norvig's count_1w.txt as used for the committed table:
    51df159fd3de12b20e403c108f526e96dbd723d9cabdd5f17955cdc16059e690

Output: `<word><TAB><rarity_milli>` where rarity_milli = round(rarity * 1000) and

    rarity(w) = (log(count_max) - log(count(w))) / (log(count_max) - log(count_min))

i.e. min-max-normalized negative log-frequency over the retained words: 0 for
the most frequent, 1000 for the least. Min-max normalization is an affine
transform of log-frequency, so word *ordering* is identical to raw IDF
(log(TOTAL/count)) while the corpus total cancels out. Two consequences matter:
the output bounds are [0,1] by construction (which the scorer's factor bound
relies on), and no absolute corpus counts are redistributed.

Storing integer thousandths keeps the runtime free of math.log, so scores stay
byte-identical across platforms — a hard determinism requirement (spec 8).
"""
from __future__ import annotations

import hashlib
import math
import sys

TOP_N = 30000
# Bump when the derivation or source changes; must match
# promptcomp.scoring.frequency.TABLE_VERSION.
VERSION = "norvig-count1w-top30k-v1"
SOURCE_URL = "https://norvig.com/ngrams/"


def build(src: str, out: str) -> None:
    raw = open(src, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()

    best: dict[str, int] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        word, cnt = parts
        # Alphabetic only: digits/punctuation are handled by the protect-list,
        # and a rarity table of numerals would be meaningless.
        if not word.isalpha():
            continue
        try:
            count = int(cnt)
        except ValueError:
            continue
        if count <= 0:
            continue
        word = word.lower()
        if count > best.get(word, 0):  # keep max across case variants
            best[word] = count

    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]
    if not ranked:
        raise SystemExit(f"{src}: no usable entries found")

    log_max = math.log(ranked[0][1])
    log_min = math.log(ranked[-1][1])
    span = log_max - log_min
    if span <= 0:
        raise SystemExit("degenerate frequency range; cannot normalize")

    lines = []
    for word, count in ranked:
        rarity = (log_max - math.log(count)) / span
        lines.append(f"{word}\t{min(1000, max(0, round(rarity * 1000)))}")

    header = [
        "# promptcomp word-frequency rarity table",
        f"# version: {VERSION}",
        "# source: Norvig count_1w.txt (Google Web Trillion-Word Corpus unigram"
        f" counts, {SOURCE_URL}); see word_freq.LICENSE",
        f"# source sha256: {digest}",
        f"# entries: {len(lines)} (top alphabetic unigrams by frequency)",
        "# format: <word>\\t<rarity_milli>  where rarity_milli in [0,1000],"
        " 0 = most common (cheap to delete), 1000 = rarest in table (dear)",
        "# rarity = minmax-normalized log-frequency; corpus total cancels, so"
        " runtime needs no math.log and stays byte-deterministic.",
        "# Regenerate with: python scripts/build_word_freq.py <count_1w.txt> <out>",
    ]

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines)} entries to {out}")
    print(f"source sha256: {digest}")
    print(f"count_max={ranked[0][1]} count_min={ranked[-1][1]} log-span={span:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
