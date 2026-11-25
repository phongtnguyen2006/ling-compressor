"""Offline lexical candidate sources for controlled opportunity experiments.

Neighbors and WordNet lemmas are proposals, not certified synonyms. B0 and B1
share their complete baseline pool. B2 reserves separate baseline/generator
slots; unused slots are not borrowed. B2_UNION is an uncapped diagnostic.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..substitute import load_substitutions
from .models import LexicalCandidate


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _positive_int(value: int, label: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'} integer")


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "Vector candidates require NumPy. Install promptcomp[lexical] or numpy; "
            "no model or vector assets are downloaded automatically."
        ) from exc
    return numpy


class VectorIndex:
    """Exact, deterministic cosine retrieval over locally supplied text vectors.

    `max_words` limits retained rows, not integrity checks or the source-file
    fingerprint. Every source row is validated. This deliberately favors a
    reproducible small pilot over approximate-neighbor infrastructure.
    """

    def __init__(self, words: tuple[str, ...], matrix: Any, metadata: Mapping[str, Any]):
        self.words = words
        self._matrix = matrix
        self._positions = {word: i for i, word in enumerate(words)}
        self._metadata = dict(metadata)
        self.name = "vectors:" + _digest(self._metadata_identity())[:16]

    def _metadata_identity(self) -> dict[str, Any]:
        return {key: value for key, value in self._metadata.items() if key != "source_path"}

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._metadata, "name": self.name}

    @classmethod
    def from_text(cls, path: str | Path, max_words: int | None = None) -> VectorIndex:
        """Read GloVe text or word2vec text with an optional `rows dims` header."""
        if max_words is not None:
            _positive_int(max_words, "max_words")
        np = _numpy()
        source_path = Path(path)
        hasher = hashlib.sha256()
        expected_rows: int | None = None
        dimension: int | None = None
        header_seen = False
        first_content = True
        row_count = 0
        seen: set[str] = set()
        words: list[str] = []
        vectors: list[Any] = []
        with source_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                hasher.update(raw)
                try:
                    fields = raw.decode("utf-8").strip().split()
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Invalid UTF-8 vector row at line {line_number}") from exc
                if not fields:
                    continue
                if first_content and len(fields) == 2 and all(f.isdigit() for f in fields):
                    expected_rows, dimension = map(int, fields)
                    if expected_rows <= 0 or dimension <= 0:
                        raise ValueError("Vector header row count and dimension must be positive")
                    header_seen = True
                    first_content = False
                    continue
                first_content = False
                if len(fields) < 2:
                    raise ValueError(f"Vector row at line {line_number} has no coordinates")
                word, coordinates = fields[0], fields[1:]
                if word in seen:
                    raise ValueError(f"Duplicate vector word {word!r} at line {line_number}")
                seen.add(word)
                if dimension is None:
                    dimension = len(coordinates)
                if len(coordinates) != dimension:
                    raise ValueError(
                        f"Vector dimension mismatch at line {line_number}: "
                        f"expected {dimension}, got {len(coordinates)}"
                    )
                try:
                    values = np.asarray([float(value) for value in coordinates], dtype=np.float64)
                except ValueError as exc:
                    raise ValueError(f"Invalid vector coordinate at line {line_number}") from exc
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite vector at line {line_number}")
                scale = float(np.max(np.abs(values)))
                if scale == 0:
                    raise ValueError(f"Zero vector at line {line_number}")
                # Scale before the norm so large finite values cannot overflow.
                values /= scale
                values /= np.linalg.norm(values)
                row_count += 1
                if max_words is None or len(words) < max_words:
                    words.append(word)
                    vectors.append(values.astype(np.float32))
        if not words:
            raise ValueError("Vector file contains no vectors")
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(f"Vector header declares {expected_rows} rows; found {row_count}")
        matrix = np.stack(vectors)
        matrix.flags.writeable = False
        return cls(tuple(words), matrix, {
            "sha256": hasher.hexdigest(),
            "source_path": str(source_path.resolve()),
            "format": "word2vec-text" if header_seen else "glove-text",
            "dimensions": dimension,
            "source_rows": row_count,
            "retained_rows": len(words),
            "max_words": max_words,
            "dtype": "float32",
            "normalization": "l2",
            "retrieval": "exact-cosine-lexical-ties-casefold-unique",
        })

    def _position(self, word: str) -> int | None:
        index = self._positions.get(word)
        return self._positions.get(word.casefold()) if index is None else index

    def similarity(self, source: str, replacement: str) -> float | None:
        left, right = self._position(source), self._position(replacement)
        if left is None or right is None:
            return None
        return max(-1.0, min(1.0, float(self._matrix[left] @ self._matrix[right])))

    def nearest(
        self, source: str, *, k: int = 8, exclude: Iterable[str] = ()
    ) -> tuple[tuple[str, float], ...]:
        """Return distinct lexical alternatives; exact ties use lexical word order.

        Exclusion and provider deduplication are case-insensitive, so apply that
        same identity before the limit. Otherwise case variants can consume
        generator slots without providing additional replacement choices.
        """
        _positive_int(k, "k", allow_zero=True)
        position = self._position(source)
        if position is None or k == 0:
            return ()
        np = _numpy()
        excluded = {word.casefold() for word in exclude} | {source.casefold()}
        scores = np.clip(self._matrix @ self._matrix[position], -1.0, 1.0)
        total = len(self.words)
        pool_size = min(total, max(32, 2 * k))
        while True:
            if pool_size == total:
                candidate_indices = np.arange(total)
            else:
                # Partitioning selects the best scores in linear expected time.
                # Include every tie at the boundary before lexical sorting so
                # results remain identical to the former full-vocabulary sort.
                top = np.argpartition(scores, total - pool_size)[total - pool_size:]
                floor = scores[top].min()
                candidate_indices = np.flatnonzero(scores >= floor)
            ranked = sorted(
                ((self.words[i], float(scores[i])) for i in candidate_indices
                 if self.words[i].casefold() not in excluded),
                key=lambda item: (-item[1], item[0].casefold(), item[0]),
            )
            neighbors: list[tuple[str, float]] = []
            used = set(excluded)
            for word, score in ranked:
                normalized = word.casefold()
                if normalized in used:
                    continue
                used.add(normalized)
                neighbors.append((word, score))
                if len(neighbors) == k:
                    return tuple(neighbors)
            if pool_size == total:
                return tuple(neighbors)
            # Heavy exclusions or case variants can exhaust a small pool.
            pool_size = min(total, pool_size * 2)


def _load_wordnet() -> tuple[Any, dict[str, Any]]:
    """Require provisioned assets; never download or silently change the pool."""
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError as exc:
        raise ImportError(
            "WordNet candidates require NLTK. Install promptcomp[lexical] or nltk, "
            "then provision WordNet explicitly with python -m nltk.downloader wordnet."
        ) from exc
    try:
        wordnet.ensure_loaded()
        version = wordnet.get_version()
        # Hash logical corpus contents in a fixed file order, including filenames.
        # CorpusReader works with unpacked and zip-backed local installations.
        digest = hashlib.sha256()
        for file_id in sorted(wordnet.fileids()):
            digest.update(file_id.encode("utf-8") + b"\0")
            with wordnet.open(file_id) as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            digest.update(b"\0")
    except LookupError as exc:
        raise RuntimeError(
            "WordNet was requested but its local corpus is unavailable. Provision it "
            "explicitly with python -m nltk.downloader wordnet (and set NLTK_DATA if "
            "needed), or disable WordNet for a labeled curated-only experiment."
        ) from exc
    return wordnet, {
        "status": "available", "version": version, "nltk_version": nltk.__version__,
        "sha256": digest.hexdigest(), "digest_kind": "sorted-logical-corpus-contents",
    }


_UNIVERSAL_TO_WORDNET = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
_WORDNET_TO_UNIVERSAL = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}


class PilotProvider:
    """Fixed-budget baseline, vector-ranking, and vector-generation experiment arms.

    Curated rules retain priority in all arms. B1 ranks within provenance tiers,
    keeping candidates without vectors at the end of their tier. B2 reserves
    `max_candidates - embedding_slots` baseline slots and `embedding_slots`
    additional-neighbor slots; neither side may fill the other's unused slots.
    """

    def __init__(
        self, arm: str = "b0", *, vectors: VectorIndex | None = None,
        use_wordnet: bool = False, max_candidates: int = 8,
        embedding_slots: int = 4, phrases: Mapping[str, str] | None = None,
    ):
        if arm not in {"b0", "b1", "b2", "b2_union"}:
            raise ValueError("arm must be b0, b1, b2, or b2_union")
        _positive_int(max_candidates, "max_candidates")
        _positive_int(embedding_slots, "embedding_slots", allow_zero=True)
        if embedding_slots > max_candidates:
            raise ValueError("embedding_slots must not exceed max_candidates")
        if arm in {"b2", "b2_union"} and not 0 < embedding_slots < max_candidates:
            raise ValueError("B2 requires at least one baseline slot and one embedding slot")
        if arm != "b0" and vectors is None:
            raise ValueError(f"{arm} requires an explicit local VectorIndex")
        if not isinstance(use_wordnet, bool):
            raise ValueError("use_wordnet must be boolean")
        version = "custom"
        if phrases is None:
            phrases, version = load_substitutions()
        normalized: dict[str, str] = {}
        for source, replacement in phrases.items():
            if not isinstance(source, str) or not isinstance(replacement, str):
                raise ValueError("Phrase keys and replacements must be strings")
            source = " ".join(source.casefold().split())
            replacement = " ".join(replacement.split())
            if not source or not replacement or source == replacement.casefold():
                continue
            if source in normalized and normalized[source] != replacement:
                raise ValueError(f"Conflicting case-insensitive phrase rules for {source!r}")
            normalized[source] = replacement
        self.arm = arm
        self.vectors = vectors
        self.max_candidates = max_candidates
        self.embedding_slots = embedding_slots
        self._phrases = normalized
        self._cache_size = 1024
        self._pool_cache: OrderedDict[tuple[str, str | None], tuple[LexicalCandidate, ...]] = OrderedDict()
        self._candidate_cache: OrderedDict[tuple[str, str | None], tuple[LexicalCandidate, ...]] = OrderedDict()
        self.max_phrase_words = max((len(source.split()) for source in normalized), default=1)
        self._wordnet = None
        wordnet_metadata: dict[str, Any] = {"status": "disabled", "version": None}
        if use_wordnet:
            self._wordnet, wordnet_metadata = _load_wordnet()
        self._metadata = {
            "arm": arm,
            "max_candidates": max_candidates,
            "embedding_slots": embedding_slots,
            "baseline_slots": max_candidates - embedding_slots if arm == "b2" else max_candidates,
            "unused_slot_policy": "no-borrowing" if arm == "b2" else "not-applicable",
            "uncapped_diagnostic": arm == "b2_union",
            "union_extra_neighbor_limit": embedding_slots if arm == "b2_union" else None,
            "curated": {"version": version, "sha256": _digest(normalized), "rules": len(normalized)},
            "wordnet": wordnet_metadata,
            "vectors": None if vectors is None else vectors.metadata,
            "pool": "curated+wordnet" if use_wordnet else "curated-only",
            "ranking": "curated-first,lexical" if arm == "b0" else "curated-first,cosine,lexical",
            "selection_order": "baseline-then-additional-neighbors" if arm in {"b2", "b2_union"} else "ranked-baseline",
            "sample_provenance": ["curated", "wordnet:<synset>", "embedding:<vector-id>"],
            "warning": "Candidates are unvalidated proposals; cosine is not synonymy.",
            "memoization": {"capacity_per_pool": self._cache_size, "key": "exact-source-and-pos"},
        }
        identity = dict(self._metadata)
        if vectors is not None:
            identity["vectors"] = {k: v for k, v in vectors.metadata.items() if k != "source_path"}
        self.name = f"pilot:{arm}:" + _digest(identity)[:16]

    @property
    def metadata(self) -> dict[str, Any]:
        # Return independent nested structures so callers cannot mutate the manifest.
        return json.loads(json.dumps({**self._metadata, "name": self.name}))

    def _baseline(self, source: str, pos: str | None) -> tuple[LexicalCandidate, ...]:
        normalized = " ".join(source.casefold().split())
        if not normalized:
            return ()
        found: list[LexicalCandidate] = []
        if normalized in self._phrases:
            found.append(LexicalCandidate(source, self._phrases[normalized], "curated"))
        # Deliberately query a single word only; phrase discovery is a separate arm.
        if self._wordnet is not None and " " not in normalized:
            wordnet_pos = _UNIVERSAL_TO_WORDNET.get(pos) if pos is not None else None
            if pos is None or wordnet_pos is not None:
                for synset in sorted(self._wordnet.synsets(normalized, pos=wordnet_pos), key=lambda s: s.name()):
                    candidate_pos = _WORDNET_TO_UNIVERSAL.get(synset.pos())
                    for lemma in sorted(synset.lemmas(), key=lambda item: item.name()):
                        replacement = lemma.name().replace("_", " ").strip()
                        if replacement and replacement.casefold() != normalized:
                            found.append(LexicalCandidate(
                                source, replacement, "wordnet:" + synset.name(), pos=candidate_pos,
                            ))
        unique: dict[str, LexicalCandidate] = {}
        for candidate in found:
            key = candidate.replacement.casefold()
            if key not in unique:
                similarity = None if self.vectors is None else self.vectors.similarity(normalized, candidate.replacement)
                unique[key] = LexicalCandidate(
                    candidate.source, candidate.replacement, candidate.provenance, similarity, candidate.pos,
                )
        return tuple(unique.values())

    def _rank(self, candidates: Iterable[LexicalCandidate], *, lexical: bool = False) -> tuple[LexicalCandidate, ...]:
        def key(candidate: LexicalCandidate):
            return (
                0 if candidate.provenance == "curated" else 1,
                0 if lexical else (candidate.similarity is None),
                0 if lexical or candidate.similarity is None else -candidate.similarity,
                candidate.replacement.casefold(), candidate.replacement, candidate.provenance,
            )
        return tuple(sorted(candidates, key=key))

    def candidate_pool(self, source: str, *, pos: str | None = None) -> tuple[LexicalCandidate, ...]:
        """Expose the complete baseline pool before arm-specific caps/allocation."""
        key = (source, pos)
        if key in self._pool_cache:
            self._pool_cache.move_to_end(key)
            return self._pool_cache[key]
        result = self._rank(self._baseline(source, pos), lexical=self.arm == "b0")
        self._remember(self._pool_cache, key, result)
        return result

    def _remember(self, cache, key, result) -> None:
        cache[key] = result
        cache.move_to_end(key)
        if len(cache) > self._cache_size:
            cache.popitem(last=False)

    def candidates(self, source: str, *, pos: str | None = None) -> tuple[LexicalCandidate, ...]:
        key = (source, pos)
        if key in self._candidate_cache:
            self._candidate_cache.move_to_end(key)
            return self._candidate_cache[key]
        result = self._candidates_uncached(source, pos=pos)
        self._remember(self._candidate_cache, key, result)
        return result

    def _candidates_uncached(self, source: str, *, pos: str | None) -> tuple[LexicalCandidate, ...]:
        baseline = self.candidate_pool(source, pos=pos)
        if self.arm in {"b0", "b1"}:
            return baseline[:self.max_candidates]
        assert self.vectors is not None
        neighbors = self.vectors.nearest(
            " ".join(source.casefold().split()), k=self.embedding_slots,
            exclude=(candidate.replacement for candidate in baseline),
        )
        generated: list[LexicalCandidate] = []
        seen = {candidate.replacement.casefold() for candidate in baseline}
        for replacement, score in neighbors:
            normalized = replacement.casefold()
            if normalized in seen or not replacement.strip():
                continue
            seen.add(normalized)
            generated.append(LexicalCandidate(
                source, replacement, f"embedding:{self.vectors.name}", score,
            ))
        retained = baseline if self.arm == "b2_union" else baseline[:self.max_candidates - self.embedding_slots]
        return (*retained, *generated)
