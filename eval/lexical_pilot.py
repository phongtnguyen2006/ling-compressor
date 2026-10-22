"""Reproducible, local lexical opportunity pilot (SEM-REQ-002/007/008/009).

Run from the repository with ``python -m eval.lexical_pilot --help``.
Outputs are proposals for review, not a semantic-quality benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

from promptcomp.lexical.candidates import PilotProvider, VectorIndex
from promptcomp.lexical.engine import suggest
from promptcomp.lexical.tokenization import local_counter as _local_counter
from promptcomp.tokens import TokenCounter

SCHEMA_VERSION = "lexical-pilot-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PilotDocument:
    id: str
    text: str
    group: str
    split: str = "development"


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_documents(corpus: Path) -> list[PilotDocument]:
    """Read a preselected JSONL manifest or a directory of UTF-8 .txt files."""
    if corpus.is_dir():
        docs = [PilotDocument(p.stem, p.read_text(encoding="utf-8"), p.stem)
                for p in sorted(corpus.glob("*.txt"))]
    elif corpus.is_file() and corpus.suffix == ".jsonl":
        docs = []
        for line_number, line in enumerate(corpus.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                doc_id, text = row["id"], row["text"]
                group, split = row.get("group", doc_id), row.get("split", "development")
                if not all(isinstance(x, str) for x in (doc_id, text, group, split)):
                    raise ValueError("id, text, group, and split must be strings")
                if not doc_id.strip() or not group.strip() or not split.strip():
                    raise ValueError("id, group, and split must be nonempty")
                docs.append(PilotDocument(doc_id, text, group, split))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid corpus record on line {line_number}: {exc}") from exc
    else:
        raise ValueError("corpus must be a directory of .txt files or a .jsonl manifest")
    if not docs:
        raise ValueError("corpus contains no documents")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("corpus document IDs must be unique")
    return sorted(docs, key=lambda d: d.id)


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("promptcomp", "tiktoken", "spacy", "en-core-web-sm", "numpy", "nltk"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _software_hashes() -> dict[str, str]:
    files = [Path(__file__)]
    files.extend((REPO_ROOT / "src" / "promptcomp").rglob("*.py"))
    files.extend((REPO_ROOT / "src" / "promptcomp" / "data").glob("*.yaml"))
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for arm in dict.fromkeys(row["arm"] for row in rows):
        group = [row for row in rows if row["arm"] == arm]
        before = sum(row["tokens_before"] for row in group)
        after = sum(row["tokens_after"] for row in group)
        reductions = [1 - row["tokens_after"] / row["tokens_before"]
                      if row["tokens_before"] else 0.0 for row in group]
        latency = sorted(row["timings_ms"].get("total", 0.0) for row in group)
        edits = [edit for row in group for edit in row["edits"]]
        sources: dict[str, int] = {}
        for edit in edits:
            source = edit["provenance"]
            sources[source] = sources.get(source, 0) + 1
        summaries.append({
            "arm": arm, "documents": len(group), "tokens_before": before, "tokens_after": after,
            "tokens_saved": before - after,
            "corpus_reduction": 1 - after / before if before else 0.0,
            "median_document_reduction": statistics.median(reductions),
            "changed_documents": sum(row["original"] != row["suggested"] for row in group),
            "selected_edits": len(edits),
            "word_edits": sum(len(edit["original"].split()) == 1 for edit in edits),
            "phrase_edits": sum(len(edit["original"].split()) > 1 for edit in edits),
            "selected_edits_by_source": sources,
            "candidate_decisions": sum(len(row["decisions"]) for row in group),
            "p50_ms": statistics.median(latency),
            "p95_ms": latency[max(0, math.ceil(0.95 * len(latency)) - 1)],
            "semantic_quality": "not_evaluated", "human_review": "pending",
        })
    return summaries


class _RecordedProvider:
    """Record selected pools without making provider identity depend on history."""

    def __init__(self, provider):
        self.provider = provider
        self.name = provider.name
        self.max_phrase_words = provider.max_phrase_words
        self.metadata = getattr(provider, "metadata", {})
        self.pools = {}

    def candidates(self, source, *, pos=None):
        candidates = self.provider.candidates(source, pos=pos)
        key = (source, pos)
        if key not in self.pools:
            pool_method = getattr(self.provider, "candidate_pool", None)
            baseline = pool_method(source, pos=pos) if pool_method is not None else None
            self.pools[key] = {
                "source": source, "pos": pos,
                "baseline_pool": [asdict(candidate) for candidate in baseline] if baseline is not None else None,
                "offered_candidates": [asdict(candidate) for candidate in candidates],
            }
        return candidates


def _review_exports(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Blind source/arm in review records; keep separate keys for attribution."""
    edits, prompts, keys = {}, {}, []
    for row in rows:
        prompt = {"doc_id": row["doc_id"], "original": row["original"],
                  "suggested": row["suggested"]}
        prompt_id = _digest(prompt)
        prompts[prompt_id] = {
            "review_id": prompt_id, **prompt, "meaning_preserved": None,
            "grammatical": None, "harmful_change": None, "reviewer": None, "notes": "",
        }
        keys.append({"review_id": prompt_id, "kind": "prompt", "doc_id": row["doc_id"],
                     "arm": row["arm"], "config_hash": row["config_hash"]})
        for decision in row["decisions"]:
            if decision.get("token_delta") is None or decision["token_delta"] <= 0:
                continue
            content = {key: decision[key] for key in
                       ("span", "original", "replacement", "sentence_before", "sentence_after")}
            content["doc_id"] = row["doc_id"]
            edit_id = _digest(content)
            edits[edit_id] = {
                "review_id": edit_id, **content, "meaning_preserved": None,
                "grammatical": None, "harmful_change": None, "reviewer": None, "notes": "",
            }
            keys.append({"review_id": edit_id, "kind": "edit", "doc_id": row["doc_id"],
                         "arm": row["arm"], "provenance": decision["provenance"],
                         "decision": decision["decision"], "reason": decision["reason"],
                         "token_delta": decision["token_delta"]})
    return ([edits[key] for key in sorted(edits)], [prompts[key] for key in sorted(prompts)],
            sorted(keys, key=lambda key: (key["review_id"], key["arm"], key["kind"])))


def run_pilot(documents: list[PilotDocument], providers: dict, counter: TokenCounter,
              output_dir: Path, *, corpus_label: str, configuration: dict | None = None) -> dict:
    """Run already-selected inputs with explicit providers and export raw evidence."""
    if not documents or not providers:
        raise ValueError("at least one document and one arm are required")
    if len({d.id for d in documents}) != len(documents):
        raise ValueError("document IDs must be unique")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output path must be a new or empty directory; existing evidence is not overwritten")

    inputs = [{"id": d.id, "group": d.group, "split": d.split,
               "sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest()} for d in documents]
    config = configuration or {}
    provenance = {arm: getattr(provider, "metadata", {"name": provider.name})
                  if provider is not None else {"name": "unchanged"}
                  for arm, provider in providers.items()}
    identity = {"schema_version": SCHEMA_VERSION, "inputs": inputs,
                "counter": counter.name, "providers": provenance, "configuration": config,
                "versions": _versions(), "software_sha256": _software_hashes()}
    manifest = {
        **identity, "run_fingerprint": _digest(identity), "corpus_label": corpus_label,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "unreviewed_opportunity_pilot",
        "input_selection": "Inputs selected before candidate discovery; no quality-based filtering.",
        "limitations": [
            "No human judgments, downstream accuracy, semantic equivalence, or net API savings established.",
            "All-input token savings include unchanged documents; empty inputs contribute zero tokens.",
            "Latency is a single sequential pass, includes cold/warm variation, and is not an isolated performance benchmark.",
            "Candidate-pool headroom is not a universal compression bound.",
        ],
    }
    if not config.get("wordnet", False):
        manifest["limitations"].append("WordNet disabled: B0 is curated rules only, not the complete lexical baseline.")
    if config.get("synthetic_vectors", False):
        manifest["limitations"].append("SYNTHETIC VECTOR FIXTURE: embedding results test software only, not GloVe quality.")
    if len({record["sha256"] for record in inputs}) != len(inputs):
        manifest["limitations"].append("Corpus contains duplicate text; observations are not all independent.")
    rows = []
    candidate_pools = []
    for doc in documents:
        for arm, provider in providers.items():
            if provider is None:
                count = counter.count(doc.text)
                result = {"original": doc.text, "suggested": doc.text,
                          "tokens_before": count, "tokens_after": count,
                          "token_counter": counter.name, "provider": "unchanged",
                          "edits": [], "decisions": [], "warnings": [],
                          "config_hash": _digest({"arm": "u", "counter": counter.name}),
                          "timings_ms": {"total": 0.0}, "review_status": "pending"}
            else:
                recorded = _RecordedProvider(provider)
                result = asdict(suggest(doc.text, recorded, counter=counter))
                candidate_pools.extend({"doc_id": doc.id, "arm": arm, **pool}
                                       for pool in recorded.pools.values()
                                       if pool["baseline_pool"] or pool["offered_candidates"])
            rows.append({"doc_id": doc.id, "group": doc.group, "split": doc.split,
                         "arm": arm, **result})

    summary = _summary(rows)
    reviews, prompt_reviews, review_key = _review_exports(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest)
    _write_jsonl(output_dir / "inputs.jsonl", [asdict(d) for d in documents])
    _write_jsonl(output_dir / "results.jsonl", rows)
    _write_jsonl(output_dir / "candidate-pools.jsonl", candidate_pools)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "edit-review.jsonl", reviews)
    _write_jsonl(output_dir / "prompt-review.jsonl", prompt_reviews)
    _write_jsonl(output_dir / "review-key.jsonl", review_key)
    lines = ["---", f"id: SEM-RUN-{manifest['run_fingerprint'][:12]}",
             "type: experimental-evidence", "status: unreviewed", "tags: [semantic-compression, pilot]", "---", "",
             "# Lexical opportunity pilot", "", f"Corpus: {corpus_label}", "",
             f"Tokenizer: `{counter.name}`. Documents: {len(documents)}.", "",
             f"Run fingerprint: `{manifest['run_fingerprint']}`", "",
             "**These are unreviewed suggestions, not demonstrated semantic preservation.**", "",
             "| Arm | Before | After | Saved | Reduction | Changed inputs | Word / phrase edits |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['tokens_before']} | {row['tokens_after']} | {row['tokens_saved']} | "
                     f"{row['corpus_reduction']:.2%} | {row['changed_documents']}/{row['documents']} | "
                     f"{row['word_edits']} / {row['phrase_edits']} |")
    lines.extend(["", "## Evidence", "",
                  "- [Manifest and versions](manifest.json)", "- [Frozen inputs](inputs.jsonl)",
                  "- [Occurrence decisions and complete outputs](results.jsonl)",
                  "- [Baseline pools and offered candidates](candidate-pools.jsonl)",
                  "- [Summary data](summary.json)",
                  "- [Blind edit review](edit-review.jsonl)",
                  "- [Blind complete-prompt review](prompt-review.jsonl)",
                  "- [Attribution key — keep separate during review](review-key.jsonl)",
                  "", "Human judgment fields are intentionally blank. No annotation was inferred from cosine or token savings.",
                  "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in manifest["limitations"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output_dir.iterdir()) if p.is_file()}
    _write_json(output_dir / "checksums.json", artifacts)
    return {"run_fingerprint": manifest["run_fingerprint"], "summary": summary,
            "edit_reviews": len(reviews), "prompt_reviews": len(prompt_reviews)}


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Directory of .txt or JSONL id/text/group/split manifest")
    parser.add_argument("--out", type=Path, required=True, help="New output directory (never overwrite evidence)")
    parser.add_argument("--corpus-label", default="Local corpus; representativeness unverified")
    parser.add_argument("--arms", default=None, help="Comma-separated u,b0,b1,b2,b2_union; vectors required for b1/b2")
    parser.add_argument("--encoding", choices=("o200k_base", "cl100k_base"), default="o200k_base")
    parser.add_argument("--wordnet", action="store_true", help="Use installed NLTK WordNet assets; no automatic download")
    parser.add_argument("--vectors", type=Path, help="Local GloVe/word2vec text vectors")
    parser.add_argument("--vector-label", default="Local vector file; training provenance unverified")
    parser.add_argument("--synthetic-vectors", action="store_true", help="Label synthetic fixture results explicitly")
    parser.add_argument("--max-words", type=_positive_int, default=50000)
    parser.add_argument("--max-candidates", type=_positive_int, default=8)
    parser.add_argument("--embedding-slots", type=int, default=4)
    parser.add_argument("--max-documents", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        docs = load_documents(args.corpus)
        pool_count = len(docs)
        if args.max_documents is not None and len(docs) > args.max_documents:
            docs = sorted(random.Random(args.seed).sample(docs, args.max_documents), key=lambda d: d.id)
        arms = (args.arms or ("u,b0,b1,b2" if args.vectors else "u,b0")).split(",")
        arms = [arm.strip().lower() for arm in arms]
        if not arms or len(arms) != len(set(arms)) or any(arm not in {"u", "b0", "b1", "b2", "b2_union"} for arm in arms):
            raise ValueError("arms must be unique choices from u,b0,b1,b2,b2_union")
        if any(arm in {"b1", "b2", "b2_union"} for arm in arms) and args.vectors is None:
            raise ValueError("embedding arms require --vectors; baseline-only fallback is not implicit")
        if args.synthetic_vectors and args.vectors is None:
            raise ValueError("--synthetic-vectors requires --vectors")
        vectors = VectorIndex.from_text(args.vectors, max_words=args.max_words) if args.vectors else None
        providers = {arm: None if arm == "u" else PilotProvider(
            arm, vectors=vectors, use_wordnet=args.wordnet, max_candidates=args.max_candidates,
            embedding_slots=args.embedding_slots) for arm in arms}
        config = {"arms": arms, "wordnet": args.wordnet, "encoding": args.encoding,
                  "max_candidates": args.max_candidates, "embedding_slots": args.embedding_slots,
                  "max_words": args.max_words, "max_documents": args.max_documents, "seed": args.seed,
                  "candidate_document_count": pool_count, "vector_label": args.vector_label if vectors else None,
                  "synthetic_vectors": args.synthetic_vectors}
        result = run_pilot(docs, providers, _local_counter(args.encoding), args.out,
                           corpus_label=args.corpus_label, configuration=config)
    except (ValueError, RuntimeError, OSError, ImportError, LookupError) as exc:
        print(f"lexical pilot: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / 'summary.md'}; {result['edit_reviews']} edit reviews, "
          f"{result['prompt_reviews']} prompt reviews pending.")
    for row in result["summary"]:
        print(f"{row['arm']}: {row['tokens_before']} -> {row['tokens_after']} tokens "
              f"({row['corpus_reduction']:.2%}); unreviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
