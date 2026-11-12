"""Paired holdout opportunity report from a frozen lexical pilot.

Run ``python -m eval.semantic_decision --run RUN --out NEW_DIRECTORY``.
This report measures proposed token changes only; quality and net economics remain unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import unicodedata

from .semantic_benchmark import _source


ARMS = ("u", "b0", "b1", "b2")
COMPARISONS = (("b1", "b0"), ("b2", "b1"), ("b2", "b0"))
BOOTSTRAP_SAMPLES = 2000


def _split(value: str) -> str:
    if value == "development":
        return value
    if value in {"holdout", "held-out", "held_out"}:
        return "held-out"
    raise ValueError(f"unsupported split {value!r}; expected development or held-out")


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _category(group: str) -> str:
    """Use the declared category prefix, without changing family group identity."""
    if ":" not in group:
        return "unlabeled"
    return group.split(":", 1)[0].strip() or "unlabeled"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency(row: dict) -> float:
    timings = row.get("timings_ms")
    value = timings.get("total") if isinstance(timings, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid total latency for {row['doc_id']}/{row['arm']}")
    return float(value)


def _edits(row: dict) -> int:
    edits = row.get("edits")
    if not isinstance(edits, list) or any(not isinstance(edit, dict) for edit in edits):
        raise ValueError(f"invalid selected edits for {row['doc_id']}/{row['arm']}")
    return len(edits)


def _arm_summary(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    before = sum(row["tokens_before"] for row in selected)
    after = sum(row["tokens_after"] for row in selected)
    times = [_latency(row) for row in selected]
    return {
        "arm": arm, "documents": len(selected), "all_input_tokens": before,
        "output_tokens": after, "tokens_saved": before - after,
        "all_input_reduction": (before - after) / before if before else 0.0,
        "changed_documents": sum(row["original"] != row["suggested"] for row in selected),
        "selected_edits": sum(_edits(row) for row in selected),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
    }


def _pair(doc: dict, left: dict, right: dict, comparison: str) -> dict:
    # The delta is the first arm minus the second arm, on the same input.
    return {
        "doc_id": doc["id"], "group": doc["group"], "category": _category(doc["group"]),
        "split": _split(doc["split"]),
        "comparison": comparison,
        "tokens_saved_delta": (left["tokens_before"] - left["tokens_after"])
                              - (right["tokens_before"] - right["tokens_after"]),
        "tokens_after_delta": left["tokens_after"] - right["tokens_after"],
        "changed_delta": int(left["original"] != left["suggested"])
                         - int(right["original"] != right["suggested"]),
        "selected_edits_delta": _edits(left) - _edits(right),
        "latency_ms_delta": _latency(left) - _latency(right),
    }


def _uncertainty(rows: list[dict], seed: int) -> dict:
    """Cluster-resample whole groups; estimate the mean delta per held-out document."""
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["tokens_saved_delta"])
    names = sorted(groups)
    group_rows = [{"group": name, "documents": len(groups[name]),
                   "tokens_saved_delta": sum(groups[name]),
                   "mean_document_delta": statistics.mean(groups[name])} for name in names]
    observed = sum(row["tokens_saved_delta"] for row in rows) / len(rows)
    rng = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        draw = [groups[rng.choice(names)] for _ in names]
        samples.append(sum(map(sum, draw)) / sum(map(len, draw)))
    return {
        "method": "group_cluster_bootstrap_percentile", "unit": "group",
        "estimand": "mean_tokens_saved_delta_per_held_out_document",
        "groups": len(names), "documents": len(rows), "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_document_delta": observed,
        "interval_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "group_deltas": group_rows,
    }


def build_report(run_dir: Path) -> dict:
    """Validate and summarize a fingerprinted pilot without modifying its evidence."""
    manifest, _, results = _source(Path(run_dir))
    arms = manifest["configuration"]["arms"]
    if not set(ARMS) <= set(arms):
        raise ValueError("pilot requires complete u/b0/b1/b2 arms")
    inputs = manifest["inputs"]
    by_pair = {(row["doc_id"], row["arm"]): row for row in results}
    split_groups: dict[str, set[str]] = {}
    split_texts: dict[str, set[str]] = {}
    docs_by_split: dict[str, list[dict]] = {"development": [], "held-out": []}
    for doc in inputs:
        split = _split(doc["split"])
        row = by_pair[doc["id"], "u"]
        normalized = _normalized(row["original"])
        if normalized in split_texts and split not in split_texts[normalized]:
            raise ValueError(f"duplicate normalized text across splits: {doc['id']}")
        split_texts.setdefault(normalized, set()).add(split)
        if doc["group"] in split_groups and split not in split_groups[doc["group"]]:
            raise ValueError(f"same group across splits: {doc['group']}")
        split_groups.setdefault(doc["group"], set()).add(split)
        docs_by_split[split].append(doc)
    if any(not docs for docs in docs_by_split.values()):
        raise ValueError("pilot requires development and held-out splits")
    for doc in inputs:
        unchanged = by_pair[doc["id"], "u"]
        if (unchanged["original"] != unchanged["suggested"]
                or unchanged["tokens_before"] != unchanged["tokens_after"] or _edits(unchanged)):
            raise ValueError(f"u arm must be unchanged: {doc['id']}")
        for arm in ARMS:
            _edits(by_pair[doc["id"], arm])
            _latency(by_pair[doc["id"], arm])
    splits = {}
    pairs = []
    held_out = {}
    for split, docs in docs_by_split.items():
        ids = {doc["id"] for doc in docs}
        rows = [row for row in results if row["doc_id"] in ids and row["arm"] in ARMS]
        splits[split] = {"documents": len(docs), "groups": len({doc["group"] for doc in docs}),
                         "arms": [_arm_summary(rows, arm) for arm in ARMS], "categories": []}
        for category in sorted({_category(doc["group"]) for doc in docs}):
            category_docs = [doc for doc in docs if _category(doc["group"]) == category]
            category_ids = {doc["id"] for doc in category_docs}
            category_rows = [row for row in rows if row["doc_id"] in category_ids]
            comparisons = {}
            for left_arm, right_arm in COMPARISONS:
                label = f"{left_arm}-{right_arm}"
                category_pairs = [_pair(doc, by_pair[doc["id"], left_arm],
                                        by_pair[doc["id"], right_arm], label) for doc in category_docs]
                total = sum(row["tokens_saved_delta"] for row in category_pairs)
                comparisons[label] = {
                    "documents": len(category_docs),
                    "groups": len({doc["group"] for doc in category_docs}),
                    "tokens_saved_delta": total,
                    "mean_document_delta": total / len(category_docs),
                }
            splits[split]["categories"].append({
                "category": category, "documents": len(category_docs),
                "groups": len({doc["group"] for doc in category_docs}),
                "arms": [_arm_summary(category_rows, arm) for arm in ARMS],
                "comparisons": comparisons,
            })
        for left_arm, right_arm in COMPARISONS:
            label = f"{left_arm}-{right_arm}"
            compared = [_pair(doc, by_pair[doc["id"], left_arm],
                              by_pair[doc["id"], right_arm], label) for doc in docs]
            pairs.extend(compared)
            if split == "held-out":
                seed = int(hashlib.sha256(f"{manifest['run_fingerprint']}:{label}".encode()).hexdigest()[:16], 16)
                held_out[label] = _uncertainty(compared, seed)
    return {
        "schema_version": "semantic-decision-v1", "source_run_fingerprint": manifest["run_fingerprint"],
        "source_counter": manifest["counter"], "source_corpus_label": manifest.get("corpus_label"),
        "quality": "unreviewed", "net_economics": "unknown",
        "decision": "no_production_go_no_go_claim", "splits": splits,
        "paired_documents": sorted(pairs, key=lambda row: (row["split"], row["comparison"], row["doc_id"])),
        "held_out_uncertainty": held_out,
        "limitations": [
            "Token changes are unreviewed proposals; meaning preservation and downstream quality are unknown.",
            "Net economics are unknown: provider costs, review costs, and downstream effects are not measured.",
            "Bootstrap intervals resample observed groups; they do not correct corpus selection bias or establish production performance.",
            "Latency comes from one sequential pilot pass and is not an isolated performance benchmark.",
            "No production go/no-go decision is inferred from this report.",
        ],
    }


def write_report(report: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output must be a new directory; existing evidence is not overwritten")
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Paired semantic pilot holdout", "", f"Source run: `{report['source_run_fingerprint']}`", "",
             "Quality: **unreviewed**. Net economics: **unknown**. No production go/no-go claim.", ""]
    for split, section in report["splits"].items():
        lines.extend([f"## {split.title()} ({section['documents']} documents, {section['groups']} groups)", "",
                      "| Arm | All-input tokens | Output tokens | Saved | All-input reduction | Changed | Selected edits | p50 ms | p95 ms |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in section["arms"]:
            lines.append(f"| {row['arm']} | {row['all_input_tokens']} | {row['output_tokens']} | "
                         f"{row['tokens_saved']} | {row['all_input_reduction']:.3%} | {row['changed_documents']} | {row['selected_edits']} | "
                         f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} |")
        lines.append("")
        lines.extend(["### Per category", "", "Groups are compared using the full family label. "
                      "Unprefixed groups appear as `unlabeled`.", "",
                      "| Category | Documents | Groups | B0 saved | B1 saved | B2 saved | B1-B0 mean | B2-B1 mean |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for category in section["categories"]:
            arms = {row["arm"]: row for row in category["arms"]}
            name = category["category"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {category['documents']} | {category['groups']} | "
                         f"{arms['b0']['tokens_saved']} | {arms['b1']['tokens_saved']} | "
                         f"{arms['b2']['tokens_saved']} | "
                         f"{category['comparisons']['b1-b0']['mean_document_delta']:.3f} | "
                         f"{category['comparisons']['b2-b1']['mean_document_delta']:.3f} |")
        lines.append("")
    lines.extend(["## Held-out paired deltas", "", "Positive saved-token deltas favor the first arm. "
                  "Intervals resample whole groups (2,000 draws).", "",
                  "| Comparison | Mean saved tokens per document | 95% bootstrap interval | Groups |",
                  "| --- | ---: | ---: | ---: |"])
    for label, row in report["held_out_uncertainty"].items():
        low, high = row["interval_95"]
        lines.append(f"| {label} | {row['mean_document_delta']:.3f} | [{low:.3f}, {high:.3f}] | {row['groups']} |")
    lines.extend(["", "Per-document paired deltas and group totals are in [report.json](report.json).", "",
                  "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Frozen lexical pilot directory")
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    try:
        if args.out.resolve() == args.run.resolve() or args.run.resolve() in args.out.resolve().parents:
            raise ValueError("output directory must be outside the source run")
        report = build_report(args.run)
        write_report(report, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"semantic decision: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / 'summary.md'}; quality unreviewed, net economics unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
