"""Compare frozen 10k, 50k, and 400k lexical pilot runs offline.

Run ``python -m eval.vector_size_report --run-10k DIR --run-50k DIR
--run-400k DIR --out NEW_DIRECTORY``. Token changes are unreviewed proposals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from .semantic_benchmark import _source


SIZES = {"10k": 10_000, "50k": 50_000, "400k": 400_000}
REQUIRED_ARMS = {"u", "b0", "b1", "b2"}


def _vector_metadata(manifest: dict) -> dict:
    providers = manifest["providers"]
    if not isinstance(providers, dict) or set(providers) != set(manifest["configuration"]["arms"]):
        raise ValueError("provider metadata does not cover manifest arms")
    vectors = []
    for arm in ("b0", "b1", "b2"):
        provider = providers.get(arm)
        vector = provider.get("vectors") if isinstance(provider, dict) else None
        if not isinstance(vector, dict):
            raise ValueError(f"missing vector metadata for {arm}")
        vectors.append(vector)
    if any(vector != vectors[0] for vector in vectors[1:]):
        raise ValueError("vector metadata differs between arms")
    vector = vectors[0]
    for key in ("sha256", "format", "dimensions", "source_rows", "retained_rows", "max_words"):
        if key not in vector:
            raise ValueError(f"missing vector metadata: {key}")
    if (not isinstance(vector["sha256"], str) or len(vector["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in vector["sha256"])):
        raise ValueError("invalid vector source hash")
    for key in ("dimensions", "source_rows", "retained_rows", "max_words"):
        if type(vector[key]) is not int or vector[key] < 1:
            raise ValueError(f"invalid vector {key}")
    if vector["retained_rows"] != min(vector["source_rows"], vector["max_words"]):
        raise ValueError("vector retained rows do not match source rows and max_words")
    return vector


def _settings(manifest: dict) -> dict:
    """Remove only size-dependent vector identity and descriptive path/label."""
    providers = {}
    for arm, provider in manifest["providers"].items():
        if not isinstance(provider, dict):
            raise ValueError(f"invalid provider metadata for {arm}")
        cleaned = {key: value for key, value in provider.items() if key != "name"}
        vector = cleaned.get("vectors")
        if isinstance(vector, dict):
            cleaned["vectors"] = {key: value for key, value in vector.items()
                                  if key not in {"max_words", "retained_rows", "source_path", "name"}}
        providers[arm] = cleaned
    return {
        "counter": manifest["counter"],
        "configuration": {key: value for key, value in manifest["configuration"].items()
                          if key not in {"max_words", "vector_label"}},
        "providers": providers,
        "versions": manifest["versions"],
        "software_sha256": manifest["software_sha256"],
    }


def _peak_memory(manifest: dict) -> dict:
    measurements = manifest.get("measurements", {})
    if not isinstance(measurements, dict):
        raise ValueError("invalid run measurements")
    value = measurements.get("peak_rss_bytes")
    if value is None:
        return {"status": "unmeasured", "peak_rss_bytes": None}
    if type(value) is not int or value < 1:
        raise ValueError("peak_rss_bytes must be a positive measured integer")
    return {"status": "measured", "peak_rss_bytes": value}


def _arm(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    before = sum(row["tokens_before"] for row in selected)
    after = sum(row["tokens_after"] for row in selected)
    times = []
    edits = 0
    edited_documents = 0
    for row in selected:
        value = row.get("timings_ms", {}).get("total") if isinstance(row.get("timings_ms"), dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid total runtime for {row['doc_id']}/{arm}")
        if not isinstance(row.get("edits"), list) or any(not isinstance(edit, dict) for edit in row["edits"]):
            raise ValueError(f"invalid selected edits for {row['doc_id']}/{arm}")
        times.append(float(value))
        edits += len(row["edits"])
        edited_documents += bool(row["edits"])
    times.sort()
    return {"arm": arm, "documents": len(selected), "changed_documents": sum(
        row["original"] != row["suggested"] for row in selected),
        "edited_documents": edited_documents,
        "edit_coverage": edited_documents / len(selected),
        "selected_edits": edits, "all_input_tokens": before, "output_tokens": after,
        "tokens_saved": before - after, "all_input_reduction": (before - after) / before if before else 0.0,
        "runtime_total_ms": sum(times), "runtime_p50_ms": statistics.median(times),
        "runtime_p95_ms": times[math.ceil(0.95 * len(times)) - 1]}


def build_report(runs: dict[str, Path]) -> dict:
    """Validate three independent runs, then aggregate their stored results."""
    if set(runs) != set(SIZES):
        raise ValueError("provide exactly 10k, 50k, and 400k runs")
    loaded = {}
    for label, size in SIZES.items():
        manifest, _, rows = _source(Path(runs[label]))
        config = manifest["configuration"]
        arms = config["arms"]
        if not REQUIRED_ARMS <= set(arms):
            raise ValueError(f"{label} run requires u/b0/b1/b2 arms")
        if config.get("max_words") != size:
            raise ValueError(f"{label} run max_words must be {size}")
        vector = _vector_metadata(manifest)
        if vector["max_words"] != size:
            raise ValueError(f"{label} vector max_words must be {size}")
        by_pair = {(row["doc_id"], row["arm"]): row for row in rows}
        for doc in manifest["inputs"]:
            unchanged = by_pair[doc["id"], "u"]
            if (unchanged["original"] != unchanged["suggested"]
                    or unchanged["tokens_before"] != unchanged["tokens_after"]
                    or unchanged.get("edits") != []):
                raise ValueError(f"{label} unchanged arm differs for {doc['id']}")
        loaded[label] = (manifest, rows, vector)
    reference = loaded["10k"][0]
    inputs = sorted(reference["inputs"], key=lambda item: item["id"])
    settings = _settings(reference)
    source_hash = loaded["10k"][2]["sha256"]
    reference_counts = {row["doc_id"]: row["tokens_before"] for row in loaded["10k"][1]
                        if row["arm"] == "u"}
    for label in ("50k", "400k"):
        manifest, rows, vector = loaded[label]
        if sorted(manifest["inputs"], key=lambda item: item["id"]) != inputs:
            raise ValueError(f"{label} corpus text hashes, groups, IDs, or splits differ")
        if set(manifest["configuration"]["arms"]) != set(reference["configuration"]["arms"]):
            raise ValueError(f"{label} arms differ")
        if _settings(manifest) != settings:
            raise ValueError(f"{label} settings apart from vector retention differ")
        if vector["sha256"] != source_hash:
            raise ValueError(f"{label} vector source hash differs")
        counts = {row["doc_id"]: row["tokens_before"] for row in rows if row["arm"] == "u"}
        if counts != reference_counts:
            raise ValueError(f"{label} input token counts differ")
    splits = sorted({doc["split"] for doc in inputs})
    if not splits:
        raise ValueError("corpus has no splits")
    output = []
    for label in SIZES:
        manifest, rows, vector = loaded[label]
        sections = {}
        for split in ("all", *splits):
            subset = rows if split == "all" else [row for row in rows if row["split"] == split]
            sections[split] = {"documents": len(subset), "arms": {
                arm: _arm(subset, arm) for arm in manifest["configuration"]["arms"]}}
            arms = sections[split]["arms"]
            sections[split]["incremental_tokens_saved"] = {
                "b2_vs_b0": arms["b2"]["tokens_saved"] - arms["b0"]["tokens_saved"],
                "b2_vs_b1": arms["b2"]["tokens_saved"] - arms["b1"]["tokens_saved"],
            }
        output.append({"size": label, "run_fingerprint": manifest["run_fingerprint"],
                       "source_rows": vector["source_rows"], "retained_rows": vector["retained_rows"],
                       "peak_memory": _peak_memory(manifest), "splits": sections})
    return {"schema_version": "vector-size-report-v1", "source_vector_sha256": source_hash,
            "corpus_documents": len(inputs), "splits": {split: sum(doc["split"] == split for doc in inputs)
                                                       for split in splits},
            "arms": reference["configuration"]["arms"], "counter": reference["counter"],
            "runs": output, "quality": "not_evaluated",
            "limitations": ["Token savings are unreviewed proposals, not semantic-quality or net API savings.",
                            "Runtime is a single sequential pilot pass, not an isolated performance benchmark.",
                            "Peak memory is reported only when a run records measured peak_rss_bytes."]}


def write_report(report: dict, output_dir: Path, runs: dict[str, Path]) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output must be a new directory; existing evidence is not overwritten")
    target = output_dir.resolve()
    if any(target == Path(run).resolve() or Path(run).resolve() in target.parents for run in runs.values()):
        raise ValueError("output must be outside source run directories")
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Frozen lexical pilot vector-size comparison", "",
             "All runs use the same corpus text hashes, splits, arms, and settings apart from vector retention.",
             "Token savings are unreviewed proposals. Semantic quality and net API savings were not evaluated.", "",
             f"Documents: {report['corpus_documents']}; splits: " + ", ".join(
                 f"{key}={value}" for key, value in report["splits"].items()) + ".", "",
             "| Size | Source rows | Retained rows | Peak RSS bytes | Run fingerprint |",
             "| --- | ---: | ---: | ---: | --- |"]
    for run in report["runs"]:
        memory = run["peak_memory"]
        lines.append(f"| {run['size']} | {run['source_rows']} | {run['retained_rows']} | "
                     f"{memory['peak_rss_bytes'] if memory['status'] == 'measured' else 'unmeasured'} | "
                     f"`{run['run_fingerprint']}` |")
    for split in ("all", *report["splits"]):
        lines.extend(["", f"## {split} inputs", "",
                      "| Size | Arm | Documents | Changed | Edited inputs | Edit coverage | Edits | Input tokens | Output tokens | Saved | All-input reduction | Total ms | p50 ms | p95 ms |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for run in report["runs"]:
            for arm in report["arms"]:
                row = run["splits"][split]["arms"][arm]
                lines.append(f"| {run['size']} | {arm} | {row['documents']} | {row['changed_documents']} | "
                             f"{row['edited_documents']} | {row['edit_coverage']:.1%} | "
                             f"{row['selected_edits']} | {row['all_input_tokens']} | {row['output_tokens']} | "
                             f"{row['tokens_saved']} | {row['all_input_reduction']:.3%} | "
                             f"{row['runtime_total_ms']:.2f} | {row['runtime_p50_ms']:.2f} | "
                             f"{row['runtime_p95_ms']:.2f} |")
        lines.extend(["", "| Size | B2 vs B0 extra tokens saved | B2 vs B1 extra tokens saved |",
                      "| --- | ---: | ---: |"])
        for run in report["runs"]:
            delta = run["splits"][split]["incremental_tokens_saved"]
            lines.append(f"| {run['size']} | {delta['b2_vs_b0']} | {delta['b2_vs_b1']} |")
    lines.extend(["", "## Limitations", "", *(f"- {item}" for item in report["limitations"]), ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in SIZES:
        parser.add_argument(f"--run-{label}", type=Path, required=True,
                            help=f"Frozen lexical_pilot run with max_words={SIZES[label]}")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    args = parser.parse_args(argv)
    runs = {label: getattr(args, f"run_{label}") for label in SIZES}
    try:
        report = build_report(runs)
        write_report(report, args.out, runs)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"vector size report: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / 'report.json'} and {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
