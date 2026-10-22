"""Summarize a frozen lexical pilot; optionally measure a local embedding cosine proxy.

Run ``python -m eval.semantic_benchmark --run RUN --out NEW_DIRECTORY``.
The default uses only stored evidence and makes no semantic-quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant: {value}")))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant: {value}")))
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def _nonnegative_int(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _source(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    checksums_path = run_dir / "checksums.json"
    if checksums_path.exists():
        checksums = _read_json(checksums_path)
        if not isinstance(checksums, dict) or not {"manifest.json", "results.jsonl", "summary.json"} <= checksums.keys():
            raise ValueError("incomplete run checksums")
        for name, digest in checksums.items():
            if not isinstance(name, str) or name != Path(name).name or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("invalid run checksum entry")
            if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"run checksum mismatch: {name}")
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "summary.json")
    results = _read_jsonl(run_dir / "results.jsonl")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "lexical-pilot-v1":
        raise ValueError("unsupported pilot manifest schema")
    identity_keys = ("schema_version", "inputs", "counter", "providers", "configuration",
                     "versions", "software_sha256")
    if any(key not in manifest for key in identity_keys):
        raise ValueError("manifest lacks pilot identity fields")
    identity = {key: manifest[key] for key in identity_keys}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False).encode("utf-8")).hexdigest()
    if manifest.get("run_fingerprint") != fingerprint:
        raise ValueError("manifest run fingerprint mismatch")
    configuration = manifest["configuration"]
    if not isinstance(configuration, dict) or not isinstance(manifest["counter"], str):
        raise ValueError("invalid manifest configuration or counter")
    inputs = manifest["inputs"]
    arms = configuration.get("arms")
    if not isinstance(inputs, list) or not inputs or not isinstance(arms, list) or not arms:
        raise ValueError("manifest requires inputs and arms")
    if any(not isinstance(arm, str) or not arm for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("manifest arms must be unique nonempty strings")
    docs = {}
    for item in inputs:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key]
                                              for key in ("id", "group", "split", "sha256")):
            raise ValueError("invalid manifest input")
        if item["id"] in docs:
            raise ValueError("duplicate manifest input")
        docs[item["id"]] = item
    if not isinstance(summary, list) or len(summary) != len(arms):
        raise ValueError("summary must contain one row per arm")
    summaries = {}
    for row in summary:
        if (not isinstance(row, dict) or not isinstance(row.get("arm"), str)
                or row["arm"] not in arms or row["arm"] in summaries):
            raise ValueError("invalid or duplicate summary arm")
        summaries[row["arm"]] = row
    if len(summaries) != len(arms):
        raise ValueError("summary arm coverage mismatch")
    if len(results) != len(docs) * len(arms):
        raise ValueError("results do not cover every input and arm")
    seen = set()
    originals = {}
    for row in results:
        arm, doc_id = row.get("arm"), row.get("doc_id")
        if (not isinstance(arm, str) or not isinstance(doc_id, str)
                or arm not in arms or doc_id not in docs or (arm, doc_id) in seen):
            raise ValueError("invalid or duplicate result pair")
        seen.add((arm, doc_id))
        before = _nonnegative_int(row.get("tokens_before"), "tokens_before")
        after = _nonnegative_int(row.get("tokens_after"), "tokens_after")
        if after > before:
            raise ValueError("result token count increased")
        original, suggested = row.get("original"), row.get("suggested")
        if not isinstance(original, str) or not isinstance(suggested, str):
            raise ValueError("result text must be strings")
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != docs[doc_id]["sha256"]:
            raise ValueError(f"original text hash mismatch: {doc_id}")
        if doc_id in originals and originals[doc_id] != (original, before):
            raise ValueError(f"input differs between arms: {doc_id}")
        originals[doc_id] = (original, before)
        if row.get("group") != docs[doc_id]["group"] or row.get("split") != docs[doc_id]["split"]:
            raise ValueError(f"result group or split mismatch: {doc_id}")
        if row.get("token_counter") != manifest.get("counter"):
            raise ValueError("result token counter differs from manifest")
    for arm in arms:
        rows = [row for row in results if row["arm"] == arm]
        source = summaries[arm]
        expected = {"documents": len(rows),
                    "tokens_before": sum(row["tokens_before"] for row in rows),
                    "tokens_after": sum(row["tokens_after"] for row in rows),
                    "tokens_saved": sum(row["tokens_before"] - row["tokens_after"] for row in rows),
                    "changed_documents": sum(row["original"] != row["suggested"] for row in rows)}
        for key, value in expected.items():
            if source.get(key) != value or type(source.get(key)) is not int:
                raise ValueError(f"summary mismatch for {arm}: {key}")
    return manifest, summary, results


def _embedding_cosines(rows: list[dict], model_path: Path) -> dict[tuple[str, str], float]:
    """Use an already present SentenceTransformer directory, with remote lookups disabled."""
    if not model_path.is_dir():
        raise ValueError("--model-path must be an existing local model directory")
    # Transformers and huggingface_hub honor these flags, including nested model assets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError("sentence-transformers is required for --model-path") from exc
    try:
        model = SentenceTransformer(str(model_path.resolve()), local_files_only=True)
    except TypeError as exc:
        raise ValueError("installed sentence-transformers must support local_files_only") from exc
    changed = [row for row in rows if row["original"] != row["suggested"]]
    if not changed:
        return {}
    texts = list(dict.fromkeys(text for row in changed for text in (row["original"], row["suggested"])))
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
    if len(vectors) != len(texts):
        raise ValueError("embedding model returned the wrong number of vectors")
    by_text = dict(zip(texts, vectors))
    scores = {}
    for row in changed:
        left, right = by_text[row["original"]], by_text[row["suggested"]]
        if len(left) != len(right) or not len(left):
            raise ValueError("embedding dimensions differ or are empty")
        a = [float(value) for value in left]
        b = [float(value) for value in right]
        anorm = math.sqrt(sum(value * value for value in a))
        bnorm = math.sqrt(sum(value * value for value in b))
        if not math.isfinite(anorm * bnorm) or anorm == 0 or bnorm == 0:
            raise ValueError("embedding vector must be finite and nonzero")
        score = sum(x * y for x, y in zip(a, b)) / (anorm * bnorm)
        if not math.isfinite(score):
            raise ValueError("embedding cosine must be finite")
        scores[row["arm"], row["doc_id"]] = max(-1.0, min(1.0, score))
    return scores


def benchmark(run_dir: Path, *, model_path: Path | None = None) -> dict:
    """Build dashboard-ready measurements from a validated pilot run."""
    manifest, _, results = _source(Path(run_dir))
    arms = manifest["configuration"]["arms"]
    cosines = _embedding_cosines(results, Path(model_path)) if model_path is not None else {}
    output = []
    for arm in arms:
        rows = [row for row in results if row["arm"] == arm]
        changed = [row for row in rows if row["original"] != row["suggested"]]
        before = sum(row["tokens_before"] for row in rows)
        after = sum(row["tokens_after"] for row in rows)
        scores = [cosines[arm, row["doc_id"]] for row in changed] if model_path is not None else []
        output.append({
            "arm": arm, "documents": len(rows), "tokens_before": before,
            "tokens_after": after, "tokens_saved": before - after,
            "token_reduction": (before - after) / before if before else 0.0,
            "coverage": {"result_pairs": len(rows), "expected_pairs": len(manifest["inputs"]),
                         "changed_outputs": len(changed), "unchanged_outputs": len(rows) - len(changed)},
            "semantic_preservation": {
                "status": ("not_computed" if model_path is None else
                           "no_changed_outputs" if not changed else "computed"),
                "metric": "sentence_transformer_cosine" if model_path is not None else None,
                "mean_cosine_changed_outputs": sum(scores) / len(scores) if scores else None,
                "evaluated_pairs": len(scores), "eligible_changed_pairs": len(changed),
            },
        })
    return {
        "schema_version": "semantic-benchmark-v1",
        "source_run_fingerprint": manifest.get("run_fingerprint"),
        "source_counter": manifest["counter"],
        "source_corpus_label": manifest.get("corpus_label"),
        "embedding_model_path": str(Path(model_path).resolve()) if model_path is not None else None,
        "arms": output,
        "limitations": [
            "Token savings describe stored pilot outputs; they are not net API savings.",
            "Coverage counts stored document/arm pairs and changed outputs, not semantic correctness.",
            "Sentence-transformer cosine is a similarity proxy, not proof of meaning preservation.",
            "No downstream task accuracy or human judgment is inferred.",
        ],
    }


def write_report(report: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output must be a new or empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Semantic benchmark", "", f"Source run: `{report['source_run_fingerprint']}`", "",
             f"Counter: `{report['source_counter']}`", "",
             "| Arm | Coverage | Changed | Tokens before → after | Saved | Reduction | Mean cosine (changed) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["arms"]:
        coverage = row["coverage"]
        semantic = row["semantic_preservation"]
        cosine = semantic["mean_cosine_changed_outputs"]
        score = f"{cosine:.4f} ({semantic['evaluated_pairs']} pairs)" if cosine is not None else semantic["status"]
        lines.append(f"| {row['arm']} | {coverage['result_pairs']}/{coverage['expected_pairs']} | "
                     f"{coverage['changed_outputs']} | {row['tokens_before']} → {row['tokens_after']} | "
                     f"{row['tokens_saved']} | {row['token_reduction']:.2%} | {score} |")
    lines.extend(["", "Semantic scores are shown only when computed with an explicitly supplied local model.",
                  "Cosine is a proxy, not proof of meaning preservation.", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "[Machine-readable report](report.json)", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing lexical pilot run directory")
    parser.add_argument("--out", type=Path, required=True, help="New or empty report directory")
    parser.add_argument("--model-path", type=Path,
                        help="Explicit local SentenceTransformer directory; requires sentence-transformers")
    args = parser.parse_args(argv)
    try:
        if args.out.resolve() == args.run.resolve() or args.run.resolve() in args.out.resolve().parents:
            raise ValueError("output directory must be outside the source run")
        report = benchmark(args.run, model_path=args.model_path)
        write_report(report, args.out)
    except (OSError, ValueError, ImportError) as exc:
        print(f"semantic benchmark: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
