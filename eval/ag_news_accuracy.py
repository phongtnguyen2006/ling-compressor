"""Offline, paired AG News topic-classification proxy for a frozen lexical pilot.

Requires local AG News train Parquet and the JSONL produced by prepare_ag_news.
No dataset download or network access is performed. Run with
``python -m eval.ag_news_accuracy --help``. This is not LLM downstream accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path
import sys

ARMS = ("b0", "b1", "b2")
LABELS = ("World", "Sports", "Business", "Sci/Tech")
SCHEMA = "ag-news-classification-proxy-v1"
PILOT_IDENTITY_KEYS = ("schema_version", "inputs", "counter", "providers",
                       "configuration", "versions", "software_sha256")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha(value: str) -> str:
    return _sha(value.encode("utf-8"))


def _digest(value: object) -> str:
    return _text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                allow_nan=False))


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected object")
        rows.append(row)
    return rows


def _load_frozen(pilot_dir: Path, corpus: Path) -> tuple[dict, list[dict], list[dict]]:
    required = ("manifest.json", "inputs.jsonl", "results.jsonl", "checksums.json")
    for name in required:
        if not (pilot_dir / name).is_file():
            raise ValueError(f"pilot is missing {name}")
    checksums = json.loads((pilot_dir / "checksums.json").read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or any(name not in checksums for name in required[:-1]):
        raise ValueError("pilot checksums are incomplete")
    for name, expected in checksums.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str):
            raise ValueError("invalid pilot checksum entry")
        if not (pilot_dir / name).is_file() or _file_sha(pilot_dir / name) != expected:
            raise ValueError(f"pilot checksum mismatch: {name}")
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lexical-pilot-v1":
        raise ValueError("unsupported pilot schema")
    try:
        fingerprint = _digest({key: manifest[key] for key in PILOT_IDENTITY_KEYS})
    except KeyError as exc:
        raise ValueError(f"pilot manifest missing {exc.args[0]}") from exc
    if manifest.get("run_fingerprint") != fingerprint:
        raise ValueError("pilot run fingerprint mismatch")
    if not set(ARMS).issubset(manifest["providers"]):
        raise ValueError("pilot must contain B0, B1, and B2 arms")

    inputs = _jsonl(pilot_dir / "inputs.jsonl")
    selected = _jsonl(corpus)
    if not inputs or not selected:
        raise ValueError("pilot and corpus must contain nonempty selections")
    try:
        pilot_inputs = {row["id"]: row for row in inputs}
        corpus_rows = {row["id"]: row for row in selected}
    except KeyError as exc:
        raise ValueError(f"selection missing {exc.args[0]}") from exc
    if (len(pilot_inputs) != len(inputs) or len(corpus_rows) != len(selected)
            or not pilot_inputs.keys() <= corpus_rows.keys()):
        raise ValueError("pilot/corpus document identity mismatch or duplicate IDs")
    if manifest["inputs"] != [{"id": row["id"], "group": row["group"],
                                "split": row["split"], "sha256": _text_sha(row["text"])}
                               for row in inputs]:
        raise ValueError("pilot input manifest/hash mismatch")
    hashes: set[str] = set()
    for row in selected:
        doc_id = row["id"]
        if (not isinstance(row.get("text"), str) or not isinstance(row.get("source_sha256"), str)
                or type(row.get("gold_label")) is not int or row["gold_label"] not in range(4)
                or row.get("split") not in ("development", "held-out")):
            raise ValueError(f"invalid AG News corpus record: {doc_id}")
        if "category" in row and row["category"] != LABELS[row["gold_label"]]:
            raise ValueError(f"AG News category/gold label mismatch: {doc_id}")
        digest = _text_sha(row["text"])
        if row["source_sha256"] != digest:
            raise ValueError(f"source hash mismatch: {doc_id}")
        if digest in hashes:
            raise ValueError("duplicate selected original hash across development/test")
        hashes.add(digest)
        if doc_id in pilot_inputs:
            source = pilot_inputs[doc_id]
            if row["text"] != source["text"] or row["split"] != source["split"]:
                raise ValueError(f"pilot/corpus original or split mismatch: {doc_id}")
    if not any(corpus_rows[doc_id]["split"] == "held-out" for doc_id in pilot_inputs):
        raise ValueError("no held-out AG News documents selected")

    results = _jsonl(pilot_dir / "results.jsonl")
    expected = {(doc_id, arm) for doc_id in pilot_inputs for arm in manifest["providers"]}
    pairs: dict[tuple[str, str], dict] = {}
    for row in results:
        key = (row.get("doc_id"), row.get("arm"))
        if key not in expected or key in pairs:
            raise ValueError(f"unexpected or duplicate pilot pair: {key}")
        original = pilot_inputs[key[0]]
        if (row.get("original") != original["text"] or not isinstance(row.get("suggested"), str)
                or row.get("group") != original["group"] or row.get("split") != original["split"]):
            raise ValueError(f"pilot pair/original mismatch: {key}")
        if key[1] == "u" and row["suggested"] != row["original"]:
            raise ValueError(f"unchanged arm differs from original: {key}")
        pairs[key] = row
    if pairs.keys() != expected:
        raise ValueError(f"pilot pairs incomplete: {len(expected - pairs.keys())} missing")
    pilot_selected = [corpus_rows[doc_id] for doc_id in sorted(pilot_inputs)]
    return manifest, [{**row, "suggestions": {arm: pairs[(row["id"], arm)]["suggested"]
                                          for arm in ARMS}} for row in pilot_selected], selected


def _training_rows(train_parquet: Path, selected: list[dict], per_label: int) -> tuple[list[str], list[int], list[str], int]:
    if per_label < 1:
        raise ValueError("train_per_label must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for local AG News Parquet") from exc
    table = parquet.read_table(train_parquet, columns=["text", "label"])
    selected_labels = {row["source_sha256"]: row["gold_label"] for row in selected}
    excluded = set(selected_labels)
    candidates: dict[int, dict[str, str]] = {label: {} for label in range(4)}
    labels_by_hash: dict[str, int] = {}
    ambiguous_hashes: set[str] = set()
    for row in table.to_pylist():
        raw, label = row["text"], row["label"]
        if not isinstance(raw, str) or type(label) is not int or label not in candidates:
            raise ValueError("invalid AG News train row")
        text = raw.strip()  # Same canonical text used by prepare_ag_news.
        digest = _text_sha(text)
        if digest in selected_labels and selected_labels[digest] != label:
            raise ValueError("selected gold label mismatches official AG News train label")
        prior_label = labels_by_hash.setdefault(digest, label)
        if prior_label != label:
            ambiguous_hashes.add(digest)
            candidates[prior_label].pop(digest, None)
        if digest not in excluded and digest not in ambiguous_hashes:
            candidates[label][digest] = text
    chosen = [(digest, candidates[label][digest], label)
              for label in range(4) for digest in sorted(candidates[label])[:per_label]]
    if len(chosen) != 4 * per_label:
        raise ValueError("insufficient distinct non-overlapping AG News train rows per label")
    if any(digest in excluded for digest, _, _ in chosen):
        raise ValueError("training/selection leakage")
    return ([text for _, text, _ in chosen], [label for _, _, label in chosen],
            [digest for digest, _, _ in chosen], len(ambiguous_hashes))


def _state(value):
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "tolist"):
        return _state(value.tolist())
    if isinstance(value, dict):
        return {str(key): _state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_state(item) for item in value]
    return value


def _identity(component) -> dict:
    state = {name: _state(getattr(component, name)) for name in
             ("vocabulary_", "idf_", "classes_", "coef_", "intercept_")
             if hasattr(component, name)}
    return {"class": f"{type(component).__module__}.{type(component).__qualname__}",
            "parameters": _state(component.get_params()) if hasattr(component, "get_params") else {},
            "fitted_state_sha256": _digest(state)}


def evaluate(pilot_dir: Path, train_parquet: Path, corpus: Path, out: Path, *,
             train_per_label: int = 5000, vectorizer_factory=None, classifier_factory=None) -> dict:
    """Validate frozen evidence, fit one model, and score paired held-out texts."""
    pilot_dir, train_parquet, corpus, out = map(Path, (pilot_dir, train_parquet, corpus, out))
    if out.exists():
        raise ValueError("output directory already exists; evidence is not overwritten")
    manifest, selected, corpus_rows = _load_frozen(pilot_dir, corpus)
    train_texts, train_labels, train_hashes, ambiguous_count = _training_rows(
        train_parquet, corpus_rows, train_per_label)
    if vectorizer_factory is None or classifier_factory is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for classification proxy") from exc
        vectorizer_factory = vectorizer_factory or (lambda: TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=50000, sublinear_tf=True))
        classifier_factory = classifier_factory or (lambda: LinearSVC(C=1.0, dual=False, random_state=17))
    vectorizer = vectorizer_factory()
    classifier = classifier_factory()
    classifier.fit(vectorizer.fit_transform(train_texts), train_labels)
    held_out = [row for row in selected if row["split"] == "held-out"]
    original_predictions = [int(x) for x in classifier.predict(
        vectorizer.transform([row["text"] for row in held_out]))]
    predictions_by_arm = {arm: [int(x) for x in classifier.predict(vectorizer.transform(
        [row["suggestions"][arm] for row in held_out]))] for arm in ARMS}
    predictions = []
    for index, row in enumerate(held_out):
        gold, original = row["gold_label"], original_predictions[index]
        predictions.append({"id": row["id"], "source_sha256": row["source_sha256"],
                            "gold_label": gold, "original_prediction": original,
                            "original_correct": original == gold,
                            "arms": {arm: {"prediction": predictions_by_arm[arm][index],
                                           "correct": predictions_by_arm[arm][index] == gold,
                                           "changed": row["suggestions"][arm] != row["text"]}
                                     for arm in ARMS}})
    total = len(predictions)
    base_correct = sum(row["original_correct"] for row in predictions)
    summary = {}
    for arm in ARMS:
        changed = [row for row in predictions if row["arms"][arm]["changed"]]
        correct = sum(row["arms"][arm]["correct"] for row in predictions)
        changed_base = sum(row["original_correct"] for row in changed)
        changed_correct = sum(row["arms"][arm]["correct"] for row in changed)
        summary[arm] = {"documents": total, "original_accuracy": base_correct / total,
                        "suggested_accuracy": correct / total,
                        "paired_accuracy_delta": (correct - base_correct) / total,
                        "changed_only": {"documents": len(changed),
                                         "original_accuracy": changed_base / len(changed) if changed else None,
                                         "suggested_accuracy": changed_correct / len(changed) if changed else None,
                                         "paired_accuracy_delta": (changed_correct - changed_base) / len(changed)
                                         if changed else None}}
    identity = {"pilot_run_fingerprint": manifest["run_fingerprint"],
                "pilot_checksums_sha256": _file_sha(pilot_dir / "checksums.json"),
                "corpus_sha256": _file_sha(corpus), "train_parquet_sha256": _file_sha(train_parquet),
                "train_per_label": train_per_label, "training_text_sha256": train_hashes,
                "ambiguous_train_hashes_excluded": ambiguous_count,
                "training_selection": "first N distinct SHA-256 hashes per label after excluding all selected development and held-out original hashes and ambiguous-label hashes",
                "evaluator_sha256": _file_sha(Path(__file__)),
                "vectorizer": _identity(vectorizer), "classifier": _identity(classifier),
                "sklearn_version": metadata.version("scikit-learn")}
    report = {"schema_version": SCHEMA, "scope": "AG News topic classification proxy; not LLM downstream accuracy",
              "source_license": "unknown; local evaluation only",
              "production_approval": False, "pilot_corpus_label": manifest.get("corpus_label"),
              "identity": identity, "labels": LABELS, "held_out_documents": total,
              "summary": summary, "predictions": predictions,
              "limitations": ["Topic accuracy cannot establish semantic preservation or production safety.",
                              "AG News license is unknown; do not redistribute article text.",
                              "This report is not production approval."]}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                               allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# AG News classification proxy", "", "**Not LLM downstream accuracy or production approval.**",
             "", "AG News license: unknown. Local evaluation only.", "",
             f"Pilot fingerprint: `{manifest['run_fingerprint']}`", "",
             f"Held-out documents: {total}. Training documents: {len(train_texts)}.",
             f"Ambiguous-label training text hashes excluded: {ambiguous_count}.", "",
             "| Arm | Original accuracy | Suggested accuracy | Paired delta | Changed documents | Changed-only delta |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, metrics in summary.items():
        changed = metrics["changed_only"]
        delta = changed["paired_accuracy_delta"]
        lines.append(f"| {arm.upper()} | {metrics['original_accuracy']:.2%} | "
                     f"{metrics['suggested_accuracy']:.2%} | {metrics['paired_accuracy_delta']:+.2%} | "
                     f"{changed['documents']} | {delta:+.2%} |" if delta is not None else
                     f"| {arm.upper()} | {metrics['original_accuracy']:.2%} | "
                     f"{metrics['suggested_accuracy']:.2%} | {metrics['paired_accuracy_delta']:+.2%} | "
                     "0 | n/a |")
    lines += ["", "Raw paired predictions and model/vectorizer identity: [report.json](report.json).", "",
              "This topic classification proxy cannot establish semantic preservation or production safety."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True, help="Frozen lexical pilot run directory")
    parser.add_argument("--train", type=Path, required=True, help="Local official AG News train Parquet")
    parser.add_argument("--corpus", type=Path, required=True, help="Frozen AG News JSONL with id/text/split/gold_label/source_sha256")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--train-per-label", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.pilot, args.train, args.corpus, args.out,
                          train_per_label=args.train_per_label)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"AG News classification proxy: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / 'report.md'} for {report['held_out_documents']} held-out documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
