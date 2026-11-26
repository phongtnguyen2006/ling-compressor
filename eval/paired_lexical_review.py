"""Compare two independent, blinded complete-prompt reviews of a frozen pilot.

Run ``python -m eval.paired_lexical_review --help``. Missing judgments stay pending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import unicodedata

from .lexical_pilot import _digest, _write_json, _write_jsonl
from .lexical_review import (
    JUDGMENTS, _accepted, _merge_annotations, _read_jsonl, _state, summarize_reviews,
)


def _normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _one_reviewer(rows: dict[str, dict], label: str) -> str | None:
    """A sparse packet may be unanswered, but its judgments have one identity."""
    identities = [row["reviewer"] for row in rows.values() if row["reviewer"] is not None]
    if len({_normalized_name(name) for name in identities}) > 1:
        raise ValueError(f"{label} has inconsistent reviewer identities")
    return identities[0] if identities else None


def _counts(ids: set[str], a: dict[str, dict], b: dict[str, dict]) -> dict:
    counts = {"total": len(ids), "pending": 0, "partial": 0, "both_complete": 0,
              "agreement": 0, "disagreement": 0, "any_harmful_true": 0,
              "jointly_accepted": 0}
    for review_id in ids:
        left, right = a[review_id], b[review_id]
        left_state, right_state = _state(left), _state(right)
        if left["harmful_change"] is True or right["harmful_change"] is True:
            counts["any_harmful_true"] += 1
        if left_state == right_state == "completed":
            counts["both_complete"] += 1
            if all(left[key] is right[key] for key in JUDGMENTS):
                counts["agreement"] += 1
            else:
                counts["disagreement"] += 1
            if _accepted(left) and _accepted(right):
                counts["jointly_accepted"] += 1
        elif left_state == right_state == "pending":
            counts["pending"] += 1
        else:
            counts["partial"] += 1
    return counts


def summarize_paired_reviews(run_dir: Path, reviewer_a: Path, reviewer_b: Path) -> dict:
    """Validate a frozen run and both packets, then summarize unique changed outputs."""
    run_dir, reviewer_a, reviewer_b = map(Path, (run_dir, reviewer_a, reviewer_b))
    if reviewer_a.resolve() == reviewer_b.resolve():
        raise ValueError("reviewer A and B must use separate files")

    # This performs the frozen checksum, manifest, result, and export checks.
    frozen = summarize_reviews(run_dir)
    baseline = frozen["annotations"]["prompts"]
    a = _merge_annotations(baseline, reviewer_a)
    b = _merge_annotations(baseline, reviewer_b)
    identity_a, identity_b = _one_reviewer(a, "reviewer A"), _one_reviewer(b, "reviewer B")
    if identity_a is not None and identity_b is not None and _normalized_name(identity_a) == _normalized_name(identity_b):
        raise ValueError("reviewer A and B identities must differ")

    results = _read_jsonl(run_dir / "results.jsonl")
    changed_by_arm: dict[str, set[str]] = {}
    for row in results:
        changed_by_arm.setdefault(row["arm"], set())
        if row["original"] != row["suggested"]:
            review_id = _digest({"doc_id": row["doc_id"], "original": row["original"],
                                 "suggested": row["suggested"]})
            changed_by_arm[row["arm"]].add(review_id)
    changed_ids = set().union(*changed_by_arm.values())
    missing = changed_ids - set(a) | changed_ids - set(b)
    if missing:
        raise ValueError(f"frozen changed output is missing from prompt review export: {sorted(missing)[0]}")
    jointly_accepted_ids = {review_id for review_id in changed_ids
                            if _accepted(a[review_id]) and _accepted(b[review_id])}

    arms = []
    for arm, ids in changed_by_arm.items():
        arm_results = [row for row in results if row["arm"] == arm]
        before = sum(row["tokens_before"] for row in arm_results)
        saved = sum(row["tokens_before"] - row["tokens_after"] for row in arm_results
                    if row["original"] != row["suggested"] and
                    _digest({"doc_id": row["doc_id"], "original": row["original"],
                             "suggested": row["suggested"]}) in jointly_accepted_ids)
        arms.append({
            "arm": arm, "documents": len(arm_results), "tokens_before": before,
            "changed_review_ids": _counts(ids, a, b),
            "tokens_saved_in_jointly_accepted_changed_outputs": saved,
            "jointly_accepted_savings_over_all_input_tokens": saved / before if before else 0.0,
        })

    disagreements = []
    triage = []
    for review_id in sorted(changed_ids):
        left, right = a[review_id], b[review_id]
        both_complete = _state(left) == _state(right) == "completed"
        differs = both_complete and any(left[key] is not right[key] for key in JUDGMENTS)
        reasons = []
        if differs:
            reasons.append("disagreement")
        if left["harmful_change"] is True or right["harmful_change"] is True:
            reasons.append("harmful_flag")
        if (_state(left) == "completed") != (_state(right) == "completed"):
            reasons.append("one_side_incomplete")
        if both_complete and not (_accepted(left) and _accepted(right)):
            reasons.append("both_complete_rejected")
        if reasons:
            item = {
                "review_id": review_id, "doc_id": left["doc_id"],
                "original": left["original"], "suggested": left["suggested"],
                "reasons": reasons,
                "reviewer_a": {"reviewer": left["reviewer"],
                               "notes": left["notes"], **{key: left[key] for key in JUDGMENTS}},
                "reviewer_b": {"reviewer": right["reviewer"],
                               "notes": right["notes"], **{key: right[key] for key in JUDGMENTS}},
            }
            triage.append(item)
            if differs:
                disagreements.append(item)

    evidence = {
        "schema_version": "paired-lexical-review-v1",
        "run_fingerprint": frozen["run_fingerprint"],
        "source_checksums": frozen["source_checksums"],
        "reviewer_a_sha256": hashlib.sha256(reviewer_a.read_bytes()).hexdigest(),
        "reviewer_b_sha256": hashlib.sha256(reviewer_b.read_bytes()).hexdigest(),
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return {
        **evidence, "report_fingerprint": _digest(evidence),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "paired_annotation_summary_not_deployment_approval",
        "reviewers": {"a": identity_a, "b": identity_b},
        "changed_review_ids": _counts(changed_ids, a, b),
        "reviewer_coverage": {
            label: {state: sum(_state(rows[review_id]) == state for review_id in changed_ids)
                    for state in ("pending", "partial", "completed")}
            for label, rows in (("a", a), ("b", b))
        },
        "arms": arms,
        "limitations": [
            "Only changed complete-prompt IDs enter the quality counts; unchanged controls are excluded.",
            "Shared review IDs count once in overall quality counts. Arm savings use each arm's original input-token denominator.",
            "Pending and partial judgments are not passes. Agreement is counted only when both reviews are complete.",
            "Any harmful true includes partial reviews; jointly accepted requires both complete reviews to accept.",
            "Reviewer names are supplied annotations, not authenticated identities.",
            "This is review evidence, not production readiness, downstream quality, or net API savings.",
        ],
        "disagreements": disagreements,
        "triage": triage,
    }


def write_report(report: dict, output_dir: Path, *, run_dir: Path | None = None) -> None:
    output_dir = Path(output_dir)
    if run_dir is not None and output_dir.resolve().is_relative_to(Path(run_dir).resolve()):
        raise ValueError("review output cannot be placed inside the frozen run")
    if output_dir.exists():
        raise ValueError("review output must be a new directory; existing evidence is not overwritten")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "report.json", {key: value for key, value in report.items()
                                             if key not in {"disagreements", "triage"}})
    _write_jsonl(output_dir / "disagreements.jsonl", report["disagreements"])
    _write_jsonl(output_dir / "triage.jsonl", report["triage"])
    counts = report["changed_review_ids"]
    lines = [
        "# Paired lexical review", "", f"Frozen run: `{report['run_fingerprint']}`", "",
        "Review evidence only; pending and partial judgments are not passes or deployment approval.", "",
        f"Unique changed prompts: {counts['total']}; pending: {counts['pending']}; "
        f"partial: {counts['partial']}; both complete: {counts['both_complete']}; "
        f"agreement: {counts['agreement']}; disagreement: {counts['disagreement']}; "
        f"any harmful true: {counts['any_harmful_true']}; jointly accepted: {counts['jointly_accepted']}.",
        "", "| Arm | Changed IDs jointly accepted | Saved tokens | Savings / all input tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in report["arms"]:
        lines.append(f"| {arm['arm']} | {arm['changed_review_ids']['jointly_accepted']}/"
                     f"{arm['changed_review_ids']['total']} | "
                     f"{arm['tokens_saved_in_jointly_accepted_changed_outputs']} | "
                     f"{arm['jointly_accepted_savings_over_all_input_tokens']:.2%} |")
    lines.extend(["", "[Report data](report.json) · [Blinded disagreement queue](disagreements.jsonl) · [Triage queue](triage.jsonl)",
                  "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "checksums.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir()) if path.is_file()
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Frozen lexical pilot directory")
    parser.add_argument("--reviewer-a", type=Path, required=True, help="Sparse blinded prompt-review JSONL")
    parser.add_argument("--reviewer-b", type=Path, required=True, help="Independent sparse blinded prompt-review JSONL")
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    try:
        report = summarize_paired_reviews(args.run, args.reviewer_a, args.reviewer_b)
        write_report(report, args.out, run_dir=args.run)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"paired lexical review: {exc}", file=sys.stderr)
        return 2
    counts = report["changed_review_ids"]
    print(f"Wrote {args.out / 'summary.md'}; {counts['both_complete']}/{counts['total']} "
          "unique changed prompts jointly complete. No deployment approval inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
