"""Score a complete subset of frozen IFEval pilot responses with Google's evaluator.

Run ``python -m eval.ifeval_paired --run RUN --gold GOLD.jsonl
--responses RESPONSES.jsonl --evaluator-path PARENT --out NEW_DIR``.
This command reads stored responses; it never calls a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .semantic_benchmark import _read_json, _read_jsonl, _source
from .prepare_ifeval import split_for_key


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive_identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _gold_rows(path: Path, pilot: dict[str, dict]) -> dict[str, dict]:
    provenance = _read_json(path.parent / "provenance.json")
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != "ifeval-frozen-v1"
            or not isinstance(provenance.get("file_sha256"), dict)
            or provenance["file_sha256"].get("gold.jsonl") != _sha(path.read_bytes())
            or not isinstance(provenance.get("source_sha256"), str)
            or len(provenance["source_sha256"]) != 64):
        raise ValueError("frozen IFEval provenance or gold file SHA mismatch")
    gold = {}
    for row in _read_jsonl(path):
        key = row.get("key")
        doc_id = row.get("id")
        if type(key) is not int or doc_id != f"ifeval-{key}":
            raise ValueError("gold id must equal ifeval-{key}")
        if doc_id in gold:
            raise ValueError(f"duplicate gold id: {doc_id}")
        prompt, instructions, kwargs = (row.get(name) for name in
                                         ("prompt", "instruction_id_list", "kwargs"))
        if (not isinstance(prompt, str) or not isinstance(instructions, list)
                or not instructions or any(not isinstance(x, str) or not x for x in instructions)
                or not isinstance(kwargs, list) or len(kwargs) != len(instructions)
                or any(not isinstance(x, dict) for x in kwargs)):
            raise ValueError(f"invalid original IFEval constraints: {doc_id}")
        original_record = {name: row[name] for name in
                           ("key", "prompt", "instruction_id_list", "kwargs")}
        digest = _sha(json.dumps(original_record, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False).encode("utf-8"))
        if (row.get("record_sha256") != digest
                or row.get("source_sha256") != provenance["source_sha256"]):
            raise ValueError(f"gold record or source SHA mismatch: {doc_id}")
        gold[doc_id] = row
    if provenance.get("records") != len(gold):
        raise ValueError("frozen IFEval provenance record count mismatch")
    frozen_prompts = None
    prompts_sha = provenance["file_sha256"].get("prompts.jsonl")
    if prompts_sha is not None:
        prompts_path = path.parent / "prompts.jsonl"
        if _sha(prompts_path.read_bytes()) != prompts_sha:
            raise ValueError("frozen IFEval prompts file SHA mismatch")
        frozen_prompts = {row["id"]: row for row in _read_jsonl(prompts_path)}
        if len(frozen_prompts) != len(gold):
            raise ValueError("frozen IFEval prompt count mismatch")
    for doc_id, source in pilot.items():
        if doc_id not in gold or gold[doc_id]["prompt"] != source["original"]:
            raise ValueError(f"gold original prompt mismatch or missing: {doc_id}")
        if (provenance.get("split_rule") is not None
                and source["split"] != split_for_key(gold[doc_id]["key"])):
            raise ValueError(f"frozen IFEval split mismatch: {doc_id}")
        if frozen_prompts is not None:
            prompt_row = frozen_prompts.get(doc_id)
            if (not isinstance(prompt_row, dict)
                    or prompt_row.get("text") != source["original"]
                    or prompt_row.get("split") != source["split"]):
                raise ValueError(f"frozen IFEval prompt manifest mismatch: {doc_id}")
    return gold


def _response_rows(path: Path, pilot: dict[str, dict], arms: list[str]) -> tuple[dict, dict]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("responses must select at least one document")
    selected = {}
    identity = None
    for row in rows:
        doc_id, arm = row.get("doc_id"), row.get("arm")
        if doc_id not in pilot or arm not in arms:
            raise ValueError(f"unknown response document or extra arm: {doc_id}/{arm}")
        pair = (doc_id, arm)
        if pair in selected:
            raise ValueError(f"duplicate response pair: {doc_id}/{arm}")
        proposed = pilot[doc_id]["arms"][arm]["suggested"]
        if row.get("prompt_sha256") != _sha(proposed.encode("utf-8")):
            raise ValueError(f"response prompt SHA mismatch: {doc_id}/{arm}")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"response must be text: {doc_id}/{arm}")
        _count(row.get("model_input_tokens"), "model_input_tokens")
        _count(row.get("output_tokens"), "output_tokens")
        model = {name: _positive_identity(row.get(name), name)
                 for name in ("model_id", "model_revision")}
        decoding = row.get("decoding")
        if not isinstance(decoding, dict) or not decoding:
            raise ValueError("decoding must be a nonempty object")
        # Round-trip rejects non-JSON values supplied by an in-process caller.
        model["decoding"] = decoding
        encoded = json.dumps(model, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if identity is None:
            identity = encoded
        elif encoded != identity:
            raise ValueError("model identity or decoding settings differ across responses")
        selected[pair] = row
    doc_ids = {doc_id for doc_id, _ in selected}
    for doc_id in doc_ids:
        missing = set(arms) - {arm for found_id, arm in selected if found_id == doc_id}
        if missing:
            raise ValueError(f"missing response arms for {doc_id}: {sorted(missing)}")
    return selected, json.loads(identity)


def _generation_manifest(path: Path, manifest_path: Path, run_fingerprint: str,
                         selected: dict, arms: list[str], model: dict,
                         pilot: dict[str, dict]) -> tuple[str, set[str], dict]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("invalid generation manifest")
    files, selected_ids = manifest.get("files"), manifest.get("selected_doc_ids")
    model_record, settings, truncation = (manifest.get(name) for name in
                                          ("model", "settings", "truncation"))
    if (not isinstance(files, dict) or not isinstance(selected_ids, list)
            or any(not isinstance(doc, str) for doc in selected_ids)
            or len(selected_ids) != len(set(selected_ids))
            or not isinstance(model_record, dict) or not isinstance(settings, dict)
            or not isinstance(truncation, list)):
        raise ValueError("invalid generation manifest fields")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "ifeval-generate-v1"
            or files.get("responses.jsonl") != _sha(path.read_bytes())
            or manifest.get("source_run_fingerprint") != run_fingerprint
            or type(manifest.get("rows")) is not int or manifest["rows"] != len(selected)
            or set(selected_ids) != {doc for doc, _ in selected}
            or manifest.get("arms") != arms
            or model_record.get("id") != model["model_id"]
            or model_record.get("revision") != model["model_revision"]
            or json.dumps(settings.get("decoding"), sort_keys=True, allow_nan=False)
               != json.dumps(model["decoding"], sort_keys=True, allow_nan=False)):
        raise ValueError("generation manifest does not match responses or frozen run")
    requested_split = settings.get("split")
    if requested_split is not None and any(pilot[doc_id]["split"] != requested_split
                                           for doc_id in selected_ids):
        raise ValueError("generation manifest split does not match selected documents")
    changed_arm = settings.get("changed_arm")
    if changed_arm is not None and set(arms) != {"u", changed_arm}:
        raise ValueError("generation changed_arm does not match selected arms")
    truncated = {}
    for row in truncation:
        if (not isinstance(row, dict) or type(row.get("truncated")) is not bool
                or (row.get("doc_id"), row.get("arm")) not in selected):
            raise ValueError("invalid generation truncation list")
        pair = row["doc_id"], row["arm"]
        if pair in truncated:
            raise ValueError("duplicate generation truncation pair")
        truncated[pair] = row["truncated"]
    if (set(truncated) != set(selected) or type(manifest.get("truncated_rows")) is not int
            or manifest["truncated_rows"] != sum(truncated.values())):
        raise ValueError("generation truncation list does not cover response pairs")
    max_new_tokens = model["decoding"].get("max_new_tokens")
    if type(max_new_tokens) is int and max_new_tokens > 0:
        for pair, response in selected.items():
            count = response["output_tokens"]
            if count > max_new_tokens or (truncated[pair] and count < max_new_tokens):
                raise ValueError(f"generation truncation and output token count disagree: {pair}")
    excluded = {doc_id for (doc_id, _), value in truncated.items() if value}
    return _sha(manifest_path.read_bytes()), excluded, {
        "model": model_record, "settings": settings, "truncated_rows": sum(truncated.values())}


def _official(path: Path):
    package = path / "instruction_following_eval"
    library = package / "evaluation_lib.py"
    if not package.is_dir() or not library.is_file():
        raise ValueError("--evaluator-path must be the parent of instruction_following_eval")
    # A caller can evaluate multiple local checkouts in one process.
    for name in list(sys.modules):
        if name == "instruction_following_eval" or name.startswith("instruction_following_eval."):
            del sys.modules[name]
    search_paths = [str(path.resolve())]
    local_deps = path.parent / "deps"
    if local_deps.is_dir():
        search_paths.append(str(local_deps.resolve()))
    old_path = sys.path[:]
    sys.path[:0] = search_paths
    try:
        importlib.invalidate_caches()
        evaluator = importlib.import_module("instruction_following_eval.evaluation_lib")
        if Path(evaluator.__file__).resolve() != library.resolve():
            raise ValueError("official evaluator import path does not match hashed source")
        langdetect = sys.modules.get("langdetect")
        if langdetect is not None:
            langdetect.DetectorFactory.seed = 0
    finally:
        sys.path[:] = old_path
    return evaluator, {str(file.relative_to(package)): _sha(file.read_bytes())
                       for file in sorted(package.rglob("*.py"))}


def _rate(passed: int, total: int) -> float | None:
    return passed / total if total else None


def _metrics(rows: list[dict], arm: str) -> dict:
    chosen = [row for row in rows if row["arm"] == arm]
    # Baseline model-input tokens are stored on each paired result.
    before = sum(row["baseline_input_tokens"] for row in chosen)
    after = sum(row["model_input_tokens"] for row in chosen)
    slices = {}
    for label, subset in (("all_selected", chosen),
                          ("changed_only", [row for row in chosen if row["changed_by_any"]])):
        modes = {}
        for mode in ("strict", "loose"):
            passed = sum(row[mode]["prompt_pass"] for row in subset)
            instruction_passes = sum(sum(row[mode]["instruction_passes"]) for row in subset)
            instruction_total = sum(len(row[mode]["instruction_passes"]) for row in subset)
            modes[mode] = {
                "prompt_passes": passed, "prompt_total": len(subset),
                "prompt_pass_rate": _rate(passed, len(subset)),
                "instruction_passes": instruction_passes,
                "instruction_total": instruction_total,
                "instruction_pass_rate": _rate(instruction_passes, instruction_total),
                "regressions_vs_u": sum(row[mode]["baseline_prompt_pass"] and
                                        not row[mode]["prompt_pass"] for row in subset),
                "recoveries_vs_u": sum(not row[mode]["baseline_prompt_pass"] and
                                       row[mode]["prompt_pass"] for row in subset),
            }
        slices[label] = modes
    return {"arm": arm, "documents": len(chosen),
            "changed_documents": sum(row["changed"] for row in chosen),
            "common_changed_documents": sum(row["changed_by_any"] for row in chosen),
            "model_input_tokens": after,
            "baseline_model_input_tokens": before, "model_input_tokens_saved": before - after,
            "model_input_token_savings_rate": _rate(before - after, before),
            "output_tokens": sum(row["output_tokens"] for row in chosen),
            "slices": slices}


def evaluate(run: Path, gold_path: Path, responses_path: Path, evaluator_path: Path,
             *, generation_manifest: Path | None = None) -> tuple[dict, list[dict]]:
    """Validate all evidence before scoring original IFEval instructions."""
    run, gold_path, responses_path, evaluator_path = map(Path,
                                                          (run, gold_path, responses_path, evaluator_path))
    manifest, _, pilot_rows = _source(run)
    if not (run / "checksums.json").is_file():
        raise ValueError("pilot checksums.json is required")
    pilot_arms = manifest["configuration"]["arms"]
    if "u" not in pilot_arms:
        raise ValueError("pilot requires unchanged u arm for paired comparison")
    arms = pilot_arms
    if generation_manifest is not None:
        generation_record = _read_json(Path(generation_manifest))
        selected_arms = generation_record.get("arms") if isinstance(generation_record, dict) else None
        if (not isinstance(selected_arms, list)
                or any(not isinstance(arm, str) for arm in selected_arms)
                or "u" not in selected_arms
                or len(selected_arms) < 2
                or len(selected_arms) != len(set(selected_arms))
                or selected_arms != [arm for arm in pilot_arms if arm in selected_arms]):
            raise ValueError("generation arms must be an ordered pilot subset including u and a comparison arm")
        arms = selected_arms
    pilot = {}
    for row in pilot_rows:
        doc = pilot.setdefault(row["doc_id"], {"original": row["original"],
                                               "split": row["split"], "arms": {}})
        doc["arms"][row["arm"]] = row
    gold = _gold_rows(gold_path, pilot)
    for doc_id, row in pilot.items():
        if row["arms"]["u"]["suggested"] != gold[doc_id]["prompt"]:
            raise ValueError(f"baseline arm u modified original prompt: {doc_id}")
    responses, model = _response_rows(responses_path, pilot, arms)
    response_documents = len(responses) // len(arms)
    cap = model["decoding"].get("max_new_tokens")
    potentially_capped = (sum(row["output_tokens"] >= cap for row in responses.values())
                          if type(cap) is int and cap > 0 else None)
    generation_sha, excluded, generation = None, set(), None
    if generation_manifest is not None:
        generation_sha, excluded, generation = _generation_manifest(
            responses_path, Path(generation_manifest), manifest["run_fingerprint"],
            responses, arms, model, pilot)
    if excluded:
        responses = {pair: row for pair, row in responses.items() if pair[0] not in excluded}
    if not responses:
        raise ValueError("all selected documents were truncated; nothing can be scored")
    evaluator, code_hashes = _official(evaluator_path)
    scores = []
    for doc_id in sorted({doc_id for doc_id, _ in responses}):
        original = gold[doc_id]
        changed_by_any = any(pilot[doc_id]["arms"][arm]["suggested"] != original["prompt"]
                             for arm in arms if arm != "u")
        example = evaluator.InputExample(key=original["key"], prompt=original["prompt"],
                                         instruction_id_list=original["instruction_id_list"],
                                         kwargs=original["kwargs"])
        evaluated = {}
        for arm in arms:
            row = responses[doc_id, arm]
            answer = row["response"]
            mode_scores = {}
            for mode in ("strict", "loose"):
                function = getattr(evaluator, f"test_instruction_following_{mode}")
                result = function(example, {original["prompt"]: answer})
                bits = result.follow_instruction_list
                if (not isinstance(bits, (list, tuple)) or len(bits) != len(original["instruction_id_list"])
                        or any(type(bit) is not bool for bit in bits)
                        or type(result.follow_all_instructions) is not bool):
                    raise ValueError(f"official evaluator returned invalid {mode} result")
                mode_scores[mode] = {"prompt_pass": result.follow_all_instructions,
                                     "instruction_passes": list(bits)}
            evaluated[arm] = mode_scores
        for arm in arms:
            row = responses[doc_id, arm]
            scores.append({
                "doc_id": doc_id, "split": pilot[doc_id]["split"], "arm": arm,
                "changed": pilot[doc_id]["arms"][arm]["suggested"] != original["prompt"],
                "changed_by_any": changed_by_any,
                "prompt_sha256": row["prompt_sha256"],
                "model_input_tokens": row["model_input_tokens"],
                "baseline_input_tokens": responses[doc_id, "u"]["model_input_tokens"],
                "output_tokens": row["output_tokens"],
                **{mode: {**evaluated[arm][mode],
                          "baseline_prompt_pass": evaluated["u"][mode]["prompt_pass"]}
                   for mode in ("strict", "loose")},
            })
    splits = sorted({row["split"] for row in scores})
    report = {
        "schema_version": "ifeval-paired-v1", "status": "scored_stored_responses",
        "selected_documents": len(scores) // len(arms), "selected_pairs": len(scores),
        "response_documents": response_documents,
        "potentially_capped_rows": potentially_capped,
        "excluded_truncated_doc_ids": sorted(excluded) if generation_manifest is not None else None,
        "excluded_truncated_count": len(excluded) if generation_manifest is not None else None,
        "changed_subset_definition": "Selected documents where any scored non-u arm changed the original prompt; the same documents are used for every scored arm.",
        "pilot_documents": len(pilot), "source_run_fingerprint": manifest["run_fingerprint"],
        "source_hashes": {
            "pilot_checksums": {name: _sha((run / name).read_bytes()) for name in
                                ("manifest.json", "summary.json", "results.jsonl")},
            "gold_jsonl_sha256": _sha(gold_path.read_bytes()),
            "gold_provenance_sha256": _sha((gold_path.parent / "provenance.json").read_bytes()),
            "responses_jsonl_sha256": _sha(responses_path.read_bytes()),
            "generation_manifest_sha256": generation_sha,
            "official_evaluator_py_sha256": code_hashes,
        },
        "model": model, "generation": generation,
        "langdetect_seed": 0 if "langdetect" in sys.modules else None,
        "pilot_arms": pilot_arms, "scored_arms": arms,
        "arms": [_metrics(scores, arm) for arm in arms],
        "splits": {split: [_metrics([row for row in scores if row["split"] == split], arm)
                            for arm in arms] for split in splits},
        "limitations": [
            "Responses were supplied externally; this command did not run a model.",
            "Scores describe only selected complete document sets, not the whole pilot.",
            "A local model run is a feasibility proxy, not production-model quality.",
            "Model-input token counts include the supplied chat template and are taken from response evidence.",
            ("Documents with any truncated arm were excluded from all paired metrics."
             if generation_manifest is not None else
             "No generation manifest was supplied; response truncation is unverified."
             + (f" {potentially_capped} rows reached the decoding token cap."
                if potentially_capped is not None else "")),
        ],
    }
    return report, scores


def write_report(report: dict, scores: list[dict], out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                 allow_nan=False) + "\n", encoding="utf-8")
    (out / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False,
                                                          allow_nan=False) + "\n" for row in scores),
                                       encoding="utf-8")
    excluded_text = (str(report["excluded_truncated_count"])
                     if report["excluded_truncated_count"] is not None
                     else "unverified (no generation manifest)")
    lines = ["# Paired IFEval", "", f"Selected documents: {report['selected_documents']}",
             f"Excluded truncated documents: {excluded_text}",
             f"Model: {report['model']['model_id']} @ {report['model']['model_revision']}", "",
             "Changed-only is the same set of documents changed by any scored comparison arm for every row, including u.", "",
             "| Arm | Split | Own changes | Common changed | Input tokens | Saved vs u | Strict all | Strict changed | Loose all | Loose changed |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, arms in (("all", report["arms"]), *report["splits"].items()):
        for arm in arms:
            s = arm["slices"]
            def cell(mode, subset):
                metric = s[subset][mode]
                return (f"{metric['prompt_passes']}/{metric['prompt_total']}"
                        if metric["prompt_total"] else "n/a")
            lines.append(f"| {arm['arm']} | {split} | {arm['changed_documents']} | "
                         f"{arm['common_changed_documents']} | {arm['model_input_tokens']} | "
                         f"{arm['model_input_tokens_saved']} | {cell('strict', 'all_selected')} | "
                         f"{cell('strict', 'changed_only')} | {cell('loose', 'all_selected')} | "
                         f"{cell('loose', 'changed_only')} |")
    lines.extend(["", "Instruction pass rates and paired regressions/recoveries are in report.json.",
                  "No model was run by this evaluator.", "", "## Limitations", ""])
    lines.extend(f"- {line}" for line in report["limitations"])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "gold", "responses", "evaluator-path", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("output must be a new directory; refusing overwrite")
        report, scores = evaluate(args.run, args.gold, args.responses, args.evaluator_path,
                                  generation_manifest=args.generation_manifest)
        write_report(report, scores, args.out)
    except (OSError, ValueError, ImportError, AttributeError, TypeError, LookupError) as exc:
        print(f"paired IFEval: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
