"""Re-evaluate a tree revision against the frozen original answers and previous tree outputs."""
import argparse
import json
import random
import sqlite3
from pathlib import Path
from importlib.metadata import version

from .ifeval_paired import _official
from .ifeval_workflows import generate, model_input, prepare
from .ifeval_workflow_report import report, write_evidence


def main():
    from mlx_lm.utils import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(".cache/semantic-ifeval"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = {r["id"]: r for r in json.loads((args.baseline / "scores.json").read_text())}
    settings = json.loads((args.baseline / "settings.json").read_text())
    cases, gold, fingerprint = prepare(args.base)
    if fingerprint != settings["source"] or set(previous) != {r["id"] for r in cases}:
        raise ValueError("The baseline and revision must use the same frozen pilot and prompts")
    if (settings["backend"] != "mlx-lm" or settings["dtype"] != "float16"
            or settings["greedy"] is not True
            or any(version(name) != value for name, value in settings["versions"].items())):
        raise ValueError("Revision generation must match the baseline runtime")
    evaluator, evaluator_hashes = _official(args.base / "official")
    if evaluator_hashes != json.loads((args.baseline / "evaluator-hashes.json").read_text()):
        raise ValueError("Evaluator sources differ from the baseline")
    snapshot = Path(settings["snapshot"])
    if not snapshot.is_dir():
        raise ValueError("The original model snapshot must be available locally")
    with sqlite3.connect((args.baseline / "answers.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        answers = {text: (response, bool(capped)) for text, response, capped in
                   db.execute("SELECT prompt,response,capped FROM answers")}
    for row in previous.values():
        if any(text not in answers for text in row["prompts"].values()):
            raise ValueError("The baseline answer cache is incomplete")
    texts = {r["prompts"]["tree"] for r in cases}
    unseen = sorted(texts - answers.keys(), key=lambda text: (len(text), text))
    args.out.mkdir(exist_ok=True, parents=True)
    answers.update(generate(unseen, snapshot, args.out, settings["max_new_tokens"],
                            settings["batch_size"], fingerprint))
    tokenizer = load_tokenizer(snapshot)
    for case in cases:
        doc = case["id"]
        prior = previous[doc]
        if any(prior["prompts"][arm] != case["prompts"][arm] for arm in ("original", "semantic")):
            raise ValueError(f"Control prompt changed for {doc}")
        case["scores"] = dict(prior["scores"])
        case["model_tokens"] = dict(prior["model_tokens"])
        text = case["prompts"]["tree"]
        example = evaluator.InputExample(**{k: gold[doc][k] for k in
                                            ("key", "prompt", "kwargs", "instruction_id_list")})
        answer, capped = answers[text]
        scores = {"capped": capped}
        for metric in ("strict", "loose"):
            random.seed(0)
            check = getattr(evaluator, f"test_instruction_following_{metric}")
            scores[metric] = bool(check(example, {example.prompt: answer}).follow_all_instructions)
        case["scores"]["tree"] = scores
        case["model_tokens"]["tree"] = len(model_input(tokenizer, text))
    write_evidence(cases, args.out / "paired-scores.csv")
    (args.out / "summary.md").write_text(report(cases))
    (args.out / "scores.json").write_text(json.dumps(cases, indent=2))
    print(f"Evaluated {len(cases)} pairs, {len(unseen)} distinct answers outside the baseline cache")


if __name__ == "__main__":
    main()
