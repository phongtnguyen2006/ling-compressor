"""Compare standalone tree pruning and frozen semantic proposals on held-out IFEval."""
import argparse
import json
import random
import sqlite3
from pathlib import Path

from promptcomp import compress
from promptcomp.lexical.tokenization import local_counter
from .ifeval_paired import _gold_rows, _official, _sha
from .prepare_ifeval import split_for_key
from .semantic_benchmark import _source


def prepare(base):
    pilot = base / "pilot-541-400k"
    if not (pilot / "checksums.json").is_file():
        raise ValueError("Frozen semantic pilot checksums are required")
    manifest, _, rows = _source(pilot)
    arms = {(row["doc_id"], row["arm"]): row for row in rows}
    originals = {row["doc_id"]: row for row in rows
                 if row["arm"] == "u" and row["split"] == "held-out"}
    gold = _gold_rows(base / "frozen-541/gold.jsonl", originals)
    expected = {doc for doc, row in gold.items() if split_for_key(row["key"]) == "held-out"}
    if set(originals) != expected:
        raise ValueError("Pilot does not cover the complete frozen held-out split")
    counter = local_counter("o200k_base")
    cases = []
    for doc, row in originals.items():
        original = row["original"]
        if original != row["suggested"]:
            raise ValueError(f"Unchanged arm altered original prompt for {doc}")
        tree = compress(original, target_ratio=0.9, substitute=False,
                        min_tokens=0, counter=counter).compressed
        texts = dict(original=original, tree=tree, semantic=arms[doc, "b2"]["suggested"])
        cases.append(dict(id=doc, prompts=texts, tokens={arm: counter.count(text)
                          for arm, text in texts.items()}))
    return cases, gold, manifest["run_fingerprint"]


def model_input(tokenizer, text):
    message = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                            tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(message, add_special_tokens=False)


def generate(texts, snapshot, out, cap, batch_size, fingerprint):
    from importlib.metadata import version
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import batch_generate

    identity = dict(snapshot=str(snapshot.resolve()), max_new_tokens=cap, batch_size=batch_size,
                    source=fingerprint, backend="mlx-lm", dtype="float16", greedy=True,
                    versions={name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
                    prompts_sha256=_sha(json.dumps(texts).encode()))
    config = out / "settings.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Resume requires matching prompts and generation settings")
    if not config.exists():
        if (out / "answers.sqlite").exists():
            raise ValueError("Missing settings for existing answers")
        with config.open("x") as stream:
            json.dump(identity, stream, indent=2)
    with sqlite3.connect(out / "answers.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS answers (prompt TEXT PRIMARY KEY, response TEXT, capped INTEGER)")
        answers = {text: (answer, bool(capped)) for text, answer, capped in db.execute("SELECT * FROM answers")}
        model = None
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if all(text in answers for text in batch):
                continue
            if model is None:
                model, tokenizer = load(str(snapshot.resolve()))
                model.set_dtype(mx.float16)
                mx.eval(model.parameters())
                stops = json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"]
                tokenizer.eos_token_ids = stops if isinstance(stops, list) else [stops]
            ids = [model_input(tokenizer, text) for text in batch]
            output = batch_generate(model, tokenizer, ids, max_tokens=cap, return_token_ids=True)
            for text, generated in zip(batch, output.token_ids, strict=True):
                answer = tokenizer.decode(generated, skip_special_tokens=True)
                # MLX removes stop tokens from the token-ID result.
                capped = len(generated) == cap
                db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (text, answer, capped))
                answers[text] = answer, capped
            db.commit()
            print(f"Generated {len(answers)}/{len(texts)}", flush=True)
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(".cache/semantic-ifeval"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.batch_size) < 1:
        parser.error("batch size and token cap must be positive")
    if not args.snapshot.is_dir():
        parser.error("--snapshot must point to local model weights")
    args.out.mkdir(exist_ok=True, parents=True)
    cases, gold, fingerprint = prepare(args.base)
    evaluator, evaluator_hashes = _official(args.base / "official")
    import nltk
    nltk.data.find("tokenizers/punkt_tab/english/")
    texts = sorted({text for case in cases for text in case["prompts"].values()}, key=lambda s: (len(s), s))
    answers = generate(texts, args.snapshot, args.out, args.max_new_tokens, args.batch_size, fingerprint)
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(args.snapshot)
    lengths = {text: len(model_input(tokenizer, text)) for text in texts}
    for case in cases:
        case["model_tokens"] = {arm: lengths[text] for arm, text in case["prompts"].items()}
        row = gold[case["id"]]
        example = evaluator.InputExample(**{k: row[k] for k in ("key", "prompt", "kwargs", "instruction_id_list")})
        case["scores"] = {}
        for arm, text in case["prompts"].items():
            answer, capped = answers[text]
            score = dict(capped=capped)
            for metric in ("strict", "loose"):
                random.seed(0)
                check = getattr(evaluator, f"test_instruction_following_{metric}")
                score[metric] = bool(check(example, {row["prompt"]: answer}).follow_all_instructions)
            case["scores"][arm] = score
    (args.out / "scores.json").write_text(json.dumps(cases, indent=2))
    (args.out / "evaluator-hashes.json").write_text(json.dumps(evaluator_hashes, indent=2))


if __name__ == "__main__":
    main()
