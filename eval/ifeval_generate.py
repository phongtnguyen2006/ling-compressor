"""Generate local, deterministic model responses for frozen lexical-pilot prompts (T-29).

Run ``python -m eval.ifeval_generate --run PILOT --model-path LOCAL_SNAPSHOT --model-id ID
--model-revision SHA --out NEW_DIR (--doc-ids FILE | --limit N) [--split development]``.
Only local files are used (no download, no API key). Output is raw responses plus a
provenance manifest; nothing here scores or claims response quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from .semantic_benchmark import _source

SCHEMA = "ifeval-generate-v1"
FIELDS = ("doc_id", "arm", "prompt_sha256", "response", "model_id", "model_revision",
          "decoding", "model_input_tokens", "output_tokens")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_doc_ids(path: Path) -> list[str]:
    """Read ids from .jsonl (strings or objects with doc_id/id) or one-per-line text."""
    ids = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if Path(path).suffix == ".jsonl":
            value = json.loads(line)
            if isinstance(value, dict):
                value = value.get("doc_id", value.get("id"))
        else:
            value = line
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}:{number}: invalid doc id")
        ids.append(value)
    if not ids:
        raise ValueError("doc id file is empty")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate doc ids requested")
    return ids


def select_rows(manifest: dict, results: list[dict], *, doc_ids: list[str] | None = None,
                limit: int | None = None, split: str | None = None,
                changed_arm: str | None = None) -> tuple[list[str], list[str], list[dict]]:
    """Return (doc_ids, arms, rows) with only complete doc-arm groups, deterministic order."""
    arms = list(manifest["configuration"]["arms"])
    if changed_arm is not None:
        if changed_arm not in arms:
            raise ValueError(f"changed arm not in pilot: {changed_arm}")
        arms = [arm for arm in arms if arm in ("u", changed_arm)]
        if "u" not in arms or len(arms) != 2:
            raise ValueError("pilot lacks the unchanged arm 'u' for paired generation")
    by_key = {(row["doc_id"], row["arm"]): row for row in results}
    complete = sorted(d for d in {row["doc_id"] for row in results}
                      if all((d, arm) in by_key for arm in arms)
                      and (split is None or by_key[(d, arms[0])]["split"] == split))
    if doc_ids is not None:
        missing = [d for d in doc_ids if d not in complete]
        if missing:
            raise ValueError(f"requested doc ids not available as complete groups: {missing[:5]}")
        chosen = sorted(doc_ids)
    else:
        chosen = complete
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("no documents selected")
    return chosen, arms, [by_key[(d, arm)] for d in chosen for arm in arms]


def load_local_model(model_path: Path):
    """Load pinned local tokenizer/model; never touches the network."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("transformers", "torch"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def generate_one(tokenizer, model, device, prompt: str, max_new_tokens: int):
    """Greedy-generate one response. Returns (text, input_tokens, output_tokens, truncated)."""
    import torch
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device) if "attention_mask" in encoded else None
    kwargs = {"attention_mask": attention} if attention is not None else {}
    with torch.no_grad():
        output = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                **kwargs)
    n_in = int(input_ids.shape[1])
    new_ids = output[0][n_in:].tolist()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    eos = tokenizer.eos_token_id
    truncated = len(new_ids) >= max_new_tokens and (not new_ids or new_ids[-1] != eos)
    return response, n_in, len(new_ids), truncated


def run(args, loader=load_local_model, generator=generate_one) -> dict:
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise ValueError(f"model path is not a local directory: {model_path}")
    if re.fullmatch(r"[0-9a-f]{40}", model_path.name) and model_path.name != args.model_revision:
        raise ValueError("model revision does not match snapshot directory name")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if (args.doc_ids is None) == (args.limit is None):
        raise ValueError("provide exactly one of --doc-ids or --limit")
    manifest, _, results = _source(Path(args.run))
    ids = read_doc_ids(args.doc_ids) if args.doc_ids else None
    doc_ids, arms, rows = select_rows(manifest, results, doc_ids=ids, limit=args.limit,
                                      split=args.split, changed_arm=args.changed_arm)
    tokenizer, model, device = loader(model_path)
    decoding = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    records, truncation = [], []
    for row in rows:
        prompt = row["suggested"]
        response, n_in, n_out, truncated = generator(tokenizer, model, device, prompt,
                                                     args.max_new_tokens)
        records.append({"doc_id": row["doc_id"], "arm": row["arm"],
                        "prompt_sha256": _sha(prompt.encode("utf-8")), "response": response,
                        "model_id": args.model_id, "model_revision": args.model_revision,
                        "decoding": dict(decoding), "model_input_tokens": n_in,
                        "output_tokens": n_out})
        truncation.append({"doc_id": row["doc_id"], "arm": row["arm"], "truncated": bool(truncated)})
    out.mkdir(parents=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    (out / "responses.jsonl").write_text(body, encoding="utf-8")
    out_manifest = {
        "schema_version": SCHEMA,
        "source_run": str(Path(args.run)),
        "source_run_fingerprint": manifest["run_fingerprint"],
        "selected_doc_ids": doc_ids, "arms": arms, "rows": len(records),
        "model": {"path": str(model_path), "id": args.model_id, "revision": args.model_revision,
                  "device": device, "local_files_only": True},
        "versions": _versions(),
        "settings": {"split": args.split, "limit": args.limit,
                     "doc_ids_file": str(args.doc_ids) if args.doc_ids else None,
                     "changed_arm": args.changed_arm, "decoding": decoding},
        "truncation": truncation,
        "truncated_rows": sum(t["truncated"] for t in truncation),
        "files": {"responses.jsonl": _sha(body.encode("utf-8"))},
        "note": "Raw generations only; no quality claim is made.",
    }
    (out / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return out_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--doc-ids", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--split", choices=("development", "held-out"))
    p.add_argument("--changed-arm", help="e.g. b2; generate only arm u and this arm")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args(argv)
    try:
        m = run(args)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {m['rows']} responses to {args.out} ({m['truncated_rows']} truncated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
