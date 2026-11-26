# Two-reviewer prompt fidelity study

The full 400k GloVe sweep proposed more edits, including some that visibly change meaning. This study workflow keeps two independent complete-prompt judgments separate and counts savings only when both reviewers accept the same changed output. It does not fill judgments automatically or approve production rewriting.

## Prepare the blinded packet

With the [frozen 400k pilot](full-vocabulary-sweep-2026-11-25.md) provisioned locally, run:

```bash
python -m eval.lexical_review_packet \
  --run .cache/semantic-real/pilot-dolly-glove-400k-400 \
  --out .cache/semantic-real/paired-review-packet-400k
```

The packet contains `reviewer-a.jsonl` and `reviewer-b.jsonl`. Give each file to a different person. Keep `packet.json` and the pilot's `review-key.jsonl` away from reviewers because they expose study metadata or algorithm attribution. Each reviewer file has the same unique prompt pairs in a different deterministic order. The frozen 400k pilot measured on 2026-11-25 yields **41 distinct changed outputs plus 20 unchanged controls**, or 61 records per reviewer. The original 400 prompts and multiple algorithm arms account for the difference between input count and unique review count. The controls exclude documents that also have changed outputs, so no reviewer sees the same document twice. Reviewers can recognize an unchanged control because its two texts match; the blinding hides the algorithm arm, provider, selection decision, and token savings.

The packet generator validates the frozen pilot's checksums and review exports before writing. The files stay under ignored `.cache/`; prompt text and external vectors are not committed. Use a new output directory for another packet.

## Review visually

Run `python scripts/ui.py`, open `http://localhost:8000/review.html`, enter a reviewer name, and choose your assigned JSONL file. The screen shows the original and proposed complete prompts side by side, with no arm or token count. For each record, mark whether meaning is preserved, whether the proposal is grammatical, and whether it might harm the task. Use **Unsure** when evidence is insufficient. Add notes for possible harm or ambiguity. Export the JSONL file before closing the tab; the browser does not persist in-progress work. Do not exchange files or discuss answers until both independent files are complete.

The JSONL review fields are `meaning_preserved`, `grammatical`, and `harmful_change`, each a JSON Boolean or `null`, plus a reviewer name and notes. The importer rejects changed source text, unknown or duplicate IDs, and invalid fields. It rejects the same normalized name in both files, including case and spacing variants; names are recorded but are not authenticated proof of two different people. The visual screen helps with entry; the CLI performs final evidence validation.

## Compare the two files

After both reviewers export their files, run:

```bash
python -m eval.paired_lexical_review \
  --run .cache/semantic-real/pilot-dolly-glove-400k-400 \
  --reviewer-a /path/to/reviewer-a-completed.jsonl \
  --reviewer-b /path/to/reviewer-b-completed.jsonl \
  --out .cache/semantic-real/paired-review-report-400k
```

The report counts unique changed outputs once for review coverage and disagreement. Per-arm token savings use the original full-input denominator and include only changed outputs that **both** reviewers fully accepted. Blank, partial, and unsure fields do not pass. Review `triage.jsonl` for disagreements, harmful flags, and one-sided incomplete reviews before interpreting quality. Unchanged controls are excluded from the changed-output quality rate. The report is an annotation summary, not a deployment approval or a downstream LLM accuracy test.

For a pipeline smoke check, the two blank packet files can be passed directly to the paired reporter. In the 2026-11-25 400k pilot smoke run, that yielded **0/41** jointly completed changed prompts and **zero** reviewed savings. These are pending records, not negative human findings. No human annotations have been collected yet.

The provisional production gate requires an upper 95% bound below 1% for harmful changed prompts. Even if all 41 changed outputs in this public proxy were reviewed with zero harms, the one-sided binomial upper bound would be about **7.0%** under an independent-sample assumption. This packet alone cannot establish that gate, and the public proxy does not represent the intended customer workload. A larger, representative, predeclared study and adjudication procedure are still needed.

[Project status](STATUS.md) · [Production gates](production-gates.md) · [Usage guide](usage.md)
