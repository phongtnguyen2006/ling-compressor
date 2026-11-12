---
id: SEM-USAGE-001
title: "Lexical Opportunity Pilot Usage"
aliases:
  - Semantic Compression Pilot Usage
type: usage-guide
created: 2026-10-07
updated: 2026-11-11
status: experimental
tags:
  - ling-compressor
  - semantic-compression
  - implementation
  - reproducibility
---

# Lexical opportunity pilot usage

## Unified workbench and benchmark report

Install `python -m pip install -e '.[lexical]'`, then run `python scripts/ui.py`
from the repository root and open `http://localhost:8000`. Select **Semantic
substitution** in the same workbench that contains tree pruning and guarded
LLMLingua-2. The bundled vectors are synthetic and labeled as such. Point the
semantic controls at a locally provisioned GloVe text file for a real vector
lookup. The tokenizer cache and spaCy model still require the explicit setup
below. The local workbench accepts vector files up to 2 GB and validates the
entire file when first loaded; later runs reuse its cached index. No assets are
downloaded by the workbench during a run.

### Review and reuse an exact prompt template

The semantic workbench presents proposed substitutions as **pending**. For a
repeated, authored prompt template, inspect each occurrence and mark it
**Accept** or **Reject**. The preview recounts the complete text under the
selected tokenizer. **Save Approved Template** requires every proposal to have
an explicit decision, at least one accepted substitution, and a positive
complete-text token saving. Export the reviewed text and audit JSON from the
same panel. Saved records live only in ignored `.cache/approved_templates/`.

This workflow is for exact repeated text. The saved record binds the original
content, tokenizer, candidate configuration, and vector asset identity.
`POST /api/semantic/validate` checks those bindings and returns the original
input unchanged if validation fails. For an authored template containing
placeholders, validate the template **before** filling the placeholders, then
fill the approved output in the application. If the source template wording
changes, review and save that revised template separately. The private
approval and template endpoints are available only on a loopback-bound local
workbench. Use the displayed provenance and sentence context to judge each
edit; the software's cosine and grammar checks do not approve meaning.

The two public proxy pilots in the [decision report](real-vector-pilot-2026-11-11.md)
do not justify automatic vector rewriting. This limited workflow creates
value only for templates that a person reviews and reuses enough times to
offset review and execution cost. The workbench does not measure that return
on investment or claim suitability as a network-facing service.

Summarize a frozen pilot run without downloading a model:

```bash
python -m eval.semantic_benchmark \
  --run docs/semantic-token-compression/evidence/2026-10-07-lexical-smoke \
  --out eval/results/semantic-benchmark-demo
```

The reporter validates the pilot manifest, results, summary, and checksums when
present. Its JSON and Markdown outputs show token reduction and changed-output
coverage. Semantic quality remains `not_computed` by default. To add a sentence
embedding cosine **proxy**, install `.[semantic-eval]` and pass `--model-path`
with an existing local SentenceTransformer directory. The reporter enables
offline model loading; cosine does not replace human review or downstream task
accuracy.

### Reproduce the real-vector prompt pilot locally

The [2026-11-11 pilot report](real-vector-pilot-2026-11-11.md) used the
[Gensim data GloVe 50d mirror](https://github.com/piskvorky/gensim-data/blob/master/list.json)
and [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
The source files and frozen prompt text belong under ignored `.cache/`, not in
the repository. Provision the sources explicitly, then run:

```bash
mkdir -p .cache/semantic-real
curl -fL https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz -o .cache/semantic-real/glove-wiki-gigaword-50.gz
gunzip -c .cache/semantic-real/glove-wiki-gigaword-50.gz > .cache/semantic-real/glove-wiki-gigaword-50.txt
curl -fL https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl -o .cache/semantic-real/databricks-dolly-15k.jsonl
python -m eval.prepare_dolly --source .cache/semantic-real/databricks-dolly-15k.jsonl --out .cache/semantic-real/dolly-400
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
export NLTK_DATA="$PWD/.cache/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" wordnet
OPENBLAS_NUM_THREADS=1 python -m eval.lexical_pilot \
  --corpus .cache/semantic-real/dolly-400/prompts.jsonl \
  --out .cache/semantic-real/pilot-dolly-glove-10k-400 \
  --corpus-label 'Databricks Dolly-15k balanced 400 instruction/context proxy; CC BY-SA 3.0' \
  --arms u,b0,b1,b2 --encoding o200k_base --wordnet \
  --vectors .cache/semantic-real/glove-wiki-gigaword-50.txt \
  --vector-label 'Gensim data mirror of Stanford GloVe Wiki+Gigaword 6B 50d; first 10000 source-order vectors retained' \
  --max-words 10000 --max-candidates 8 --embedding-slots 4
python -m eval.semantic_decision \
  --run .cache/semantic-real/pilot-dolly-glove-10k-400 \
  --out .cache/semantic-real/decision-dolly
```

Each output path must be new; the tools refuse to overwrite frozen evidence.
The public prompt corpus is a proxy for a real production workload. The
separate AG News classification proxy requires locally provisioned Parquet
files and `pyarrow`; its dataset card lists the license as unknown.

[Documentation index](README.md) · [Implementation and evidence record](implementation.md) · [Agreed experiment plan](debate-plan.md)

The offline SEM-M1 engine produces **unreviewed suggestions** and evidence for an opportunity study. The local workbench adds a reviewed exact-template workflow. Token savings, cosine scores, and mechanical guards do not establish semantic equivalence. The existing deletion compressor does not invoke this engine automatically.

**Traceability:** SEM-REQ-002 through SEM-REQ-010 and SEM-REQ-013; SEM-DEC-001 through SEM-DEC-004 and SEM-DEC-006. The [implementation record](implementation.md#implementation-and-verification-evidence) distinguishes verified software behavior from the unstarted representative study, SEM-M2.

## Set up local assets

Use Python 3.12 or later and run these commands from the repository root in your selected virtual environment. Installation and asset provisioning may access the network; they are explicit setup steps, separate from running the pilot.

```bash
python -m pip install -e '.[lexical,dev]'
python -m spacy download en_core_web_sm
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

The pilot also supports `cl100k_base`. If selecting it, provision that exact encoding explicitly:

```bash
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

The built-in [local tokenizer loader](../../src/promptcomp/lexical/tokenization.py) blocks a cache miss from downloading an asset. A missing tokenizer cache or spaCy model produces an error rather than a silently changed configuration. Initialize and reuse the tokenizer during application setup. An API caller supplying its own provider or counter is responsible for that component's I/O behavior.

WordNet is optional. To run the complete curated-plus-WordNet baseline, provision the NLTK corpus and set `--wordnet`. This example keeps the corpus in the ignored project cache and configures the same absolute path during both download and execution:

```bash
export NLTK_DATA="$PWD/.cache/nltk_data"
mkdir -p "$NLTK_DATA"
python -m nltk.downloader -d "$NLTK_DATA" wordnet
```

Run the later `--wordnet` command in that shell, or set the same `NLTK_DATA` value again in a new shell. Setting it before the downloader also makes the custom location visible to NLTK versions that validate permitted data paths. NLTK's Python package alone does not supply WordNet. A requested but missing corpus is an error; omitting `--wordnet` explicitly runs and labels a **curated-only baseline**.

For a real embedding experiment, supply a locally obtained GloVe text file or word2vec text file with an optional `rows dimensions` header. Record its training provenance in `--vector-label`. The loader records a SHA-256 hash of the complete source file, validates every row, and retains up to `--max-words` rows. Thus lowering `--max-words` reduces retained matrix size but does not skip file validation or hashing. Retrieval uses exact cosine scoring with deterministic lexical tie-breaking. Multiword embedding composition, approximate search, FastText binaries, and Sentence-Transformer checkpoints are not implemented here.

Do not substitute the bundled synthetic vector file for trained embeddings in a research claim. Keep large external assets outside tracked source files; this repository does not automatically download them.

## Run the bundled software demonstration

The bundled manifest contains six authored smoke inputs, including repeated phrases, concise text, protected literals, and deliberately chosen vector examples. The small vector file is synthetic. Neither is a representative corpus or a semantic-quality benchmark.

```bash
python -m eval.lexical_pilot \
  --corpus data/lexical-demo/prompts.jsonl \
  --out eval/results/lexical-demo-001 \
  --corpus-label "Six authored software smoke inputs; not representative" \
  --vectors data/lexical-demo/synthetic-vectors.txt \
  --vector-label "Synthetic software fixture; not trained embeddings" \
  --synthetic-vectors \
  --arms u,b0,b1,b2,b2_union \
  --max-candidates 8 \
  --embedding-slots 4 \
  --encoding o200k_base
```

This command intentionally omits WordNet. Its `b0` arm is curated-only, as the manifest and summary limitations state. If the output directory already contains any entry, choose a fresh run name such as `lexical-demo-002`. The runner refuses nonempty directories and existing files; it does not overwrite prior evidence. An empty existing directory is allowed. Preserve a partially written run for diagnosis and use a fresh directory when retrying.

The `eval/results/` directory is ignored by Git. Copy only intentionally selected, reviewable evidence into a versioned evidence location and link it from the implementation record. The same no-overwrite rule applies when `--out` is a path under `docs/`.

To inspect all supported options:

```bash
python -m eval.lexical_pilot --help
```

## Define a representative corpus before candidate discovery

Acceptable inputs are a directory of UTF-8 `.txt` files or a JSONL manifest. A directory uses filename stems as IDs and groups, with `development` as the default split. A manifest gives explicit provenance labels:

```json
{"id":"template-001","group":"support-templates","split":"development","text":"They worked in order to finish the assignment."}
{"id":"template-002","group":"support-templates","split":"held-out","text":"Write a short answer."}
```

Each record requires a unique nonempty string `id` and a string `text`; `group` defaults to the ID and `split` to `development`. Empty text is permitted and contributes zero tokens. Inputs are sorted by ID. The CLI optionally samples with `--max-documents` and `--seed` before candidate discovery. `group` and `split` are metadata: the runner does not create statistically independent splits or automatically filter to one split. Supply the intended frozen subset yourself.

For example, after preparing the named local files:

```bash
python -m eval.lexical_pilot \
  --corpus /absolute/path/to/preselected-prompts.jsonl \
  --out eval/results/opportunity-001 \
  --corpus-label "Preselected English prompt study; development split" \
  --wordnet \
  --vectors /absolute/path/to/glove.6B.100d.txt \
  --vector-label "GloVe 6B, 100 dimensions; document acquisition provenance" \
  --arms u,b0,b1,b2,b2_union \
  --max-words 50000 \
  --max-candidates 8 \
  --embedding-slots 4 \
  --encoding o200k_base
```

Freeze the workload, preprocessing, split definitions, candidate budgets, assets, and tokenizer before comparing runs. Do not select inputs after discovering which ones compress well. The run manifest records identities and declared labels; it cannot verify the truth of a corpus label or establish representativeness.

## What each arm measures

CLI arm names are lowercase. All substitution arms use the same editing engine, mechanical guards, and one-selected-edit-per-sentence limit.

| Arm | Candidate construction | Interpretation |
| --- | --- | --- |
| `u` | Original text, no candidate provider | Unchanged reference |
| `b0` | Curated rules plus optional WordNet; curated candidates first, then lexical ordering; at most `max_candidates` per source/POS query | Lexical baseline; curated-only when WordNet is disabled |
| `b1` | The same complete baseline candidate membership as B0, with curated candidates first and cosine ranking within source class; at most `max_candidates` after ranking | Ranking experiment; vectors do not generate extra candidates |
| `b2` | Baseline ranked as B1, plus vector neighbors absent from the complete baseline; separate fixed slot allocations | Additional-discovery experiment under a common total candidate limit |
| `b2_union` | Entire baseline plus up to `embedding_slots` additional vector neighbors | Diagnostic with no baseline cap; not a fair-budget competitor |

For the default budget of eight candidates with four embedding slots, B2 offers at most four baseline candidates and four additional neighbors for each query. Empty slots are not borrowed by the other source. B2 requires at least one slot of each kind. B2 union removes the baseline cap but retains the extra-neighbor limit; it is not an exhaustive search over every vector.

B0 and B1 share their **pre-cap membership**, not necessarily their offered subset or order after truncation. Missing vectors do not remove baseline candidates; their similarity is `null`, and they rank after candidates with a score within the same source class. Curated candidates retain priority. `candidate-pools.jsonl` records complete baseline pools and offered candidates so these distinctions can be checked.

The engine selects the largest measured complete-text token reduction among candidates passing guards in a sentence. Provider rank breaks equal-saving ties before occurrence and lexical tie-breakers. Consequently, B1 ranking matters when the cap changes candidate inclusion or a tie occurs; it is not a semantic validator. Candidate budgets apply to a source/POS query, not to an entire prompt, and the arms can have different actual evaluation work.

Phrase and word edit counts are reported separately. These counts alone do not partition token savings by edit type. Use the occurrence audit and an explicit analysis when attributing gains to phrases, WordNet, or additional embeddings.

## Use the Python API

```python
from dataclasses import asdict
import json

from promptcomp.lexical import PilotProvider, suggest
from promptcomp.lexical.tokenization import local_counter

counter = local_counter("o200k_base")
provider = PilotProvider("b0", use_wordnet=False)
text = "They worked in order to finish the assignment."

result = suggest(text, provider, counter=counter)
print(result.suggested)
print(result.tokens_before, result.tokens_after)
print(json.dumps(asdict(result), indent=2))

# Freeze an exact original occurrence using Python character offsets.
start = text.index("in order to")
protected_result = suggest(
    text,
    provider,
    counter=counter,
    protected_spans=[(start, start + len("in order to"))],
)
```

To use a local vector asset with the API:

```python
from promptcomp.lexical import PilotProvider, VectorIndex, suggest
from promptcomp.lexical.tokenization import local_counter

vectors = VectorIndex.from_text("/absolute/path/to/glove.6B.100d.txt", max_words=50000)
provider = PilotProvider(
    "b2", vectors=vectors, use_wordnet=True,
    max_candidates=8, embedding_slots=4,
)
result = suggest("They worked in order to finish the assignment.", provider,
                 counter=local_counter("o200k_base"))
```

For generated embedding neighbors, an optional cosine floor can be supplied to
`suggest(..., min_embedding_similarity=0.8, adaptive_similarity=True)`. The
adaptive exploratory rule adds 0.05 to the floor for a one-token gain and 0.05
for a noun or verb edit, capped at 1.0. Candidates lacking a score are rejected
when the floor is active. The rule is recorded in the configuration hash and
rejection audit; its numerical values have not been validated as a fidelity
threshold. Curated and WordNet baseline candidates are outside this filter.

The [API result](../../src/promptcomp/lexical/engine.py) contains original and suggested text, complete-prompt counts, selected edits, all candidate decisions and rejection reasons, provenance, timing, a configuration hash, warnings, and `review_status="pending"`. The API returns data; the [pilot runner](../../eval/lexical_pilot.py) adds corpus-level manifests, files, and summaries.

Offsets always refer to the original Python string, not UTF-8 bytes or positions in already edited output. The engine supports exactly one selected edit per sentence. It rejects overlapping or protected edits, checks supported case/POS/morphology/dependency properties, recounts complete text after each selected edit, and rolls back if final invariants fail. These conservative guards can abstain on a whole sentence. They cannot establish that a replacement means the same thing.

## Inspect the evidence and conduct review

Every successful pilot run writes these files:

| File | Contents and use |
| --- | --- |
| `summary.md` | Obsidian-readable run note with a stable fingerprint-derived ID, arm table, artifact links, and limitations |
| `manifest.json` | Input hashes, run fingerprint, UTC creation time, declared workload/configuration, provider and asset metadata, dependency versions, software hashes, and limitations |
| `inputs.jsonl` | Exact frozen input texts with IDs, groups, and splits |
| `results.jsonl` | One document/arm result: composed text, occurrence-level selections/rejections, counts, warnings, and timing |
| `candidate-pools.jsonl` | Queried baseline pools and offered candidates, with source, POS, document, and arm; empty pools are omitted |
| `summary.json` | All-input aggregate counts, changed-input coverage, word/phrase and provenance counts, single-pass timing summaries, and pending quality/review status |
| `edit-review.jsonl` | Deduplicated, arm-blinded positive-token-delta proposals, including some mechanically rejected proposals; human judgment fields remain `null` |
| `prompt-review.jsonl` | Deduplicated, arm-blinded complete outputs, including unchanged outputs; human judgment fields remain `null` |
| `review-key.jsonl` | Separate attribution mapping from review IDs to arms, decisions, and available provenance; keep away from blinded reviewers |
| `checksums.json` | SHA-256 checksums for the other output files; it does not include itself |

An edit-review row is not necessarily a selected edit or a valid paraphrase. Consult the separate attribution key after review. Positive measured gain can coexist with a later grammar rejection. Reviewing both candidate edits and composed prompts is necessary because local judgments do not prove whole-prompt fidelity.

Save human annotations as a separate versioned review artifact with reviewer identity, rubric, and date. Do not replace `null` with an inferred judgment from cosine or token savings. Avoid editing raw generated evidence in place; doing so invalidates its recorded checksums.

Corpus reduction is `1 - sum(tokens_after) / sum(tokens_before)`, including unchanged inputs. Median per-document reduction is a different quantity. Empty inputs remain in document counts and contribute zero tokens. The timings are a single sequential pass with cold/warm effects; they are not isolated performance measurements. Timestamps and timings can change between runs even when selections are deterministic. Compare configuration, input and asset identities, selected outputs, and declared environments instead of expecting byte-identical complete run directories.

## Validate annotations and report coverage

The [review reporter](../../eval/lexical_review.py) accepts edited **copies** of the two review forms. It verifies the original run's checksums, manifest fingerprint, unique document/arm records, and reconstructed review subjects/attribution key before importing judgments. Keep the original run intact.

For a coverage-only report with every judgment still unanswered:

```bash
python -m eval.lexical_review \
  --run docs/semantic-token-compression/evidence/2026-10-07-lexical-smoke \
  --out eval/results/review-coverage-001
```

For a human review, copy `edit-review.jsonl` and `prompt-review.jsonl` into a separate review directory. Reviewers may change only `meaning_preserved`, `grammatical`, `harmful_change`, `reviewer`, and `notes`. Each judgment is a JSON Boolean (`true` or `false`) or `null` for unanswered. Any supplied judgment requires a nonblank reviewer identity. `notes` must remain a string. Keep review IDs and all source text/offsets unchanged. The importer rejects unknown or duplicate IDs, changed subjects, missing/extra fields, and invalid types.

Use this initial rubric and record any study-specific refinements before collecting judgments:

| Field | Review question |
| --- | --- |
| `meaning_preserved` | Does the suggested text retain the original instruction, facts, scope, relationships, and relevant nuance? |
| `grammatical` | Is the suggestion grammatically appropriate in the supplied context? |
| `harmful_change` | Could a change alter the requested task, an important fact, a constraint, or the intended interpretation? |
| `reviewer` | Which person supplied these judgments? This identifier is recorded, not authenticated. |
| `notes` | Why was a change accepted/rejected, or why is a judgment uncertain? Leave uncertain judgments `null`. |

Then submit the copies:

```bash
python -m eval.lexical_review \
  --run docs/semantic-token-compression/evidence/2026-10-07-lexical-smoke \
  --edit-reviews /absolute/path/to/review/edit-review.jsonl \
  --prompt-reviews /absolute/path/to/review/prompt-review.jsonl \
  --out eval/results/review-annotated-001
```

Either annotation file may be omitted, and a file may contain a subset of its original rows. Missing subjects remain pending. A partial row remains partial until all three judgments are answered. A completed review is accepted only when meaning and grammar are `true` and harmful change is `false`. Reviewers should assess complete outputs separately from candidate edits; candidate acceptance never supplies a missing prompt judgment.

The new output directory contains `report.json`, `summary.md`, merged `edit-annotations.jsonl` and `prompt-annotations.jsonl`, and `checksums.json`. The report binds itself to the frozen run and submitted file hashes, preserves pending/partial/completed counts, separates changed outputs from unchanged controls, and attributes candidate and selected-edit judgments separately. A review shared across arms is counted once within each arm; those arm observations are correlated.

`tokens_saved_in_review_accepted_outputs` includes only fully accepted **changed complete outputs**. Its all-input fraction uses all original tokens, including unreviewed and unchanged inputs. A value of zero with all reviews pending means no savings have yet been approved by review, not that every proposal failed. It does not estimate net API savings or semantic quality in the population. No confidence intervals, sampling weights, downstream accuracy, multi-rater agreement, or deployment approval are inferred.

This importer supports one annotation set per report. A representative study still needs a predeclared sampling plan, reviewer instructions, and a separate record of disagreements/adjudication if multiple people participate. Checksums detect accidental file changes; they are not signed proof of authenticity.

## Limits and next evidence gate

SEM-M1 demonstrates that the software can generate, constrain, count, and export substitution proposals. The synthetic smoke run cannot establish GloVe quality, incremental value over a complete WordNet baseline, task accuracy, semantic preservation, representative savings, or net API cost reduction. No Sentence-Transformer validator or downstream LLM evaluation runs in this prototype.

SEM-M2 still requires roughly 200 preselected representative prompts, stratified human review of roughly 200–400 candidate edits plus complete outputs, documented sampling and rubric, and a continue/stop decision. Those pilot sizes are planning estimates, not statistical reliability certification. Follow the [debate plan](debate-plan.md) before moving to contextual validation or automatic use.

## Dated maintenance

### 2026-10-07 — Usage and reproduction guide added

Documented the implemented local API/CLI, asset setup, candidate-arm distinctions, evidence retention, and review handoff. Verification outcomes belong in [SEM-IMPL-001](implementation.md); this guide does not substitute for test or study evidence.
