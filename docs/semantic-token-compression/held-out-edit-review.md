---
id: SEM-REVIEW-IFEVAL-400K-001
title: "Held-Out IFEval B2 Edit Inspection (Single-AI)"
type: inspection-note
created: 2026-11-27
status: single-ai-inspection
tags:
  - ling-compressor
  - semantic-compression
  - ifeval
  - edit-review
---

# Held-out IFEval u/b2 edit inspection

> [!IMPORTANT]
> **Single-AI inspection only.** Every judgment below was produced by a single AI reviewer with access to provenance, token counts, and gold IFEval constraints. This is **not** blinded, **not** independent human annotation, and does **not** satisfy the two-reviewer human fidelity gate in [production-gates.md](production-gates.md) or [two-reviewer-study-2026-11-25.md](two-reviewer-study-2026-11-25.md). Raw prompt text stays in ignored `.cache/semantic-ifeval/`.

## Scope and summary

Source: `.cache/semantic-ifeval/pilot-541-400k/results.jsonl` (run fingerprint `abe7060e8e010efccfae6be8661d6e3000cd81fb4db0813dd78c49cdfd1d4a7f`, `tiktoken:o200k_base`, 400k GloVe 50d + WordNet, 8 candidates / 4 embedding slots).

- **Held-out split:** 414 prompts, **18,448** original tokens (`u` arm).
- **Changed `u`/`b2` pairs:** **34 / 414** prompts, **39** selected word edits (**16** WordNet, **23** GloVe embedding), proposing **43** saved tokens (**0.233%** of held-out tokens).
- **Rubric:** Based on [usage.md](usage.md#validate-annotations-and-report-coverage) (`meaning_preserved`, `grammatical`, `harmful_change`):
  - **Accept:** preserves task, facts, scope, and constraints; grammatical; non-harmful.
  - **Borderline:** grammatical and low task-failure risk, but introduces lexical/register drift, slight scope broadening, or awkward idiom (would not pass strict meaning preservation without human adjudication).
  - **Reject:** alters task intent, introduces factual/domain error, produces nonsense/ungrammatical text, or corrupts an executable constraint (including `combination:repeat_prompt`).

| Judgment | Prompts | Edits (WordNet / Embedding) | Proposed tokens saved | Share of 18,448 held-out tokens |
| --- | ---: | ---: | ---: | ---: |
| **Accept** | 3 / 34 | 3 (3 / 0) | 4 | 0.022% |
| **Borderline** | 10 / 34 | 10 (8 / 2) | 10 | 0.054% |
| **Reject** | 21 / 34 | 26 (5 / 21) | 29 | 0.157% |
| **Total changed** | **34 / 34** | **39 (16 / 23)** | **43** | **0.233%** |

By prompt provenance: **WordNet-only** (14 prompts: 3 Accept, 8 Borderline, 3 Reject), **Embedding-only** (19 prompts: 0 Accept, 2 Borderline, 17 Reject), **Mixed** (1 prompt, `ifeval-136`: 0 Accept, 0 Borderline, 1 Reject). None of the 23 GloVe embedding edits yielded a clean acceptance.

## Recurring failure patterns

1. **Static-embedding proper-noun / sense conflation (8 prompts, 9 edits):** `limerick` → `waterford` (6 prompts: Irish city confusion) and `riddle` → `embry` (2 prompts: surname artifact).
2. **Co-hyponym and domain-term substitution (6 prompts, 8 edits):** `protons` → `electrons` (2x), `embankment` → `overpass`, `retriever` → `puppy` + `poodle` → `pug`, `megaliths` → `henges`, `shrines` → `temples`, `projectiles` → `shells`.
3. **Instruction reversal or sense drift (7 prompts, 8 edits):** Embedding `Criticize` → `Ignore` (polarity reversal, overriding B0's `Criticize` → `Knock`), `punchline` → `cliché`, `notations` → `meanings` (2 prompts); WordNet `Pretend` → `Guess`, `Exclude` → `Shut`, `Summarize` → `Resume` (+ `asterisk` → `star`).
4. **`repeat_prompt` constraint breakage (4 prompts):** `ifeval-1281`, `ifeval-288`, `ifeval-3369`, and `ifeval-3371` instruct verbatim repetition of the request; editing text inside the repeated span breaks exact match against original gold constraints.
5. **Lexical/register drift with intact marker (8 prompts, 8 edits):** WordNet `postscript` → `supplement` next to explicit `P.S.` / `P.P.S` markers.

## Per-document review (all 34 changed held-out pairs)

Listed in `results.jsonl` order. Source abbreviations: `emb` = `embedding:vectors:a2c9f52593f1240d`, `wn` = `wordnet:<synset>`.

| # | Doc ID | Saved tok | Edit(s) (`orig -> repl`, source, cos) | Judgment | Reasoning |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `ifeval-1108` | 1 | `hamburgers` -> `burgers` (`emb`, 0.90) | Borderline | Grammatical and colloquially close in "Are hamburgers sandwiches?" (Kannada response), but "burgers" broadens scope beyond beef hamburgers. |
| 2 | `ifeval-1219` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Register/meaning drift ("supplement" is broader than "postscript"), though retained `P.S.` marker keeps the postscript check clear. |
| 3 | `ifeval-1281` | 1 | `limerick` -> `waterford` (`emb`, 0.86) | Reject | Irish-city sense conflation ("Write a waterford"), inconsistent with second unedited `limerick`, and breaks `repeat_prompt` verbatim gold match. |
| 4 | `ifeval-1305` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker mitigates constraint risk. |
| 5 | `ifeval-136` | 3 | `protons` -> `electrons` x2 (`emb`, 0.89); `Elaborate` -> `Expand` (`wn:elaborate.v.01`, 0.24) | Reject | `Elaborate` -> `Expand` is fine, but `protons` -> `electrons` twice makes the physics premise false ("nucleus is a cluster of electrons and neutrons") and contradicts the next line (`A proton is`). |
| 6 | `ifeval-1367` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 7 | `ifeval-1582` | 2 | `Criticize` -> `Ignore` (`emb`, 0.80) | Reject | Direct task reversal ("Ignore this sentence in a funny way") contradicting the next sentence ("Provide exactly two critiques"). |
| 8 | `ifeval-1705` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 9 | `ifeval-1776` | 1 | `reasoning` -> `logic` (`emb`, 0.81) | Borderline | Math problem and quotation constraint remain intact, but "logic process" is less idiomatic and narrower than "reasoning process". |
| 10 | `ifeval-1825` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 11 | `ifeval-19` | 1 | `riddle` -> `embry` (`emb`, 0.73) | Reject | Proper-noun/surname embedding artifact ("a funny embry"); nonsensical, ungrammatical, and destroys the requested genre. |
| 12 | `ifeval-1928` | 2 | `Elaborate` -> `Expand` (`wn:elaborate.v.01`, 0.24) | Accept | "Expand on the following sentence into a formal story" is idiomatic, grammatical, and preserves task intent and all constraints. |
| 13 | `ifeval-2225` | 1 | `embankment` -> `overpass` (`emb`, 0.78) | Reject | Replaces a civil-engineering term paired with "levee" with an unrelated structure ("overpass"), changing the factual question. |
| 14 | `ifeval-227` | 2 | `retriever` -> `puppy` (`emb`, 0.73); `poodle` -> `pug` (`emb`, 0.70) | Reject | Corrupts dog breeds inconsistently across sentences ("golden puppy and a poodle" vs. "pug... golden retriever"), breaking coreference and altering the size comparison answer. |
| 15 | `ifeval-2362` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 16 | `ifeval-2396` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 17 | `ifeval-2497` | 1 | `conquering` -> `conquest` (`wn:conquest.n.01`, 0.70) | Accept | "chatting and conquest" is grammatical and preserves the open-ended topic and formatting constraints (only minor loss of `-ing` parallelism). |
| 18 | `ifeval-2549` | 1 | `postscript` -> `supplement` (`wn:addendum.n.01`, 0.26) | Borderline | Same lexical drift from "postscript" to "supplement"; retained `P.P.S` marker keeps formatting intent recoverable. |
| 19 | `ifeval-2577` | 2 | `megaliths` -> `henges` (`emb`, 0.73) | Reject | Substitutes a narrower, distinct archaeological category (British earthwork enclosures) for European stone "megaliths", altering the factual subject. |
| 20 | `ifeval-2602` | 1 | `limerick` -> `waterford` (`emb`, 0.86) | Reject | Irish-city sense conflation ("Write a waterford"); only mention of "limerick" is erased, making the creative form nonsensical. |
| 21 | `ifeval-2677` | 1 | `limerick` -> `waterford` (`emb`, 0.86) | Reject | Irish-city sense conflation ("Write a waterford"), contradicting the two subsequent unedited occurrences of "limerick". |
| 22 | `ifeval-2785` | 1 | `shrines` -> `temples` (`emb`, 0.84) | Reject | Domain/cultural error: Shinto sites are *shrines* (*jinja*), not Buddhist *temples* (*tera*), shifting the factual topic of the lecture. |
| 23 | `ifeval-281` | 1 | `punchline` -> `cliché` (`emb`, 0.78) | Reject | Replaces a core joke-structure requirement ("punchline") with an unrelated, pejorative concept ("cliché"). |
| 24 | `ifeval-288` | 1 | `projectiles` -> `shells` (`emb`, 0.77) | Reject | Narrows the definition of ballistics from all projectiles to shells, and edits inside a `repeat_prompt` span so exact repetition fails original gold. |
| 25 | `ifeval-2880` | 1 | `notations` -> `meanings` (`emb`, 0.75) | Reject | Alters an explicit content directive ("Use mathematical notations" -> "Use mathematical meanings"), dropping the symbolic-notation requirement. |
| 26 | `ifeval-2957` | 2 | `limerick` -> `waterford` x2 (`emb`, 0.86) | Reject | Irish-city sense conflation applied to both occurrences; completely erases the poetic form and renders the prompt nonsensical. |
| 27 | `ifeval-3188` | 1 | `Pretend` -> `Guess` (`wn:guess.v.02`, 0.79) | Reject | WordNet sense error ("Guess that you are a fortune teller..."); unidiomatic and weakens/distorts the role-play instruction. |
| 28 | `ifeval-32` | 1 | `limerick` -> `waterford` (`emb`, 0.86) | Reject | Irish-city sense conflation ("Write a waterford about writing a limerick"); destroys the self-referential poetic task. |
| 29 | `ifeval-3369` | 1 | `limerick` -> `waterford` (`emb`, 0.86) | Reject | Irish-city sense conflation contradicting the later "limerick", and edits inside a `repeat_prompt` span so verbatim repetition fails original gold. |
| 30 | `ifeval-3371` | 1 | `Exclude` -> `Shut` (`wn:exclude.v.02`, 0.24) | Reject | WordNet sense error ("Shut the words..."); ungrammatical, destroys the forbidden-words constraint, and breaks `repeat_prompt` verbatim gold match. |
| 31 | `ifeval-3453` | 3 | `Summarize` -> `Resume` (`wn:sum_up.v.01`, 0.15); `asterisk` -> `star` (`wn:asterisk.n.01`, 0.14) | Reject | Archaic WordNet sense ("Resume the history of Japan" reads as "continue", not "summarize"); `asterisk` -> `star` is minor given `*italic*` example, but the prompt as a whole fails. |
| 32 | `ifeval-3623` | 1 | `riddle` -> `embry` (`emb`, 0.73) | Reject | Same surname/embedding artifact as `ifeval-19` ("Write a short embry"); nonsensical, ungrammatical, and destroys the requested form. |
| 33 | `ifeval-3629` | 1 | `notations` -> `meanings` (`emb`, 0.75) | Reject | Changes "using notations" to "using meanings", making the continuation instruction awkward and semantically different. |
| 34 | `ifeval-3690` | 1 | `Critique` -> `Review` (`wn:review.v.02`, 0.52) | Accept | "Review the following ad copy" is natural, grammatical, and preserves the ad-copy evaluation task and `<<title>>` constraint (mild softening of critical tone). |

## Takeaway

In this single-AI inspection of all 34 changed held-out `u`/`b2` pairs, **21/34 (61.8%)** are clear rejections, **10/34 (29.4%)** are borderline, and only **3/34 (8.8%)** — all WordNet edits saving 4 of 18,448 held-out tokens (0.022%) — are clean acceptances. Static 50d GloVe discovery (B2) accounts for 18 of the 21 rejected prompts (17 embedding-only plus mixed `ifeval-136`) and 0 acceptances, consistent with the current no-go status for automatic vector-based rewriting in [STATUS.md](STATUS.md). Formal evidence still requires blinded, two-reviewer human annotation.
