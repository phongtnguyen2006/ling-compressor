---
id: SEM-REVIEW-e351d9a2980f
type: review-evidence
status: annotation-summary
tags: [semantic-compression, review]
---

# Lexical review coverage

Source run: `f71a5481946c18c2358ee9c3cc8b62244d1443534e44eaedcc83011b9a148896`

Annotation status only; this report does not approve deployment.

| Arm | Changed outputs reviewed | Accepted / rejected | Unchanged controls reviewed | Tokens saved in accepted outputs |
| --- | ---: | ---: | ---: | ---: |
| u | 0/0 | 0 / 0 | 0/6 | 0 |
| b0 | 0/2 | 0 / 0 | 0/4 | 0 |
| b1 | 0/2 | 0 / 0 | 0/4 | 0 |
| b2 | 0/2 | 0 / 0 | 0/4 | 0 |
| b2_union | 0/2 | 0 / 0 | 0/4 | 0 |

Pending and partial judgments remain in the [report data](report.json).

- [Edit annotations](edit-annotations.jsonl)
- [Complete-prompt annotations](prompt-annotations.jsonl)

## Limitations

- Judgments are supplied annotations; reviewer identity is not authenticated.
- Incomplete and omitted reviews are not passes. Candidate ratings cannot approve composed outputs.
- Shared review IDs are deduplicated within each arm; arms and repeated templates are not independent samples.
- Changed outputs and unchanged controls have separate denominators; no population error rate is estimated.
- Review-accepted savings count only fully accepted changed outputs against all input tokens; this is not net API savings.
- One annotation set is supported; no rater agreement, sampling weighting, confidence interval, or downstream evaluation is inferred.
- Checksums bind artifacts for reproducibility and accidental-edit detection; they are not signed proof of authenticity.
