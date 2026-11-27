"""Summarize paired scores from eval.ifeval_workflows as Markdown."""
import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest


def comparison(rows, arm, metric):
    pairs = [(r["scores"]["original"][metric], r["scores"][arm][metric]) for r in rows]
    kept = pairs.count((True, True))
    lost = pairs.count((True, False))
    gained = pairs.count((False, True))
    size = len(rows)
    if not size:
        return "0 | — | — | — | — | — | —"
    # Simultaneous exact binomial bounds for the two discordant pair probabilities.
    # Bonferroni gives at least 95% joint coverage, including at zero discordances.
    loss = binomtest(lost, size).proportion_ci(confidence_level=0.975)
    gain = binomtest(gained, size).proportion_ci(confidence_level=0.975)
    interval = (100 * (gain.low - loss.high), 100 * (gain.high - loss.low))
    return (f"{size} | {kept + lost} | {kept + gained} | {kept}/{kept + lost} | "
            f"{lost} | {gained} | {100 * (gained - lost) / size:+.2f} "
            f"[{interval[0]:+.2f}, {interval[1]:+.2f}]")


def report(rows):
    if not rows:
        raise ValueError("Paired report requires at least one scored prompt")
    lines = ["# Held-out IFEval paired results", "",
             "Token counts use o200k_base on the user prompt. A strict pass means the answer "
             "satisfied every original instruction checked by IFEval.", "",
             "| Method | Prompts | Edited | Original tokens | Proposed tokens | Saved | Original answers capped | Method answers capped |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ("tree", "semantic"):
        before = sum(row["tokens"]["original"] for row in rows)
        after = sum(row["tokens"][arm] for row in rows)
        changed = sum(row["prompts"][arm] != row["prompts"]["original"] for row in rows)
        capped = {key: sum(row["scores"][key]["capped"] for row in rows) for key in ("original", arm)}
        lines.append(f"| {arm} | {len(rows)} | {changed} | {before:,} | {after:,} | "
                     f"{100 * (before - after) / before:.2f}% | {capped['original']} | {capped[arm]} |")
    lines += ["", "| Method | Qwen input tokens including chat template | Saved versus original |",
              "| --- | ---: | ---: |"]
    original_count = sum(r["model_tokens"]["original"] for r in rows)
    for arm in ("original", "tree", "semantic"):
        count = sum(r["model_tokens"][arm] for r in rows)
        lines.append(f"| {arm} | {count:,} | {original_count - count:,} "
                     f"({100 * (original_count - count) / original_count:.2f}%) |")
    for metric in ("strict", "loose"):
        lines += ["", f"## {metric.capitalize()} instruction following", "",
                  "| Method | Sample | Pairs | Original passes | Method passes | Baseline passes retained | Lost | Gained | Change in percentage points [95% interval] |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ("tree", "semantic"):
            changed = [r for r in rows if r["prompts"]["original"] != r["prompts"][arm]]
            uncapped = [r for r in rows if not any(r["scores"][k]["capped"] for k in ("original", arm))]
            edited_uncapped = [r for r in uncapped if r["prompts"]["original"] != r["prompts"][arm]]
            shared_uncapped = [r for r in rows if not any(v["capped"] for v in r["scores"].values())]
            for label, group in (("All", rows), ("Edited", changed), ("Both uncapped", uncapped),
                                 ("Edited and both uncapped", edited_uncapped), ("All three uncapped", shared_uncapped)):
                lines.append(f"| {arm} | {label} | {comparison(group, arm, metric)} |")
    lines += ["", "Intervals combine two 97.5% exact binomial intervals for paired losses and gains using Bonferroni bounds. "
              "The all-prompt rows include answers that reached the generation cap. "
              "Each uncapped row excludes a pair if either answer reached the cap. "
              "Edited and pairwise uncapped subsets are specific to each method. Capping can itself depend on compression, so uncapped results are descriptive. Loose checks also allow specified formatting normalizations."]
    return "\n".join(lines) + "\n"


def write_evidence(rows, path):
    """Export prompt IDs and measurements without redistributing model answers."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for row in rows:
            values = {"id": row["id"]}
            for arm in ("original", "tree", "semantic"):
                values[f"{arm}_changed"] = int(row["prompts"][arm] != row["prompts"]["original"])
                values[f"{arm}_tokens"] = row["tokens"][arm]
                values[f"{arm}_model_tokens"] = row["model_tokens"][arm]
                for key in ("capped", "strict", "loose"):
                    values[f"{arm}_{key}"] = int(row["scores"][arm][key])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=values, lineterminator="\n")
                writer.writeheader()
            writer.writerow(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--csv", type=Path, help="Also export per-prompt measurements")
    args = parser.parse_args()
    results = json.loads(args.scores.read_text())
    if args.csv:
        write_evidence(results, args.csv)
    print(report(results), end="")
