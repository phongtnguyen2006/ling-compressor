"""QA task grading (offline) and gold-answer cache; model-gated generation/judge."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(".,;:!?\"'()[] ")


def grade_exact(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def grade_f1(pred: str, gold: str) -> float:
    p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def load_gold(doc_id: str, gold_dir: Path) -> list[QAItem] | None:
    path = Path(gold_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [QAItem(**d) for d in data]


def save_gold(doc_id: str, items: list[QAItem], gold_dir: Path) -> None:
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"{doc_id}.json"
    if path.exists():
        return  # fixed across runs — never overwrite
    path.write_text(json.dumps([asdict(i) for i in items], indent=2))


def generate_gold(text: str, n: int, client) -> list[QAItem]:  # pragma: no cover - model
    raise RuntimeError(
        "generate_gold requires the [anthropic] extra and ANTHROPIC_API_KEY; "
        "run gold generation once, then rely on the cached eval/gold/*.json."
    )
