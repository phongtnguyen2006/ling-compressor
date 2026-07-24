from __future__ import annotations

import re

from .types import Block, BlockKind

_URL_RE = re.compile(r"^\s*https?://\S+\s*$")
_PATH_RE = re.compile(r"^\s*(/[^\s/]+){2,}/?\s*$")
_STACK_RE = re.compile(r'^\s*(File \"|at\s+\S+\(|Traceback|\s+at\s)')
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_lines_keepends(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _is_digit_heavy(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    digits = sum(c.isdigit() for c in stripped)
    return digits / len(stripped) > 0.30


def _is_nonalpha_heavy(line: str) -> bool:
    non_space = [c for c in line if not c.isspace()]
    if not non_space:
        return False
    nonalpha = sum(not c.isalpha() for c in non_space)
    return nonalpha / len(non_space) > 0.40


def _looks_like_table_row(line: str, lines: list[str], idx: int) -> bool:
    if "|" not in line:
        return False
    # a table needs a separator row somewhere adjacent
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(lines) and _TABLE_SEP_RE.match(lines[j]):
            return True
    return bool(_TABLE_SEP_RE.match(line))


def _classify_line(line: str, lines: list[str], idx: int) -> BlockKind:
    if _URL_RE.match(line) or _PATH_RE.match(line) or _STACK_RE.match(line):
        return BlockKind.PASSTHROUGH
    if _looks_like_table_row(line, lines, idx):
        return BlockKind.PASSTHROUGH
    if _is_digit_heavy(line) or _is_nonalpha_heavy(line):
        return BlockKind.PASSTHROUGH
    return BlockKind.PROSE


def segment(text: str) -> list[Block]:
    if text == "":
        return []

    lines = _split_lines_keepends(text)
    # First pass: classify each line, honoring fenced-code state.
    kinds: list[BlockKind] = []
    in_fence = False
    for idx, line in enumerate(lines):
        is_fence_delim = line.lstrip().startswith("```")
        if in_fence:
            kinds.append(BlockKind.PASSTHROUGH)
            if is_fence_delim:
                in_fence = False
            continue
        if is_fence_delim:
            in_fence = True
            kinds.append(BlockKind.PASSTHROUGH)
            continue
        kinds.append(_classify_line(line, lines, idx))

    # Second pass: coalesce adjacent same-kind lines into offset-preserving blocks.
    blocks: list[Block] = []
    pos = 0
    run_start = 0
    run_kind = kinds[0]
    run_text: list[str] = []
    for line, kind in zip(lines, kinds):
        if kind != run_kind and run_text:
            joined = "".join(run_text)
            blocks.append(Block(run_kind, joined, run_start, run_start + len(joined)))
            run_start = run_start + len(joined)
            run_kind = kind
            run_text = []
        run_text.append(line)
        pos += len(line)
    if run_text:
        joined = "".join(run_text)
        blocks.append(Block(run_kind, joined, run_start, run_start + len(joined)))
    return blocks
