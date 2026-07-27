from pathlib import Path

from eval.tasks import grade_exact, grade_f1, QAItem, save_gold, load_gold


def test_exact_match_normalizes():
    assert grade_exact("  $2,500.  ", "$2,500") == 1.0
    assert grade_exact("thirty days", "30 days") == 0.0


def test_f1_partial_overlap():
    assert grade_f1("the quick brown fox", "quick brown fox") > 0.8
    assert grade_f1("completely different", "the quick brown fox") == 0.0


def test_gold_cache_roundtrip_and_fixed(tmp_path):
    items = [QAItem(question="How much?", answer="$2,500"),
             QAItem(question="When?", answer="thirty days")]
    assert load_gold("doc1", tmp_path) is None
    save_gold("doc1", items, tmp_path)
    loaded = load_gold("doc1", tmp_path)
    assert loaded == items
    # save must NOT overwrite an existing gold set (fixed across runs)
    save_gold("doc1", [QAItem(question="x", answer="y")], tmp_path)
    assert load_gold("doc1", tmp_path) == items
