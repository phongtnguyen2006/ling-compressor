from eval.ifeval_workflow_report import report


def test_report_counts_lost_and_gained_passes_even_when_overall_score_is_tied():
    rows = []
    for i, (original, tree, capped) in enumerate([(True, False, True), (False, True, False)]):
        rows.append({
            "id": str(i),
            "prompts": {"original": "abc", "tree": "ab", "semantic": "abc"},
            "model_tokens": {"original": 9, "tree": 8, "semantic": 9},
            "tokens": {"original": 3, "tree": 2, "semantic": 3},
            "scores": {arm: {"strict": passed, "loose": passed, "capped": arm == "tree" and capped}
                       for arm, passed in [("original", original), ("tree", tree), ("semantic", original)]},
        })
    result = report(rows)
    assert "| tree | All | 2 | 1 | 1 | 0/1 | 1 | 1 | +0.00" in result
    assert "| tree | Both uncapped | 1 | 0 | 1 |" in result
    assert "| semantic | Edited | 0 |" in result
    assert "| semantic | All | 2 | 1 | 1 | 1/1 | 0 | 0 |" in result
