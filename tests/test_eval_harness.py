from eval.harness import run_grid, RunResult
from eval.baselines import NoneCompressor, StopwordCompressor, OursCompressor
from promptcomp.tokens import TiktokenCounter


DOCS = {
    "d1": ("The committee approved the annual budget of $2,500 in the morning after a long "
           "debate, and the vendor delivered the goods quickly without any delay."),
}


def test_grid_shape_and_offline_accuracy_is_none():
    results = run_grid(DOCS, [NoneCompressor(), OursCompressor()], [0.4, 0.6], TiktokenCounter())
    assert len(results) == 1 * 2 * 2
    for r in results:
        assert isinstance(r, RunResult)
        assert r.accuracy_exact is None  # no grader -> offline
        assert r.tokens_after <= r.tokens_before


def test_ours_zero_violations_stopword_nonzero_in_grid():
    results = run_grid(DOCS, [OursCompressor(), StopwordCompressor()], [0.4], TiktokenCounter())
    by_name = {r.compressor: r for r in results}
    assert by_name["ours"].protected_violation_count == 0
    assert by_name["stopword"].protected_violation_count > 0


def test_none_baseline_is_ceiling():
    results = run_grid(DOCS, [NoneCompressor()], [0.5], TiktokenCounter())
    assert results[0].compression_ratio == 1.0
    assert results[0].protected_violation_count == 0
