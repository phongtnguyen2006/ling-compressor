from promptcomp import compress

DOC = (
    "The committee approved the annual budget in the morning after a long debate, "
    "and the vendor delivered the goods quickly, in good faith, without any delay."
)


def test_compress_is_byte_identical_across_calls(tmp_path):
    r1 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "a.sqlite")
    r2 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "b.sqlite")
    assert r1.compressed == r2.compressed
    assert r1.config_hash == r2.config_hash


def test_idempotence_second_pass_deletes_little(tmp_path):
    r1 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "a.sqlite")
    r2 = compress(r1.compressed, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "b.sqlite")
    # second pass should not delete a large additional fraction
    assert r2.tokens_after >= 0.7 * r1.tokens_after
