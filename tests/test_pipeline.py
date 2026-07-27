from promptcomp.pipeline import compress


def test_short_text_passes_through_unchanged():
    text = "This is a short instruction under the size gate."
    result = compress(text, min_tokens=500)
    assert result.compressed == text
    assert result.ratio == 1.0
    assert result.deleted == ()
    assert result.token_counter.startswith("tiktoken:")


def test_prose_substitution_applies_above_gate(tmp_path):
    # Force the gate open with min_tokens=0 so short fixtures still compress.
    text = "We proceeded in order to finish, due to the fact that time was short."
    result = compress(text, min_tokens=0, cache_path=tmp_path / "c.sqlite")
    assert "in order to" not in result.compressed
    assert "due to the fact that" not in result.compressed
    assert result.tokens_after <= result.tokens_before
    assert "in order to" in result.substitutions


def test_passthrough_block_is_byte_identical(tmp_path):
    text = "Explanation prose here in order to demonstrate.\n```\nx = in order to\n```\n"
    result = compress(text, min_tokens=0, cache_path=tmp_path / "c.sqlite")
    # The code fence content must be untouched even though it contains the phrase.
    assert "x = in order to" in result.compressed


def test_substitute_disabled_is_identity(tmp_path):
    text = "We did this in order to win the contract."
    # target_ratio=1.0 zeroes the per-block deletion budget so this test
    # isolates the substitute flag: deletion now runs independently of
    # substitution as of the Plan B3 extractive pipeline (Task 4).
    result = compress(
        text, substitute=False, target_ratio=1.0, min_tokens=0,
        cache_path=tmp_path / "c.sqlite"
    )
    assert result.compressed == text


def test_determinism_same_input_same_output(tmp_path):
    text = "We proceeded in order to finish the report on time today."
    r1 = compress(text, min_tokens=0, cache_path=tmp_path / "c1.sqlite")
    r2 = compress(text, min_tokens=0, cache_path=tmp_path / "c2.sqlite")
    assert r1.compressed == r2.compressed
    assert r1.config_hash == r2.config_hash


def test_deletes_clean_adjunct_below_ratio(tmp_path):
    text = (
        "The committee approved the annual budget in the morning after a long debate. "
        "The vendor delivered the goods quickly."
    )
    result = compress(text, target_ratio=0.6, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert result.tokens_after < result.tokens_before
    assert len(result.deleted) >= 1
    # a deleted span must be a real adjunct, not a core argument
    assert all(d.deprel not in {"nsubj", "dobj", "ROOT"} for d in result.deleted)


def test_protected_amount_is_never_deleted(tmp_path):
    text = (
        "The company shall reimburse expenses of $2,500 within thirty days, "
        "and the vendor delivered the goods in the morning after a long delay."
    )
    result = compress(text, target_ratio=0.5, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert "$2,500" in result.compressed          # money survives
    assert "shall" in result.compressed           # modal survives
    assert any(v.protect_class in {"number", "modal"} for v in result.vetoed)


def test_passthrough_still_byte_identical_with_deletion(tmp_path):
    text = (
        "The committee approved the budget in the morning after long debate.\n"
        "```\ncode = in order to stay\n```\n"
    )
    result = compress(text, target_ratio=0.6, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert "code = in order to stay" in result.compressed  # fence untouched


def test_result_stamps_protect_version(tmp_path):
    result = compress("The committee approved the budget in the morning.",
                      min_tokens=0, cache_path=tmp_path / "c.sqlite")
    assert result.protect_list_version.startswith("protect-")
