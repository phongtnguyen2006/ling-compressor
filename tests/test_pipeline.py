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
    result = compress(
        text, substitute=False, min_tokens=0, cache_path=tmp_path / "c.sqlite"
    )
    assert result.compressed == text


def test_determinism_same_input_same_output(tmp_path):
    text = "We proceeded in order to finish the report on time today."
    r1 = compress(text, min_tokens=0, cache_path=tmp_path / "c1.sqlite")
    r2 = compress(text, min_tokens=0, cache_path=tmp_path / "c2.sqlite")
    assert r1.compressed == r2.compressed
    assert r1.config_hash == r2.config_hash
