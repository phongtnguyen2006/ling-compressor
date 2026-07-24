from promptcomp.parse import parse, load_nlp


def test_parse_splits_sentences_and_has_root():
    doc = parse("The cat sat. The dog ran fast.")
    sents = list(doc.sents)
    assert len(sents) == 2
    assert any(t.dep_ == "ROOT" for t in doc)


def test_load_nlp_is_cached_singleton():
    assert load_nlp() is load_nlp()


def test_parse_preserves_text():
    text = "She quickly finished the difficult task."
    doc = parse(text)
    assert doc.text == text
