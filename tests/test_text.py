from avatar_pipeline.text import chunk_text, estimate_speech_seconds, normalize_script, split_sentences


def test_normalize_collapses_whitespace_and_keeps_paragraphs():
    text = "Hello   world.\r\nSecond line.\n\n\nNew paragraph.  "
    assert normalize_script(text) == "Hello world. Second line.\n\nNew paragraph."


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_chunks_respect_limit_and_paragraphs():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta.\n\nIota kappa."
    chunks = chunk_text(text, max_chars=40)
    assert chunks == ["Alpha beta gamma. Delta epsilon zeta.", "Eta theta.", "Iota kappa."]
    assert all(len(c) <= 40 for c in chunks)


def test_overlong_sentence_is_split_on_clauses_then_words():
    sentence = "word " * 30 + "and, " + "more " * 30
    chunks = chunk_text(sentence.strip(), max_chars=60)
    assert len(chunks) > 2
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_empty_script_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_estimate_speech_seconds():
    assert estimate_speech_seconds("one two three four five", words_per_second=2.5) == 2.0
    assert estimate_speech_seconds("") == 1.0
