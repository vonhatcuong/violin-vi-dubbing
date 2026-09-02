from pipeline import vi_text


def test_lowercase_and_nfc():
    # "Chào" written with combining accent must collapse to precomposed lowercase
    # Using C-h-a + U+0300 combining grave + o
    decomposed = "Chào"
    assert vi_text.normalize_for_tts(f"Xin {decomposed} bạn", use_vinorm=False) == "xin chào bạn"


def test_numbers_become_vietnamese_words():
    out = vi_text.normalize_for_tts("có 23 người", use_vinorm=False)
    assert out == "có hai mươi ba người"


def test_percent_and_decimal():
    out = vi_text.normalize_for_tts("tăng 2.5%", use_vinorm=False)
    assert out == "tăng hai phẩy năm mươi phần trăm"


def test_loanword_map_is_case_insensitive_and_word_bounded():
    out = vi_text.normalize_for_tts("dùng GPU và gpus", use_vinorm=False, loanwords={"GPU": "gi pi u"})
    assert out == "dùng gi pi u và gpus"


def test_unsupported_symbols_are_dropped():
    out = vi_text.normalize_for_tts("a → b \"c\"", use_vinorm=False)
    assert "→" not in out and "\"" not in out
    assert out == "a b c"


def test_count_syllables_after_normalization():
    assert vi_text.count_syllables("Xin chào các bạn, 2 người!") == 6
