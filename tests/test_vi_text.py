import unicodedata

from pipeline import vi_text


def test_lowercase_and_nfc():
    # "Chào" written with combining accent (C-h-a + U+0300 combining grave + o) must collapse to precomposed lowercase
    decomposed = "Cha" + "\u0300" + "o"  # Explicitly constructed with combining grave U+0300
    assert unicodedata.is_normalized("NFC", decomposed) is False, "Input must be decomposed to test NFC normalization"
    assert vi_text.normalize_for_tts(f"Xin {decomposed} bạn", use_vinorm=False) == "xin chào bạn"


def test_numbers_become_vietnamese_words():
    out = vi_text.normalize_for_tts("có 23 người", use_vinorm=False)
    assert out == "có hai mươi ba người"


def test_decimal_digit_by_digit():
    out = vi_text.normalize_for_tts("tăng 2.5%", use_vinorm=False)
    assert out == "tăng hai phẩy năm phần trăm"


def test_decimal_with_period_three_digit_groups():
    out = vi_text.normalize_for_tts("3.14", use_vinorm=False)
    assert out == "ba phẩy một bốn"


def test_decimal_starting_with_zero():
    out = vi_text.normalize_for_tts("0.75", use_vinorm=False)
    assert out == "không phẩy bảy năm"


def test_thousands_with_comma():
    out = vi_text.normalize_for_tts("1,000 người", use_vinorm=False)
    assert out == "một nghìn người"


def test_thousands_with_period():
    out = vi_text.normalize_for_tts("1.000.000 đồng", use_vinorm=False)
    assert out == "một triệu đồng"


def test_thousands_integer():
    out = vi_text.normalize_for_tts("2.500", use_vinorm=False)
    assert out == "hai nghìn năm trăm"


def test_negative_number():
    out = vi_text.normalize_for_tts("giảm -5 độ", use_vinorm=False)
    assert out == "giảm âm năm độ"


def test_negative_decimal():
    out = vi_text.normalize_for_tts("-2.5%", use_vinorm=False)
    assert out == "âm hai phẩy năm phần trăm"


def test_hyphenated_word_untouched():
    out = vi_text.normalize_for_tts("mạng nơ-ron", use_vinorm=False)
    assert out == "mạng nơ-ron"


def test_loanword_map_is_case_insensitive_and_word_bounded():
    out = vi_text.normalize_for_tts("dùng GPU và gpus", use_vinorm=False, loanwords={"GPU": "gi pi u"})
    assert out == "dùng gi pi u và gpus"


def test_unsupported_symbols_are_dropped():
    out = vi_text.normalize_for_tts("a → b " + chr(0x201c) + "c" + chr(0x201d) + "", use_vinorm=False)
    assert "→" not in out and chr(0x201c) not in out
    assert out == "a b c"


def test_count_syllables_after_normalization():
    assert vi_text.count_syllables("Xin chào các bạn, 2 người!") == 6
