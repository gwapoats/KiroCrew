"""The native speech fixtures select Samantha across macOS display languages."""

import pytest
from test_apple_speech import _en_us_say_voice


@pytest.mark.parametrize(
    "name",
    ["Samantha", "Samantha (English (US))", "Samantha (英语（美国）)"],
)
def test_preferred_voice_keeps_its_full_localized_name(name):
    listing = f"Albert en_US # Hello\n{name} en_US # Hello\n"
    assert _en_us_say_voice(listing) == name


def test_preferred_voice_locale_and_name_are_both_checked():
    listing = (
        "Samantha de_DE # Hallo\n"
        "Samantha Else en_US # Hello\n"
        "Samantha (英语（美国）) en_US # Hello\n"
        "Samantha (英语（美国）) en_US # Hello\n"
    )
    assert _en_us_say_voice(listing) == "Samantha (英语（美国）)"
