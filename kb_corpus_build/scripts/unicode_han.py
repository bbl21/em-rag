"""Dependency-free Unicode Script=Han detection shared by active RAG paths."""

from __future__ import annotations

from typing import Any


UNICODE_HAN_DATA_VERSION = "17.0.0"

# Copied from the Script=Han records in Unicode 17.0.0 UCD Scripts.txt:
# https://www.unicode.org/Public/17.0.0/ucd/Scripts.txt
# ScriptExtensions.txt is intentionally not unioned here:
# https://www.unicode.org/Public/17.0.0/ucd/ScriptExtensions.txt
# Characters that only have Han in Script_Extensions are Common/Inherited
# punctuation or symbols usable beside several scripts; treating them as Han
# would wrongly reject punctuation-only English input (for example U+3001).
HAN_SCRIPT_RANGES = (
    (0x2E80, 0x2E99),
    (0x2E9B, 0x2EF3),
    (0x2F00, 0x2FD5),
    (0x3005, 0x3005),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303B),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x16FE2, 0x16FE3),
    (0x16FF0, 0x16FF6),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B81D),
    (0x2B820, 0x2CEAD),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x33479),
)


def is_han_codepoint(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in HAN_SCRIPT_RANGES)


def contains_han(value: Any) -> bool:
    return any(is_han_codepoint(ord(character)) for character in str(value or ""))
