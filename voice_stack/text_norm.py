"""Spoken-form text normalization for the coding-voice bridge.

The Kyutai TTS model gets raw text with no locale-aware number parsing:
"3.14159", "0.0001", "2.4 percent" are tokenised as-is and the model has
to guess the pronunciation, which it does badly (decimal points came out
completely garbled). This module rewrites number tokens into their British
English spoken form *before* the text reaches the TTS engine:

    3.14159        -> three point one four one five nine
    0.0001         -> zero point zero zero zero one
    9.95           -> nine point nine five
    2.4 percent    -> two point four percent
    100%           -> one hundred percent
    -3.5           -> minus three point five
    2025           -> twenty twenty-five          (years)
    1996           -> nineteen ninety-six
    1,234,567      -> one million two hundred and thirty-four thousand five hundred and sixty-seven
    09727159      -> zero nine seven two seven one five nine   (long digit strings)

Only the audio path is rewritten: the bridge keeps the original text for
bookkeeping (spoken_text snapshot alignment, logs), so this must stay a
pure function of the sentence text.
"""

from __future__ import annotations

import re

from num2words import num2words

_DIGITS = "zero one two three four five six seven eight nine"
_DIGITS_L = _DIGITS.split()

# A number token: optional sign, grouped (1,234) or plain (123), optional
# decimal tail, optional dotted chain (version-like: 1.5.5). A number is not
# normalized when glued to letters/digits on either side (v1.5, int8, sm_120);
# a leading dot is blocked too, so mid-chain digits of 1.5.5 are never taken
# from the middle (the match starts at the first digit and runs to the end).
_NUM_RE = re.compile(r"""
    (?<![A-Za-z0-9_.])
    ([-+])?
    ( \d{1,3}(?:,\d{3})+(?:\.\d+)?      # comma-grouped: 1,234,567 / 1,234.5
      | \d+(?:\.\d+)+                    # dotted: 3.14159, 1.5.5
      | \d+ )                            # plain: 42
    (?![A-Za-z0-9_])                     # ...but a trailing % is allowed
    (\s*%)?
""", re.VERBOSE)


def _number_words(n: int) -> str:
    """Integer in words, British style. Capped at a billion; beyond that
    (IDs, hashes) callers read digit by digit instead."""
    if n < 0:
        return "minus " + _number_words(-n)
    if n > 999_999_999:
        return _digit_by_digit(str(n))
    return num2words(n, lang="en").replace(",", "")


def _digit_by_digit(s: str) -> str:
    return " ".join(_DIGITS_L[int(c)] for c in s)


def _year_words(n: int) -> str:
    if n == 2000:
        return "two thousand"
    if 2001 <= n <= 2099:
        return "twenty " + _number_words(n % 100).strip()
    if 1900 <= n <= 1999:
        if n % 100 == 0:
            return "nineteen hundred"
        return "nineteen " + _number_words(n % 100).strip()
    return _number_words(n)  # 1000-1899, 2100+: plain reading


def _decimal_words(int_part: str, frac_parts: list[str]) -> str:
    """'3' + ['14159'] -> 'three point one four one five nine';
    '1' + ['5', '5'] -> 'one point five point five' (version chains)."""
    if len(frac_parts) > 1:
        # version-like 1.5.5: every group digit by digit
        parts = [_digit_by_digit(int_part.lstrip("0") or "0")] + \
                [_digit_by_digit(p) for p in frac_parts]
        return " point ".join(parts)
    frac = frac_parts[0]
    ip = int_part.lstrip("0")
    if not ip:
        whole = "zero"
    elif len(ip) >= 7:
        whole = _digit_by_digit(ip)  # huge integer part: don't read it as a sum
    else:
        whole = _number_words(int(ip))
    return f"{whole} point {_digit_by_digit(frac)}"


def normalize_speech(text: str) -> str:
    """Rewrite number tokens in `text` into spoken British English form.

    Pure and total: any input returns a string; on any internal error the
    original text is returned unchanged (the voice must never fail to speak
    because of a normalizer bug).
    """
    if not text:
        return text
    try:
        return _normalize(text)
    except Exception:
        return text


def _normalize(text: str) -> str:
    def repl(m: re.Match) -> str:
        sign, num, percent = m.group(1), m.group(2), m.group(3)
        prefix = {"-": "minus ", "+": "plus "}.get(sign or "", "")

        if "." in num:
            groups = num.split(".")
            words = _decimal_words(groups[0], groups[1:])
        elif "," in num:
            words = _number_words(int(num.replace(",", "")))
        else:
            n = int(num)
            if len(num) == 4 and 1000 <= n <= 2999:
                words = _year_words(n)
            elif len(num) <= 5:
                words = _number_words(n)
            else:
                words = _digit_by_digit(num)  # 6+ digits: an ID, not a quantity

        if percent:
            words += " percent"
        return prefix + words

    return _NUM_RE.sub(repl, text)
