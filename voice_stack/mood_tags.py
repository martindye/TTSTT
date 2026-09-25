"""Mood tags: zero-width voice markers the agent writes in its visible text.

The coding agent (the LLM) knows best which part of a reply carries which
feeling, so it marks mood shifts inline with double-bracket markers::

    The build is [[contentment]]green again -- that one was [[relief]]fixed.

The bridge (coding_voice) strips the markers and switches the Pro voice's
mood at each one, so tagged text needs no word-scanning at all: the model
has already decided, the bridge just follows. A marker is zero-width and
forward-looking: it changes the voice from that point on, until the next
marker (or the end of the text). A reply starts in ``neutral``; untagged
text keeps the current mood. A reply with no markers at all is spoken the
old way (one mood picked for the whole reply by ``coding_voice.pick_mood``).
"""

from __future__ import annotations

import re

from .kyutai_tts_engine import MOODS  # canonical EARS mood names

__all__ = [
    "MOOD_TAG_RE",
    "canon_mood",
    "mood_segs",
    "has_mood_tags",
    "count_mood_tags",
]

# A single word between double brackets. Because a mood word contains no
# punctuation, split_sentence can never cut a tag in half: a boundary
# (". " / "! " / "? ") always falls between tags, and a tag split across
# chunk boundaries reassembles in the sentence buffer before parsing.
MOOD_TAG_RE = re.compile(r"\[\[([A-Za-z][A-Za-z0-9_]*)\]\]")

# Words the model is likely to reach for instead of the canonical EARS
# names, mapped onto them. Everything unknown falls back to neutral (a
# mistyped tag must degrade to the plain voice, never a wrong emotion).
_MOOD_ALIASES = {
    # anger
    "angry": "anger", "furious": "anger", "mad": "anger", "fury": "anger",
    "irritated": "anger", "annoyed": "anger", "irritation": "anger",
    "enraged": "anger", "seething": "anger", "indignant": "anger",
    "jealous": "anger", "jealousy": "anger",
    # contentment
    "happy": "contentment", "glad": "contentment", "content": "contentment",
    "pleased": "contentment", "satisfied": "contentment",
    "warm": "contentment", "cozy": "contentment",
    "grateful": "contentment", "thankful": "contentment",
    "hopeful": "contentment", "reassured": "contentment",
    # extasy
    "ecstatic": "extasy", "thrilled": "extasy", "elated": "extasy",
    "delighted": "extasy", "joy": "extasy", "joyful": "extasy",
    "overjoyed": "extasy", "euphoric": "extasy",
    # amazement
    "amazed": "amazement", "surprised": "amazement",
    "shocked": "amazement", "astonished": "amazement",
    "wonder": "amazement", "wow": "amazement",
    "awed": "amazement", "awe": "amazement",
    # amusement
    "funny": "amusement", "amused": "amusement", "joking": "amusement",
    "playful": "amusement", "teasing": "amusement",
    "cheeky": "amusement", "witty": "amusement",
    "mischievous": "amusement", "chuckling": "amusement",
    "silly": "amusement", "humorous": "amusement",
    # confusion
    "confused": "confusion", "puzzled": "confusion",
    "baffled": "confusion", "perplexed": "confusion",
    "bewildered": "confusion", "muddled": "confusion",
    # sadness
    "sad": "sadness", "unhappy": "sadness",
    "sorrowful": "sadness", "sorrow": "sadness",
    "melancholy": "sadness", "gloomy": "sadness",
    "heartbroken": "sadness", "disheartened": "sadness",
    # fear
    "scared": "fear", "afraid": "fear", "frightened": "fear",
    "terrified": "fear", "nervous": "fear",
    "spooked": "fear", "scary": "fear",
    # distress
    "distressed": "distress", "panicked": "distress",
    "panicking": "distress", "panic": "distress",
    "desperate": "distress", "despairing": "distress",
    "hopeless": "distress", "anxious": "distress",
    "anxiety": "distress", "worried": "distress",
    "stressed": "distress", "overwhelmed": "distress",
    "upset": "distress", "crying": "distress",
    "frustrated": "distress", "frustrating": "distress",
    "frazzled": "distress",
    # interest
    "curious": "interest", "intrigued": "interest",
    "interested": "interest", "fascinated": "interest",
    "eager": "interest", "attentive": "interest", "engaged": "interest",
    # pride
    "proud": "pride", "accomplished": "pride",
    "triumphant": "pride", "smug": "pride",
    # adoration
    "loving": "adoration", "adoring": "adoration",
    "in_love": "adoration", "inlove": "adoration",
    "infatuated": "adoration", "devoted": "adoration",
    "affection": "adoration",
    # cuteness
    "cute": "cuteness", "adorable": "cuteness",
    "sweet": "cuteness", "precious": "cuteness",
    # desire
    "wanting": "desire", "longing": "desire",
    "craving": "desire", "hungry": "desire",
    "thirsty": "desire", "eagerness": "desire",
    # disappointment
    "disappointed": "disappointment", "letdown": "disappointment",
    "let_down": "disappointment", "deflated": "disappointment",
    # disgust
    "disgusted": "disgust", "repulsed": "disgust",
    "sickened": "disgust", "grossed": "disgust",
    "revolted": "disgust", "icky": "disgust", "loathe": "disgust",
    # embarassment (the EARS spelling, kept)
    "embarrassed": "embarassment", "embarrassing": "embarassment",
    "cringe": "embarassment", "awkward": "embarassment",
    "sheepish": "embarassment",
    # pain
    "painful": "pain", "hurting": "pain", "hurts": "pain",
    "ouch": "pain", "aching": "pain",
    # guilt
    "guilty": "guilt", "remorseful": "guilt",
    "remorse": "guilt", "ashamed": "guilt",
    # relief
    "relieved": "relief",
    # serenity
    "calm": "serenity", "peaceful": "serenity",
    "relaxed": "serenity", "at_ease": "serenity",
    "mellow": "serenity", "serene": "serenity",
    "bored": "serenity", "nostalgic": "serenity",
    "sympathetic": "serenity", "compassionate": "serenity",
    # realization
    "realized": "realization", "realised": "realization",
    "dawned": "realization", "ah_ha": "realization",
    "epiphany": "realization", "understood": "realization",
}


def canon_mood(name: str) -> str:
    """Normalise a tag's word to a canonical EARS mood name.

    Unknown names map to ``neutral``: a mistyped tag degrades to the
    plain voice instead of a wrong emotion.
    """
    n = " ".join((name or "").lower().split())
    n = n.replace("-", "_")
    if n in MOODS:
        return n
    return _MOOD_ALIASES.get(n, "neutral")


def mood_segs(text: str):
    """Split text into ``(mood, plain_text)`` segments around mood tags.

    A tag is a zero-width, forward-looking marker: the segment AFTER it is
    spoken in the tagged mood until the next tag (or the end of the text).
    Text before the first tag carries mood ``None`` ("keep whatever mood
    is current"). Returns ``(segments, last_tag)`` where segments is a
    list of ``(mood | None, text)`` and last_tag is the canonical mood of
    the last tag seen (``None`` when the text carries no tag at all).
    """
    segs = []
    pos, cur = 0, None
    last = None
    for m in MOOD_TAG_RE.finditer(text):
        if m.start() > pos:
            segs.append((cur, text[pos:m.start()]))
        pos = m.end()
        last = canon_mood(m.group(1))
        cur = last
    if pos < len(text):
        segs.append((cur, text[pos:]))
    return segs, last


def has_mood_tags(text: str) -> bool:
    """True when the text carries at least one [[mood]] marker."""
    return bool(MOOD_TAG_RE.search(text or ""))


def count_mood_tags(text: str) -> int:
    """Number of [[mood]] markers in the text."""
    return len(MOOD_TAG_RE.findall(text or ""))
